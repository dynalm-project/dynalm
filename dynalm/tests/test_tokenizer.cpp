#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "chat_template/chat_template.h"
#include "loader/gguf/gguf.h"
#include "test_models.h"
#include "loader/gguf/gguf_tokenizer.h"
#include "tokenizer/pretokenizer.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/unicode.h"

namespace engine {
namespace {

// ---------------------------------------------------------------------------
// Unicode

TEST(Unicode, Categories) {
  using namespace unicode;
  EXPECT_TRUE(is_letter('a'));
  EXPECT_TRUE(is_letter(0x00E9));   // é
  EXPECT_TRUE(is_letter(0x4E2D));   // 中
  EXPECT_FALSE(is_letter('1'));
  EXPECT_TRUE(is_number('7'));
  EXPECT_TRUE(is_number(0x00B2));   // ² (No)
  EXPECT_TRUE(is_number(0x0663));   // Arabic-Indic 3
  EXPECT_TRUE(is_whitespace(' '));
  EXPECT_TRUE(is_whitespace(0x00A0));
  EXPECT_TRUE(is_whitespace(0x3000));
  EXPECT_FALSE(is_whitespace(0x200B));  // zero-width space is not White_Space
  EXPECT_FALSE(is_letter(0x1F600));     // emoji
}

TEST(Unicode, Utf8DecodeIsTotal) {
  // Every byte consumed exactly once, invalid bytes pass through.
  const std::string s = "a\xC3\xA9\xFF\xE4\xB8\xAD\xF0\x9F\x98\x80\xC3";
  std::vector<uint32_t> cps;
  for (size_t i = 0; i < s.size();) cps.push_back(unicode::decode_utf8(s, i));
  EXPECT_EQ(cps, (std::vector<uint32_t>{'a', 0xE9, 0xFF, 0x4E2D, 0x1F600, 0xC3}));
  std::string round;
  for (uint32_t cp : {0x41u, 0xE9u, 0x4E2Du, 0x1F600u}) unicode::append_utf8(cp, round);
  EXPECT_EQ(round, "A\xC3\xA9\xE4\xB8\xAD\xF0\x9F\x98\x80");
}

// ---------------------------------------------------------------------------
// Pre-tokenizers (expected splits derived by hand from the source regexes)

std::vector<std::string> split(PreTokenizer k, std::string_view text) {
  std::vector<std::string_view> v;
  pretokenize(k, text, v);
  return {v.begin(), v.end()};
}

using Words = std::vector<std::string>;

TEST(PreTokenizer, Gpt2) {
  EXPECT_EQ(split(PreTokenizer::kGpt2, "Hello world"), (Words{"Hello", " world"}));
  EXPECT_EQ(split(PreTokenizer::kGpt2, "I'm OK"), (Words{"I", "'m", " OK"}));
  EXPECT_EQ(split(PreTokenizer::kGpt2, "a  b"), (Words{"a", " ", " b"}));
  EXPECT_EQ(split(PreTokenizer::kGpt2, "x   "), (Words{"x", "   "}));
  EXPECT_EQ(split(PreTokenizer::kGpt2, "12345 abc"), (Words{"12345", " abc"}));
  EXPECT_EQ(split(PreTokenizer::kGpt2, "hi!!\nok"), (Words{"hi", "!!", "\n", "ok"}));
  EXPECT_EQ(split(PreTokenizer::kGpt2, "THEY'LL"), (Words{"THEY", "'", "LL"}));  // case-sensitive
}

TEST(PreTokenizer, Llama3) {
  EXPECT_EQ(split(PreTokenizer::kLlama3, "12345"), (Words{"123", "45"}));
  EXPECT_EQ(split(PreTokenizer::kLlama3, "THEY'LL"), (Words{"THEY", "'LL"}));  // case-insensitive
  EXPECT_EQ(split(PreTokenizer::kLlama3, "a\tb"), (Words{"a", "\tb"}));
  EXPECT_EQ(split(PreTokenizer::kLlama3, "hi.\n\nok"), (Words{"hi", ".\n\n", "ok"}));
  EXPECT_EQ(split(PreTokenizer::kLlama3, "a  \n  b"), (Words{"a", "  \n", " ", " b"}));
  EXPECT_EQ(split(PreTokenizer::kQwen2, "123"), (Words{"1", "2", "3"}));
}

TEST(PreTokenizer, StarCoderSplitsDigitsFirst) {
  EXPECT_EQ(split(PreTokenizer::kStarCoder, "ab12 c"), (Words{"ab", "1", "2", " c"}));
  // Whitespace before a digit is end-of-segment for the GPT-2 stage.
  EXPECT_EQ(split(PreTokenizer::kStarCoder, "x  5"), (Words{"x", "  ", "5"}));
}

TEST(PreTokenizer, CoversEveryByte) {
  const std::string text = "Mixed 中文 😀 \xFF\xFE bytes  \n\t 42!";
  for (PreTokenizer k : {PreTokenizer::kGpt2, PreTokenizer::kLlama3, PreTokenizer::kQwen2,
                         PreTokenizer::kStarCoder}) {
    std::string joined;
    for (const auto& w : split(k, text)) joined += w;
    EXPECT_EQ(joined, text);
  }
}

TEST(PreTokenizer, UnknownNameIsUnsupported) {
  EXPECT_EQ(pretokenizer_from_name("made-up").status().code(), StatusCode::kUnsupported);
  EXPECT_EQ(*pretokenizer_from_name("smollm"), PreTokenizer::kStarCoder);
}

// ---------------------------------------------------------------------------
// SPM on a synthetic vocabulary

TokenizerData tiny_spm() {
  TokenizerData d;
  d.model = TokenizerData::Model::kSpm;
  d.tokens = {"<unk>", "<s>", "</s>", "\xE2\x96\x81", "h", "e", "l", "o", "\xE2\x96\x81h", "ll", "\xE2\x96\x81he",
              "llo", "\xE2\x96\x81hello"};
  d.scores = {0, 0, 0, -1, -2, -2, -2, -2, -1, -1, -0.5f, -0.4f, -0.1f};
  d.types = std::vector<TokenType>(d.tokens.size(), TokenType::kNormal);
  d.types[0] = TokenType::kUnknown;
  d.types[1] = d.types[2] = TokenType::kControl;
  // byte fallback tokens
  for (int b = 0; b < 256; ++b) {
    char name[8];
    std::snprintf(name, sizeof(name), "<0x%02X>", b);
    d.tokens.push_back(name);
    d.scores.push_back(0);
    d.types.push_back(TokenType::kByte);
  }
  d.unk = 0;
  d.bos = 1;
  d.eos = 2;
  d.add_bos = true;
  return d;
}

TEST(SpmTokenizer, MergesByScoreAndByteFallback) {
  auto t = Tokenizer::create(tiny_spm());
  ASSERT_TRUE(t.ok()) << t.status().to_string();
  const auto ids = (*t)->encode("hello");
  // BOS + "▁hello"
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_EQ(ids[0], 1);
  EXPECT_EQ((*t)->token_text(ids[1]), "\xE2\x96\x81hello");
  EXPECT_EQ((*t)->decode(ids), "hello");

  // 'z' is not in the vocab: byte fallback <0x7A>.
  const auto z = (*t)->encode("hz", false);
  ASSERT_EQ(z.size(), 2u);
  EXPECT_EQ((*t)->token_text(z[0]), "\xE2\x96\x81h");
  EXPECT_EQ((*t)->token_text(z[1]), "<0x7A>");
  EXPECT_EQ((*t)->decode(z), "hz");
}

TEST(SpmTokenizer, SpecialTokensOnlyWhenParsing) {
  auto t = Tokenizer::create(tiny_spm());
  ASSERT_TRUE(t.ok());
  const auto parsed = (*t)->encode("hello</s>", false, true);
  EXPECT_EQ(parsed.back(), 2);
  const auto raw = (*t)->encode("hello</s>", false, false);
  EXPECT_NE(raw.back(), 2);
  EXPECT_TRUE((*t)->is_eog(2));
}

TEST(Tokenizer, RejectsBadMetadata) {
  auto d = tiny_spm();
  d.eos = 100000;
  EXPECT_EQ(Tokenizer::create(d).status().code(), StatusCode::kCorrupt);
  d = tiny_spm();
  d.types.pop_back();
  EXPECT_FALSE(Tokenizer::create(d).ok());
}

TEST(Utf8Buffer, HoldsIncompleteSequences) {
  Utf8Buffer buf;
  std::string out;
  buf.push("a\xE4", out);  // 'a' + first byte of 中
  EXPECT_EQ(out, "a");
  buf.push("\xB8", out);
  EXPECT_EQ(out, "a");
  buf.push("\xAD!", out);
  EXPECT_EQ(out, "a\xE4\xB8\xAD!");
  buf.push("\xF0\x9F", out);
  buf.flush(out);
  EXPECT_EQ(out, "a\xE4\xB8\xAD!\xF0\x9F");
}

// ---------------------------------------------------------------------------
// Chat templates

TEST(ChatTemplate, ChatMlWithDefaultSystem) {
  const std::string jinja =
      "{%- if messages[0]['role'] == 'system' %}...{%- else %}"
      "{{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}{%- endif %}"
      "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}";
  auto t = ChatTemplate::from_jinja(jinja);
  ASSERT_TRUE(t.ok()) << t.status().to_string();
  EXPECT_EQ(t->format(), ChatFormat::kChatMl);
  EXPECT_EQ(t->default_system(), "You are a helpful assistant.");
  const ChatMessage msgs[] = {{"user", "Hi"}};
  auto s = t->apply(msgs, true);
  ASSERT_TRUE(s.ok());
  EXPECT_EQ(*s,
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
  // Explicit system replaces the default.
  const ChatMessage with_sys[] = {{"system", "Be terse."}, {"user", "Hi"}};
  EXPECT_EQ(*t->apply(with_sys, false),
            "<|im_start|>system\nBe terse.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n");
}

TEST(ChatTemplate, Llama3AndGemma) {
  const ChatMessage msgs[] = {{"system", "S"}, {"user", "U"}, {"assistant", "A"}, {"user", "V"}};
  EXPECT_EQ(*ChatTemplate(ChatFormat::kLlama3).apply(msgs, true),
            "<|start_header_id|>system<|end_header_id|>\n\nS<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\nU<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\nA<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\nV<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n");
  EXPECT_EQ(*ChatTemplate(ChatFormat::kGemma).apply(msgs, true),
            "<start_of_turn>user\nS\n\nU<end_of_turn>\n<start_of_turn>model\nA<end_of_turn>\n"
            "<start_of_turn>user\nV<end_of_turn>\n<start_of_turn>model\n");
}

TEST(ChatTemplate, GraniteMatchesJinjaRendering) {
  // tests/data/granite31_chat_template.jinja is the template embedded in
  // granite-3.1-1b-a400m-instruct; expected strings were rendered from it with
  // jinja2 (default system prompt injected; a space separates turns).
  std::ifstream in(std::string(ENGINE_TEST_DATA_DIR) + "/granite31_chat_template.jinja", std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  auto t = ChatTemplate::from_jinja(ss.str());
  ASSERT_TRUE(t.ok());
  EXPECT_EQ(t->format(), ChatFormat::kGranite);
  const ChatMessage one[] = {{"user", "Hi there"}};
  EXPECT_EQ(*t->apply(one, true),
            "<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024. You are Granite, developed by IBM. "
            "You are a helpful AI assistant.<|end_of_text|> <|start_of_role|>user<|end_of_role|>Hi there<|end_of_text|> "
            "<|start_of_role|>assistant<|end_of_role|>");
  const ChatMessage conv[] = {{"system", "Be terse."}, {"user", "Hi"}, {"assistant", "Hello!"}, {"user", "Bye"}};
  EXPECT_EQ(*t->apply(conv, true),
            "<|start_of_role|>system<|end_of_role|>Be terse.<|end_of_text|> <|start_of_role|>user<|end_of_role|>Hi"
            "<|end_of_text|> <|start_of_role|>assistant<|end_of_role|>Hello!<|end_of_text|> "
            "<|start_of_role|>user<|end_of_role|>Bye<|end_of_text|> <|start_of_role|>assistant<|end_of_role|>");
}

TEST(ChatTemplate, DetectionAndErrors) {
  EXPECT_EQ(ChatTemplate::from_jinja("x<|start_header_id|>y")->format(), ChatFormat::kLlama3);
  EXPECT_EQ(ChatTemplate::from_jinja("<start_of_turn>")->format(), ChatFormat::kGemma);
  EXPECT_EQ(ChatTemplate::from_jinja("[INST] <<SYS>>")->format(), ChatFormat::kLlama2);
  EXPECT_EQ(ChatTemplate::from_jinja("{{ bos }}[INST]")->format(), ChatFormat::kMistral);
  EXPECT_EQ(ChatTemplate::from_jinja("no idea").status().code(), StatusCode::kUnsupported);
  const ChatMessage bad[] = {{"tool", "x"}};
  EXPECT_FALSE(ChatTemplate(ChatFormat::kChatMl).apply(bad, false).ok());
  EXPECT_EQ(*parse_chat_format("phi3"), ChatFormat::kPhi3);
}

// ---------------------------------------------------------------------------
// Golden tests against HF reference tokenizers (tools/gen_tokenizer_golden.py)

struct GoldenCase {
  std::string text;
  std::vector<TokenId> ids;
};

std::vector<GoldenCase> load_golden(const std::string& path) {
  std::vector<GoldenCase> cases;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    const auto tab = line.find('\t');
    if (tab == std::string::npos) continue;
    GoldenCase c;
    for (size_t i = 0; i + 1 < tab; i += 2) {
      c.text += static_cast<char>(std::stoi(line.substr(i, 2), nullptr, 16));
    }
    std::istringstream ids(line.substr(tab + 1));
    for (TokenId id; ids >> id;) c.ids.push_back(id);
    cases.push_back(std::move(c));
  }
  return cases;
}

std::unique_ptr<Tokenizer> load_model_tokenizer(const std::string& path) {
  if (!engine::testing::exists(path)) return nullptr;
  auto g = gguf::GgufFile::open(path);
  if (!g.ok()) return nullptr;
  auto data = gguf::read_tokenizer_data(**g);
  EXPECT_TRUE(data.ok()) << data.status().to_string();
  if (!data.ok()) return nullptr;
  auto t = Tokenizer::create(std::move(*data));
  EXPECT_TRUE(t.ok()) << t.status().to_string();
  return t.ok() ? std::move(*t) : nullptr;
}

void check_golden(const std::string& model, const std::string& golden_file) {
  auto tok = load_model_tokenizer(model);
  if (!tok) GTEST_SKIP() << model << " not present";
  const auto cases = load_golden(std::string(ENGINE_TEST_DATA_DIR) + "/" + golden_file);
  ASSERT_FALSE(cases.empty());
  int mismatches = 0;
  for (const auto& c : cases) {
    // HF parses special-token text in inputs; so do we here.
    const auto ids = tok->encode(c.text, /*add_special=*/false, /*parse_special=*/true);
    if (ids != c.ids) {
      ++mismatches;
      ADD_FAILURE() << "text: \"" << c.text << "\"\n  got " << ::testing::PrintToString(ids)
                    << "\n want " << ::testing::PrintToString(c.ids);
    }
    EXPECT_EQ(tok->decode(ids, /*render_special=*/true), c.text);
  }
  EXPECT_EQ(mismatches, 0);
}

TEST(TokenizerGolden, SmolLm2) { check_golden(engine::testing::smollm_model(), "golden_smollm2.txt"); }
TEST(TokenizerGolden, Qwen25) { check_golden(engine::testing::qwen_model(), "golden_qwen25.txt"); }
TEST(TokenizerGolden, Gemma3Spm) { check_golden(engine::testing::gemma_model(), "golden_gemma3.txt"); }

TEST(TokenizerGolden, SmolLm2ChatTemplate) {
  const std::string path = engine::testing::smollm_model();
  if (!engine::testing::exists(path)) GTEST_SKIP();
  auto g = gguf::GgufFile::open(path);
  ASSERT_TRUE(g.ok());
  auto tmpl = ChatTemplate::from_jinja(gguf::read_chat_template(**g));
  ASSERT_TRUE(tmpl.ok()) << tmpl.status().to_string();
  EXPECT_EQ(tmpl->format(), ChatFormat::kChatMl);
  auto tok = load_model_tokenizer(path);
  ASSERT_NE(tok, nullptr);
  const ChatMessage msgs[] = {{"user", "What is 2+2?"}};
  auto text = tmpl->apply(msgs, true);
  ASSERT_TRUE(text.ok());
  const auto ids = tok->encode(*text, false, true);
  // Control tokens are single ids, and the prompt ends with "assistant\n".
  EXPECT_EQ(tok->token_text(ids.front()), "<|im_start|>");
  EXPECT_TRUE(tok->is_eog(tok->find_token("<|im_end|>")));
}

}  // namespace
}  // namespace engine

namespace engine {
namespace {

TEST(TokenizerGolden, Qwen25ChatTemplateDefaultSystem) {
  const std::string path = engine::testing::qwen_model();
  if (!engine::testing::exists(path)) GTEST_SKIP();
  auto g = gguf::GgufFile::open(path);
  if (!g.ok()) GTEST_SKIP() << g.status().to_string();
  auto tmpl = ChatTemplate::from_jinja(gguf::read_chat_template(**g));
  ASSERT_TRUE(tmpl.ok()) << tmpl.status().to_string();
  EXPECT_EQ(tmpl->format(), ChatFormat::kChatMl);
  EXPECT_EQ(tmpl->default_system(), "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.");
}

}  // namespace
}  // namespace engine
