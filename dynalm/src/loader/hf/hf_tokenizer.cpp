#include "loader/hf/hf_tokenizer.h"

#include <algorithm>
#include <unordered_map>
#include <vector>
#include "common/core.h"

namespace dynalm::hf {
namespace {

// Pre-tokenizer split patterns, exactly as tokenizer.json stores them.
constexpr std::string_view kQwen2Regex =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
constexpr std::string_view kLlama3Regex =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

const json::Value* field(const json::Value* o, std::string_view key) {
  if (!o) return nullptr;
  const json::Value* v = o->find(key);
  return v && !v->is_null() ? v : nullptr;
}

std::string type_of(const json::Value* v) {
  const json::Value* t = field(v, "type");
  return t && t->is_string() ? t->as_string() : "";
}

bool flag(const json::Value* o, std::string_view key, bool fallback) {
  const json::Value* v = field(o, key);
  return v && v->is_bool() ? v->as_bool() : fallback;
}

// Components of a (possibly Sequence) normalizer / pre-tokenizer / decoder.
std::vector<const json::Value*> flatten(const json::Value* v, std::string_view list_key) {
  std::vector<const json::Value*> out;
  if (!v || !v->is_object()) return out;
  if (type_of(v) == "Sequence") {
    if (const json::Value* items = field(v, list_key); items && items->is_array()) {
      for (const json::Value& item : items->as_array()) {
        auto sub = flatten(&item, list_key);
        out.insert(out.end(), sub.begin(), sub.end());
      }
    }
    return out;
  }
  out.push_back(v);
  return out;
}

std::string pattern_of(const json::Value* split) {
  const json::Value* p = field(split, "pattern");
  if (const json::Value* r = field(p, "Regex"); r && r->is_string()) return r->as_string();
  if (const json::Value* s = field(p, "String"); s && s->is_string()) return s->as_string();
  return "";
}

// Token text of a tokenizer_config entry: "text" or {"content": "text"}.
std::string token_text(const json::Value* v) {
  if (!v) return "";
  if (v->is_string()) return v->as_string();
  if (const json::Value* c = field(v, "content"); c && c->is_string()) return c->as_string();
  return "";
}

// Byte-level BPE pre-tokenizer id (TokenizerData::pre) from tokenizer.json.
Result<std::string> byte_level_pre(const json::Value* pre) {
  const auto parts = flatten(pre, "pretokenizers");
  std::string summary;
  for (const json::Value* p : parts) summary += (summary.empty() ? "" : "+") + type_of(p);
  if (parts.size() == 1 && type_of(parts[0]) == "ByteLevel" && flag(parts[0], "use_regex", true)) return std::string("default");
  if (parts.size() == 2 && type_of(parts[0]) == "Digits" && flag(parts[0], "individual_digits", false) &&
      type_of(parts[1]) == "ByteLevel" && flag(parts[1], "use_regex", true)) {
    return std::string("smollm");
  }
  if (parts.size() == 2 && type_of(parts[0]) == "Split" && type_of(parts[1]) == "ByteLevel" &&
      !flag(parts[1], "use_regex", true)) {
    const std::string re = pattern_of(parts[0]);
    if (re == kQwen2Regex) return std::string("qwen2");
    if (re == kLlama3Regex) return std::string("llama-bpe");
  }
  return Unsupported("tokenizer.json pre-tokenizer (" + summary + ") is not one the engine implements yet");
}

}  // namespace

std::string read_chat_template(const json::Value* tc) {
  const json::Value* t = field(tc, "chat_template");
  if (!t) return "";
  if (t->is_string()) return t->as_string();
  if (t->is_array()) {  // [{"name": "default", "template": "..."}, ...]
    for (const json::Value& e : t->as_array()) {
      const json::Value* name = field(&e, "name");
      const json::Value* tmpl = field(&e, "template");
      if (name && name->is_string() && name->as_string() == "default" && tmpl && tmpl->is_string()) {
        return tmpl->as_string();
      }
    }
  }
  return "";
}

Result<TokenizerData> read_tokenizer(const TokenizerFiles& f) {
  const json::Value* tj = f.tokenizer;
  if (!tj || !tj->is_object()) return Corrupt("tokenizer.json is missing or not an object");
  const json::Value* model = field(tj, "model");
  if (type_of(model) != "BPE") {
    return Unsupported("tokenizer.json model type '" + type_of(model) + "' is not supported (BPE only)");
  }
  const json::Value* vocab = field(model, "vocab");
  if (!vocab || !vocab->is_object()) return Corrupt("tokenizer.json: model.vocab missing");

  TokenizerData d;
  // SentencePiece-style BPE: spaces become "▁" in the normalizer (or a
  // Metaspace pre-tokenizer) and unknown bytes fall back to <0xXX> pieces.
  bool sp_space = false, prepend = false;
  for (const json::Value* n : flatten(field(tj, "normalizer"), "normalizers")) {
    if (type_of(n) == "Replace" && pattern_of(n) == " ") sp_space = true;
    if (type_of(n) == "Prepend") prepend = true;
  }
  for (const json::Value* p : flatten(field(tj, "pre_tokenizer"), "pretokenizers")) {
    if (type_of(p) == "Metaspace") {
      sp_space = true;
      const json::Value* scheme = field(p, "prepend_scheme");
      prepend = scheme && scheme->is_string() ? scheme->as_string() != "never" : flag(p, "add_prefix_space", true);
    }
  }
  const bool spm = sp_space && flag(model, "byte_fallback", false);
  if (spm) {
    d.model = TokenizerData::Model::kSpm;
    d.add_space_prefix = prepend;
  } else {
    d.model = TokenizerData::Model::kBpe;
    ENGINE_ASSIGN_OR_RETURN(d.pre, byte_level_pre(field(tj, "pre_tokenizer")));
  }

  // Tokens: base vocabulary plus added tokens, indexed by id.
  int64_t max_id = -1;
  for (const auto& [piece, id] : vocab->as_object()) {
    if (!id.is_number() || id.as_number() < 0) return Corrupt("tokenizer.json: bad id for '" + piece + "'");
    max_id = std::max(max_id, static_cast<int64_t>(id.as_number()));
  }
  const json::Value* added = field(tj, "added_tokens");
  if (added && added->is_array()) {
    for (const json::Value& a : added->as_array()) {
      const json::Value* id = field(&a, "id");
      if (!id || !id->is_number() || id->as_number() < 0) return Corrupt("tokenizer.json: bad added token id");
      max_id = std::max(max_id, static_cast<int64_t>(id->as_number()));
    }
  }
  if (max_id < 0 || max_id > (int64_t{1} << 24)) return Corrupt("tokenizer.json: implausible vocabulary size");
  const size_t n = static_cast<size_t>(max_id + 1);
  d.tokens.assign(n, std::string());
  d.types.assign(n, TokenType::kUnused);
  for (const auto& [piece, id] : vocab->as_object()) {
    const size_t i = static_cast<size_t>(id.as_number());
    d.tokens[i] = piece;
    const bool byte_piece = spm && piece.size() == 6 && piece.starts_with("<0x") && piece.back() == '>';
    d.types[i] = byte_piece ? TokenType::kByte : TokenType::kNormal;
  }
  if (added && added->is_array()) {
    for (const json::Value& a : added->as_array()) {
      const size_t i = static_cast<size_t>(field(&a, "id")->as_number());
      const json::Value* content = field(&a, "content");
      if (!content || !content->is_string()) return Corrupt("tokenizer.json: added token without content");
      const bool special = flag(&a, "special", false);
      // Some tokenizers (Gemma) re-list ordinary vocabulary pieces such as
      // "▁▁" as non-special added tokens; those stay normal pieces.
      if (!special && d.tokens[i] == content->as_string() && d.types[i] != TokenType::kUnused) continue;
      d.tokens[i] = content->as_string();
      d.types[i] = special ? TokenType::kControl : TokenType::kUserDefined;
    }
  }
  for (size_t i = 0; i < n; ++i) {
    if (d.tokens[i].empty()) d.tokens[i] = "<unused" + std::to_string(i) + ">";  // id gap
  }
  std::unordered_map<std::string_view, TokenId> ids;
  ids.reserve(n);
  for (size_t i = 0; i < n; ++i) ids.emplace(d.tokens[i], static_cast<TokenId>(i));
  auto id_of = [&](const std::string& text) {
    auto it = ids.find(text);
    return it == ids.end() || text.empty() ? kNoToken : it->second;
  };

  // Merges: "a b" strings (older files) or ["a", "b"] pairs.
  const json::Value* merges = field(model, "merges");
  std::vector<std::pair<std::string, std::string>> pairs;
  if (merges && merges->is_array()) {
    pairs.reserve(merges->as_array().size());
    for (const json::Value& m : merges->as_array()) {
      if (m.is_string()) {
        const std::string& s = m.as_string();
        const size_t sp = s.find(' ', 1);
        if (sp == std::string::npos) return Corrupt("tokenizer.json: malformed merge '" + s + "'");
        pairs.emplace_back(s.substr(0, sp), s.substr(sp + 1));
      } else if (m.is_array() && m.as_array().size() == 2 && m.as_array()[0].is_string() &&
                 m.as_array()[1].is_string()) {
        pairs.emplace_back(m.as_array()[0].as_string(), m.as_array()[1].as_string());
      } else {
        return Corrupt("tokenizer.json: malformed merge entry");
      }
    }
  }
  if (spm) {
    // SPM merges the adjacent pair whose result scores highest; HF BPE
    // applies merges by rank. Scoring each piece by the rank of the merge
    // that creates it (earlier = higher) gives the same order.
    d.scores.assign(n, -1e9f);
    for (size_t r = 0; r < pairs.size(); ++r) {
      const TokenId id = id_of(pairs[r].first + pairs[r].second);
      if (id != kNoToken && d.scores[static_cast<size_t>(id)] == -1e9f) {
        d.scores[static_cast<size_t>(id)] = -static_cast<float>(r);
      }
    }
  } else {
    d.merges.reserve(pairs.size());
    for (const auto& [a, b] : pairs) d.merges.push_back(a + " " + b);
  }

  // Special token ids: generation_config > config > tokenizer_config names.
  auto int_id = [](const json::Value* v) -> std::vector<TokenId> {
    std::vector<TokenId> out;
    if (v && v->is_number()) out.push_back(static_cast<TokenId>(v->as_number()));
    if (v && v->is_array()) {
      for (const json::Value& e : v->as_array()) {
        if (e.is_number()) out.push_back(static_cast<TokenId>(e.as_number()));
      }
    }
    return out;
  };
  auto first_of = [&](std::string_view key) -> std::vector<TokenId> {
    for (const json::Value* src : {f.generation_config, f.config}) {
      if (auto v = int_id(field(src, key)); !v.empty()) return v;
    }
    return {};
  };
  const std::vector<TokenId> bos = first_of("bos_token_id");
  std::vector<TokenId> eos = first_of("eos_token_id");
  d.bos = !bos.empty() ? bos.front() : id_of(token_text(field(f.tokenizer_config, "bos_token")));
  if (const TokenId e = id_of(token_text(field(f.tokenizer_config, "eos_token"))); e != kNoToken) {
    if (std::find(eos.begin(), eos.end(), e) == eos.end()) eos.push_back(e);
  }
  if (!eos.empty()) {
    d.eos = eos.front();
    d.eog.assign(eos.begin() + 1, eos.end());
  }
  if (const json::Value* unk = field(model, "unk_token"); unk && unk->is_string()) d.unk = id_of(unk->as_string());
  if (d.unk != kNoToken) d.types[static_cast<size_t>(d.unk)] = TokenType::kUnknown;
  d.pad = id_of(token_text(field(f.tokenizer_config, "pad_token")));
  for (TokenId id : {d.bos, d.eos, d.unk, d.pad}) {
    if (id != kNoToken && (id < 0 || static_cast<size_t>(id) >= n)) return Corrupt("special token id out of range");
  }

  // BOS/EOS insertion: explicit flags, else the post-processor template.
  bool template_bos = false, template_eos = false;
  for (const json::Value* p : flatten(field(tj, "post_processor"), "processors")) {
    if (type_of(p) != "TemplateProcessing") continue;
    const json::Value* single = field(p, "single");
    if (single && single->is_array() && !single->as_array().empty()) {
      template_bos = field(&single->as_array().front(), "SpecialToken") != nullptr;
      template_eos = single->as_array().size() > 1 && field(&single->as_array().back(), "SpecialToken") != nullptr;
    }
  }
  d.add_bos = flag(f.tokenizer_config, "add_bos_token", template_bos) && d.bos != kNoToken;
  d.add_eos = flag(f.tokenizer_config, "add_eos_token", template_eos) && d.eos != kNoToken;
  return d;
}

}  // namespace dynalm::hf
