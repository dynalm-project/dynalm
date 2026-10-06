// `dynalm pull` helpers: link resolution, GGUF header peek, support verdict.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "loader/gguf/gguf.h"
#include "loader/model_source.h"
#include "common/core.h"

namespace dynalm {
namespace {

std::string data(const std::string& f) { return std::string(ENGINE_TEST_DATA_DIR) + "/" + f; }

TEST(ModelSource, ResolvesHuggingFaceLinks) {
  const std::string want =
      "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf";
  // Page link, download link, download link with query, short form.
  EXPECT_EQ(*resolve_model_url(
                "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/blob/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
            want);
  EXPECT_EQ(*resolve_model_url(want), want);
  EXPECT_EQ(*resolve_model_url(want + "?download=true"), want);
  EXPECT_EQ(*resolve_model_url("Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf"), want);
  // Files in subfolders and other revisions are kept.
  EXPECT_EQ(*resolve_model_url("https://huggingface.co/o/r/blob/v2/sub/m.gguf"),
            "https://huggingface.co/o/r/resolve/v2/sub/m.gguf");
  EXPECT_EQ(*resolve_model_url("o/r/sub/m.gguf"), "https://huggingface.co/o/r/resolve/main/sub/m.gguf");
  // Other hosts pass through.
  EXPECT_EQ(*resolve_model_url("https://example.com/x/m.gguf"), "https://example.com/x/m.gguf");
}

TEST(ModelSource, RejectsLinksThatAreNotOneFile) {
  EXPECT_FALSE(resolve_model_url("https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF").ok());
  EXPECT_FALSE(resolve_model_url("https://huggingface.co/o/r/tree/main").ok());
  EXPECT_FALSE(resolve_model_url("o/r").ok());
  EXPECT_FALSE(resolve_model_url("").ok());
  EXPECT_FALSE(resolve_model_url("https://example.com/a b.gguf").ok());
  EXPECT_FALSE(resolve_model_url("https://example.com/a\".gguf").ok());
}

TEST(ModelSource, FileNameFromUrl) {
  EXPECT_EQ(url_file_name("https://h/o/r/resolve/main/m.gguf?download=true"), "m.gguf");
  EXPECT_EQ(url_file_name("https://h/o/r/m.gguf#x"), "m.gguf");
  EXPECT_EQ(url_file_name("https://h/"), "");
}

std::vector<std::byte> read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::vector<char> c((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::vector<std::byte> b(c.size());
  std::memcpy(b.data(), c.data(), c.size());
  return b;
}

TEST(ModelSource, PeeksArchitectureFromTheHeaderOnly) {
  const std::vector<std::byte> file = read_file(data("tiny_llama.gguf"));
  ASSERT_GT(file.size(), 4096u);
  EXPECT_EQ(*peek_gguf_architecture(std::span(file).first(4096)), "llama");
  // Too little data: NotFound (the caller should fetch more), never a misread.
  for (size_t n : {0u, 8u, 24u, 40u}) {
    auto r = peek_gguf_architecture(std::span(file).first(n));
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(r.status().code(), n < 4 ? StatusCode::kCorrupt : StatusCode::kNotFound) << n;
  }
  // An HTML error page instead of a model.
  const std::string html = "<!DOCTYPE html><html>Access denied</html>";
  EXPECT_EQ(peek_gguf_architecture(std::as_bytes(std::span(html))).status().code(), StatusCode::kCorrupt);
}

// Minimal GGUF v3: one string KV (general.architecture) and one tensor.
std::string write_gguf(const std::string& name, const std::string& arch, uint32_t ggml_type, uint64_t elems,
                       uint64_t nbytes) {
  std::string s;
  auto u32 = [&](uint32_t v) { s.append(reinterpret_cast<const char*>(&v), 4); };
  auto u64 = [&](uint64_t v) { s.append(reinterpret_cast<const char*>(&v), 8); };
  auto str = [&](const std::string& v) { u64(v.size()); s += v; };
  s += "GGUF";
  u32(3);
  u64(1);  // tensors
  u64(1);  // kvs
  str("general.architecture");
  u32(8);
  str(arch);
  str("w");
  u32(1);
  u64(elems);
  u32(ggml_type);
  u64(0);
  s.resize((s.size() + 31) / 32 * 32, '\0');
  s.append(nbytes, '\0');
  const std::string path = (std::filesystem::temp_directory_path() / name).string();
  std::ofstream(path, std::ios::binary) << s;
  return path;
}

TEST(ModelSource, SupportVerdictNamesTheBlocker) {
  auto ok = gguf::GgufFile::open(data("tiny_llama.gguf"));
  ASSERT_TRUE(ok.ok());
  EXPECT_EQ(gguf_support_status(**ok), "ok");

  const std::string a = write_gguf("dynalm_test_arch.gguf", "qwen35", 0, 4, 16);  // f32
  auto fa = gguf::GgufFile::open(a);
  ASSERT_TRUE(fa.ok()) << fa.status().to_string();
  EXPECT_EQ(gguf_support_status(**fa), "unsupported architecture 'qwen35'");

  const std::string t = write_gguf("dynalm_test_type.gguf", "llama", 21, 256, 110);  // iq3_s
  auto ft = gguf::GgufFile::open(t);
  ASSERT_TRUE(ft.ok()) << ft.status().to_string();
  EXPECT_EQ(gguf_support_status(**ft), "unsupported tensor types: iq3_s");

  fa->reset();
  ft->reset();
  std::error_code ec;
  std::filesystem::remove(a, ec);
  std::filesystem::remove(t, ec);
}

}  // namespace
}  // namespace dynalm
