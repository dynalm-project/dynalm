// R1: model registry names and the YAML subset of the config file (DD-070).

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "config/config.h"
#include "registry/model_registry.h"
#include "common/core.h"

namespace dynalm {
namespace {

TEST(Registry, NamesAliasesAndFamilies) {
  const RegistryEntry* e = find_registry_entry("qwen3:4b");
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->file(), "Qwen3-4B-Q4_K_M.gguf");
  EXPECT_EQ(find_registry_entry("QWEN3:4B"), e);         // case-insensitive
  EXPECT_EQ(find_registry_entry("qwen3"), e);            // family default
  EXPECT_EQ(find_registry_entry("qwen3:latest"), e);     // :latest = family default
  EXPECT_EQ(find_registry_entry("gemma:270m"), find_registry_entry("gemma3:270m"));
  EXPECT_EQ(find_registry_entry("llama:3b"), find_registry_entry("llama3.2:3b"));
  EXPECT_EQ(find_registry_entry("nosuch:1b"), nullptr);
  EXPECT_EQ(registry_entry_for_file("Qwen3-4B-Q4_K_M.gguf"), e);
}

TEST(Registry, EveryEntryIsWellFormed) {
  for (const RegistryEntry& e : registry_entries()) {
    EXPECT_TRUE(e.url.starts_with("https://")) << e.name;
    EXPECT_TRUE(e.file().ends_with(".gguf")) << e.name;
    EXPECT_NE(e.name.find(':'), std::string_view::npos) << e.name;
    EXPECT_GT(e.size_bytes, 0) << e.name;
  }
}

TEST(Registry, NamesVersusPaths) {
  EXPECT_TRUE(is_model_name("qwen3:4b"));
  EXPECT_TRUE(is_model_name("qwen3"));
  EXPECT_FALSE(is_model_name("./models/x.gguf"));
  EXPECT_FALSE(is_model_name("C:\\models\\x.gguf"));
  EXPECT_FALSE(is_model_name("x.gguf"));
  EXPECT_FALSE(is_model_name(""));
}

TEST(Registry, UnknownNameListsKnownOnes) {
  auto r = resolve_model("nosuch:1b");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.status().code(), StatusCode::kNotFound);
  EXPECT_NE(r.status().message().find("qwen3:4b"), std::string::npos);
  auto missing = resolve_model("./definitely/missing.gguf");
  ASSERT_FALSE(missing.ok());
  EXPECT_NE(missing.status().message().find("not found"), std::string::npos);
}

TEST(Registry, KnownNameResolvesEvenWhenNotDownloaded) {
  auto r = resolve_model("phi3.5:3.8b");
  ASSERT_TRUE(r.ok()) << r.status().to_string();
  ASSERT_NE(r->entry, nullptr);
  EXPECT_NE(r->path.find("Phi-3.5-mini-instruct-Q4_K_M.gguf"), std::string::npos);
}

constexpr OptionSpec kServeLike[] = {{"model"}, {"threads"}, {"ctx"},  {"port"},
                                     {"host"},  {"backend"}, {"kv"},   {"disable-admin", false}};
constexpr OptionSpec kRunLike[] = {{"threads"}, {"ctx"}};

TEST(ConfigYaml, SectionsMapToOptions) {
  const char* text =
      "# DynaLM\n"
      "model:\n"
      "  path: qwen3:4b\n"
      "runtime:\n"
      "  device: auto      # default device: no option emitted\n"
      "  threads: 8\n"
      "  context_length: auto\n"
      "kv_cache:\n"
      "  dtype: \"f16\"\n"
      "server:\n"
      "  host: 0.0.0.0\n"
      "  port: 9000\n"
      "  admin: false\n";
  auto a = parse_config_text(text, kServeLike, "c.yaml");
  ASSERT_TRUE(a.ok()) << a.status().to_string();
  EXPECT_EQ(*a, (std::vector<std::string>{"--model", "qwen3:4b", "--threads", "8", "--ctx", "auto", "--kv", "f16",
                                          "--host", "0.0.0.0", "--port", "9000", "--disable-admin"}));
}

TEST(ConfigYaml, KeysOtherCommandsUseAreSkipped) {
  auto a = parse_config_text("runtime:\n  threads: 4\nserver:\n  port: 9000\n", kRunLike, "c.yaml");
  ASSERT_TRUE(a.ok()) << a.status().to_string();
  EXPECT_EQ(*a, (std::vector<std::string>{"--threads", "4"}));
}

TEST(ConfigYaml, RejectsWhatTheSubsetDoesNotSupport) {
  auto unknown = parse_config_text("runtime:\n  thread: 4\n", kRunLike, "c.yaml");
  ASSERT_FALSE(unknown.ok());
  EXPECT_NE(unknown.status().message().find("c.yaml:2"), std::string::npos);
  EXPECT_FALSE(parse_config_text("runtime:\n\tthreads: 4\n", kRunLike, "c").ok());           // tab
  EXPECT_FALSE(parse_config_text("runtime:\n  - threads\n", kRunLike, "c").ok());            // list
  EXPECT_FALSE(parse_config_text("runtime:\n  threads:\n    x: 1\n", kRunLike, "c").ok());  // nesting
  EXPECT_FALSE(parse_config_text("  threads: 4\n", kRunLike, "c").ok());                     // no section
}

TEST(ConfigYaml, EntriesForConfigShow) {
  auto e = parse_config_entries("runtime:\n  threads: 4\nserver:\n  port: '8080'\n", "c.yaml");
  ASSERT_TRUE(e.ok());
  ASSERT_EQ(e->size(), 2u);
  EXPECT_EQ((*e)[0], (std::pair<std::string, std::string>{"runtime.threads", "4"}));
  EXPECT_EQ((*e)[1], (std::pair<std::string, std::string>{"server.port", "8080"}));
  // Flat files are not YAML: no entries.
  auto flat = parse_config_entries("threads = 4\n", "c.conf");
  ASSERT_TRUE(flat.ok());
  EXPECT_TRUE(flat->empty());
}

TEST(ConfigYaml, EveryDocumentedKeyNamesAnOption) {
  for (const YamlKey& k : yaml_keys()) {
    EXPECT_NE(k.key.find('.'), std::string_view::npos) << k.key;
    EXPECT_FALSE(k.option.empty()) << k.key;
  }
}

}  // namespace
}  // namespace dynalm
