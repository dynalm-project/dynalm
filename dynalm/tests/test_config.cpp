// Phase 22: configuration sources (file < env < CLI) and AUTO sizing.

#include "config/config.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <map>
#include <string>

#include "runtime/engine.h"

namespace engine {
namespace {

constexpr OptionSpec kOpts[] = {{"threads"}, {"port"}, {"http-threads"}, {"model"}, {"disable-admin", false}};

std::map<std::string, std::string> g_env;
const char* fake_env(const char* name) {
  auto it = g_env.find(name);
  return it == g_env.end() ? nullptr : it->second.c_str();
}

TEST(Config, ParsesFileText) {
  auto a = parse_config_text("# comment\nthreads = 8\n\n  port=9000  # trailing\ndisable-admin\n", kOpts, "cfg");
  ASSERT_TRUE(a.ok()) << a.status().to_string();
  EXPECT_EQ(*a, (std::vector<std::string>{"--threads", "8", "--port", "9000", "--disable-admin"}));
  auto off = parse_config_text("disable-admin = false\n", kOpts, "cfg");
  ASSERT_TRUE(off.ok());
  EXPECT_TRUE(off->empty());
}

TEST(Config, RejectsBadFileWithLocation) {
  auto unknown = parse_config_text("threads = 2\nbogus = 1\n", kOpts, "my.conf");
  ASSERT_FALSE(unknown.ok());
  EXPECT_NE(unknown.status().message().find("my.conf:2"), std::string::npos);
  EXPECT_NE(unknown.status().message().find("bogus"), std::string::npos);
  EXPECT_FALSE(parse_config_text("threads\n", kOpts, "c").ok());              // value missing
  EXPECT_FALSE(parse_config_text("disable-admin = maybe\n", kOpts, "c").ok());  // not a boolean
}

TEST(Config, PrecedenceFileThenEnvThenCli) {
  const std::string path = ::testing::TempDir() + "engine_test.conf";
  {
    std::ofstream f(path);
    f << "threads = 2\nport = 7000\nhttp-threads = 5\n";
  }
  g_env = {{"DYNALM_PORT", "7100"}, {"DYNALM_HTTP_THREADS", "6"}, {"DYNALM_DISABLE_ADMIN", "1"}};
  const std::string_view cli[] = {"--config", path, "--port", "7200", "model.gguf"};
  auto merged = merge_config(cli, kOpts, fake_env);
  ASSERT_TRUE(merged.ok()) << merged.status().to_string();
  // A parser that applies arguments in order ends with the right winners.
  std::map<std::string, std::string> last;
  bool admin_off = false;
  std::string positional;
  for (size_t i = 0; i < merged->size(); ++i) {
    const std::string& a = (*merged)[i];
    if (a == "--disable-admin") admin_off = true;
    else if (a.starts_with("--")) last[a] = (*merged)[++i];
    else positional = a;
  }
  EXPECT_EQ(last["--threads"], "2");       // file only
  EXPECT_EQ(last["--http-threads"], "6");  // env beats file
  EXPECT_EQ(last["--port"], "7200");       // CLI beats env and file
  EXPECT_TRUE(admin_off);                  // env flag
  EXPECT_EQ(positional, "model.gguf");
  std::remove(path.c_str());

  g_env = {{"DYNALM_CONFIG", "/nonexistent/engine.conf"}};
  EXPECT_FALSE(merge_config({}, kOpts, fake_env).ok());  // a named config must exist
  g_env.clear();
}

TEST(Config, IntOrAuto) {
  int v = -1;
  EXPECT_TRUE(parse_int_or_auto("auto", v));
  EXPECT_EQ(v, 0);
  EXPECT_TRUE(parse_int_or_auto("12", v));
  EXPECT_EQ(v, 12);
  EXPECT_FALSE(parse_int_or_auto("-3", v));
  EXPECT_FALSE(parse_int_or_auto("12x", v));
}

TEST(Config, AutoKvTokensFollowsFreeRam) {
  ModelConfig c;
  c.num_layers = 24;
  c.num_heads = 14;
  c.num_kv_heads = 2;
  c.head_dim = 64;
  c.head_dim_v = 64;
  c.hidden_size = 896;
  c.context_length = 32768;
  // f16 K+V per token: 24 layers * 2 heads * 64 dims * 2 (K,V) * 2 bytes = 12 KiB.
  constexpr int64_t kGiB = int64_t{1} << 30, kPerToken = 24 * 2 * 64 * 2 * 2;
  const int64_t weights = kGiB / 2;
  // Half of (available - weights) goes to KV.
  EXPECT_EQ(auto_kv_tokens(c, DType::kF16, weights, kGiB), (kGiB / 4) / kPerToken / 16 * 16);  // 21840
  // Capped at 65536 tokens on big machines.
  EXPECT_EQ(auto_kv_tokens(c, DType::kF16, weights, 64 * kGiB), 65536);
  // Never below min(context, 2048), even when memory is short.
  EXPECT_EQ(auto_kv_tokens(c, DType::kF16, weights, weights), 2048);
  c.context_length = 512;
  EXPECT_EQ(auto_kv_tokens(c, DType::kF16, weights, weights), 512);
  // Unknown RAM: the previous fixed default.
  c.context_length = 32768;
  EXPECT_EQ(auto_kv_tokens(c, DType::kF16, weights, 0), 16384);
  // f32 KV halves the token budget.
  EXPECT_EQ(auto_kv_tokens(c, DType::kF32, weights, kGiB), (kGiB / 4) / (2 * kPerToken) / 16 * 16);
}

}  // namespace
}  // namespace engine
