#include "api/json.h"

#include <gtest/gtest.h>

#include <random>
#include "common/core.h"

namespace dynalm::json {
namespace {

TEST(Json, ParsesAllTypes) {
  auto v = parse(R"({"a": 1, "b": -2.5e3, "c": "x\ny", "d": [true, false, null], "e": {}})");
  ASSERT_TRUE(v.ok()) << v.status().to_string();
  EXPECT_EQ(v->find("a")->as_number(), 1);
  EXPECT_EQ(v->find("b")->as_number(), -2500);
  EXPECT_EQ(v->find("c")->as_string(), "x\ny");
  ASSERT_EQ(v->find("d")->as_array().size(), 3u);
  EXPECT_TRUE(v->find("d")->as_array()[0].as_bool());
  EXPECT_TRUE(v->find("d")->as_array()[2].is_null());
  EXPECT_TRUE(v->find("e")->as_object().empty());
  EXPECT_EQ(v->find("missing"), nullptr);
}

TEST(Json, UnicodeEscapes) {
  auto v = parse(R"("café 😀 中")");
  ASSERT_TRUE(v.ok());
  EXPECT_EQ(v->as_string(), "caf\xC3\xA9 \xF0\x9F\x98\x80 \xE4\xB8\xAD");
  EXPECT_FALSE(parse(R"("\ud83d")").ok());        // unpaired high surrogate
  EXPECT_FALSE(parse(R"("\ude00")").ok());        // lone low surrogate
  EXPECT_FALSE(parse(R"("\ud83dA")").ok());  // bad pair
}

TEST(Json, RoundTrip) {
  const std::string src = R"({"arr":[1,2.5,"s\"q\\",null,true],"num":-0.001,"obj":{"k":"v"},"z":"\u0001"})";
  auto v = parse(src);
  ASSERT_TRUE(v.ok());
  const std::string out = dump(*v);
  auto again = parse(out);
  ASSERT_TRUE(again.ok());
  EXPECT_EQ(dump(*again), out);
  EXPECT_NE(out.find(R"("z":"\u0001")"), std::string::npos);
  EXPECT_EQ(dump(Value(3.0)), "3");
  EXPECT_EQ(dump(Value(0.5)), "0.5");
}

TEST(Json, RejectsMalformed) {
  for (const char* bad : {"", "{", "}", "[1,]", "{\"a\":}", "{\"a\" 1}", "{a:1}", "[1 2]", "01", "1.", ".5", "1e",
                          "-", "tru", "nul", "\"unterminated", "\"bad \\x escape\"", "\"ctl \x01\"", "{} extra",
                          "[\"a\",]", "{\"a\":1,}", "NaN", "Infinity"}) {
    EXPECT_FALSE(parse(bad).ok()) << bad;
  }
}

TEST(Json, Limits) {
  std::string deep(100, '[');
  deep += std::string(100, ']');
  EXPECT_FALSE(parse(deep).ok());  // depth 100 > 64
  std::string ok_depth(60, '[');
  ok_depth += std::string(60, ']');
  EXPECT_TRUE(parse(ok_depth).ok());
  ParseLimits small;
  small.max_bytes = 10;
  EXPECT_FALSE(parse(R"({"a":"0123456789"})", small).ok());
}

// Random mutations of a valid document must never crash (ASAN/UBSAN gate).
TEST(Json, FuzzNeverCrashes) {
  const std::string base = R"({"model":"m","messages":[{"role":"user","content":"hi é"}],"stop":["a","b"],"n":1.5e2})";
  std::mt19937 rng(9);
  for (int i = 0; i < 20000; ++i) {
    std::string s = base;
    const int edits = 1 + static_cast<int>(rng() % 4);
    for (int e = 0; e < edits; ++e) {
      const size_t pos = rng() % s.size();
      switch (rng() % 3) {
        case 0: s[pos] = static_cast<char>(rng()); break;
        case 1: s.erase(pos, 1 + rng() % 3); break;
        default: s.insert(pos, 1, "{}[]\",:\\u0"[rng() % 11]);
      }
      if (s.empty()) s = "x";
    }
    auto v = parse(s);
    if (v.ok()) (void)dump(*v);
  }
}

}  // namespace
}  // namespace dynalm::json
