#include "logging/log.h"

#include <gtest/gtest.h>

namespace engine::log {
namespace {

// Restores the global level after each test.
class LogTest : public ::testing::Test {
 protected:
  void SetUp() override { saved_ = level(); }
  void TearDown() override { set_level(saved_); }
  Level saved_ = Level::kInfo;
};

TEST_F(LogTest, LevelFiltering) {
  set_level(Level::kWarn);
  EXPECT_FALSE(enabled(Level::kDebug));
  EXPECT_FALSE(enabled(Level::kInfo));
  EXPECT_TRUE(enabled(Level::kWarn));
  EXPECT_TRUE(enabled(Level::kError));
  set_level(Level::kOff);
  EXPECT_FALSE(enabled(Level::kError));
}

TEST_F(LogTest, DisabledLogDoesNotEvaluateArguments) {
  set_level(Level::kError);
  int evaluated = 0;
  auto side_effect = [&] { return ++evaluated; };
  LOG_DEBUG("value {}", side_effect());
  EXPECT_EQ(evaluated, 0);
  LOG_ERROR("value {}", side_effect());
  EXPECT_EQ(evaluated, 1);
}

TEST_F(LogTest, ParseLevel) {
  Level l = Level::kInfo;
  EXPECT_TRUE(parse_level("debug", l));
  EXPECT_EQ(l, Level::kDebug);
  EXPECT_FALSE(parse_level("verbose", l));
  EXPECT_EQ(l, Level::kDebug);  // unchanged on failure
  EXPECT_EQ(level_name(Level::kWarn), "WARN");
}

}  // namespace
}  // namespace engine::log
