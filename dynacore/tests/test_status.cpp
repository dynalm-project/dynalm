#include "dynacore/base/status.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

namespace dynacore {
namespace {

TEST(Status, DefaultIsOk) {
  Status s;
  EXPECT_TRUE(s.ok());
  EXPECT_EQ(s.code(), StatusCode::kOk);
  EXPECT_EQ(s.to_string(), "OK");
}

TEST(Status, ErrorCarriesCodeAndMessage) {
  Status s = NotFound("tensor 'blk.0.attn_q.weight'");
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), StatusCode::kNotFound);
  EXPECT_EQ(s.to_string(), "NOT_FOUND: tensor 'blk.0.attn_q.weight'");
}

TEST(Result, HoldsValue) {
  Result<int> r = 42;
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(*r, 42);
  EXPECT_TRUE(r.status().ok());
}

TEST(Result, HoldsError) {
  Result<int> r = Corrupt("bad magic");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.status().code(), StatusCode::kCorrupt);
}

TEST(Result, MoveOnlyValue) {
  Result<std::unique_ptr<int>> r = std::make_unique<int>(7);
  ASSERT_TRUE(r.ok());
  std::unique_ptr<int> p = std::move(r).value();
  EXPECT_EQ(*p, 7);
}

Result<int> parse_positive(int v) {
  if (v <= 0) return InvalidArgument("not positive");
  return v;
}

Result<int> doubled(int v) {
  ENGINE_ASSIGN_OR_RETURN(int x, parse_positive(v));
  return x * 2;
}

Status check_both(int a, int b) {
  ENGINE_RETURN_IF_ERROR(parse_positive(a).status());
  ENGINE_RETURN_IF_ERROR(parse_positive(b).status());
  return Status::Ok();
}

TEST(Result, Macros) {
  EXPECT_EQ(*doubled(21), 42);
  EXPECT_EQ(doubled(-1).status().code(), StatusCode::kInvalidArgument);
  EXPECT_TRUE(check_both(1, 2).ok());
  EXPECT_EQ(check_both(1, 0).code(), StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace dynacore
