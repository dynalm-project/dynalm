#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <cstdint>

#include "memory/host_memory.h"

namespace engine {
namespace {

TEST(TensorShape, Basics) {
  TensorShape s{2, 3, 4};
  EXPECT_EQ(s.rank(), 3);
  EXPECT_EQ(s.numel(), 24);
  EXPECT_EQ(s.to_string(), "[2, 3, 4]");
  EXPECT_EQ(s, (TensorShape{2, 3, 4}));
  EXPECT_NE(s, (TensorShape{2, 12}));
  const int64_t too_many[] = {1, 2, 3, 4, 5};
  EXPECT_FALSE(TensorShape::from(too_many).ok());
  const int64_t negative[] = {2, -1};
  EXPECT_FALSE(TensorShape::from(negative).ok());
}

TEST(TensorLayout, ContiguousStrides) {
  auto l = TensorLayout::contiguous(DType::kF32, {2, 3, 4});
  ASSERT_TRUE(l.ok());
  EXPECT_EQ(l->strides[2], 4);
  EXPECT_EQ(l->strides[1], 16);
  EXPECT_EQ(l->strides[0], 48);
  EXPECT_TRUE(l->is_contiguous());
  EXPECT_EQ(l->span_bytes(), 96);
}

TEST(TensorLayout, QuantizedRows) {
  auto l = TensorLayout::contiguous(DType::kQ4_K, {8, 512});
  ASSERT_TRUE(l.ok());
  EXPECT_EQ(l->strides[1], 0);         // packed
  EXPECT_EQ(l->strides[0], 2 * 144);   // 2 blocks per row
  EXPECT_EQ(l->span_bytes(), 8 * 288);
  EXPECT_TRUE(l->is_contiguous());
  EXPECT_FALSE(TensorLayout::contiguous(DType::kQ4_K, {8, 100}).ok());
}

TEST(Tensor, EmptyIsAlignedAndTracked) {
  const auto before = host_memory_stats();
  {
    auto t = Tensor::empty(DType::kF16, {3, 5}, 128);
    ASSERT_TRUE(t.ok());
    EXPECT_EQ(reinterpret_cast<uintptr_t>(t->data()) % 128, 0u);
    EXPECT_EQ(t->numel(), 15);
    EXPECT_GE(host_memory_stats().current_bytes, before.current_bytes + 30);
  }
  EXPECT_EQ(host_memory_stats().current_bytes, before.current_bytes);
}

TEST(Tensor, ZerosAndViewsShareStorage) {
  auto t = Tensor::zeros(DType::kF32, {4, 6});
  ASSERT_TRUE(t.ok());
  for (int i = 0; i < 24; ++i) EXPECT_EQ(t->data_as<float>()[i], 0.0f);
  for (int i = 0; i < 24; ++i) t->data_as<float>()[i] = static_cast<float>(i);

  auto r = t->reshape({2, 12});
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r->data(), t->data());
  EXPECT_EQ(r->storage(), t->storage());
  EXPECT_FALSE(t->reshape({5, 5}).ok());

  auto s = t->slice(0, 1, 2);  // rows 1..2
  ASSERT_TRUE(s.ok());
  EXPECT_EQ(s->data_as<float>()[0], 6.0f);
  EXPECT_EQ(s->shape(), (TensorShape{2, 6}));
  EXPECT_TRUE(s->view().is_contiguous());
}

TEST(TensorView, SliceInnerDimIsStrided) {
  auto t = Tensor::zeros(DType::kF32, {4, 6});
  ASSERT_TRUE(t.ok());
  float* p = t->data_as<float>();
  for (int i = 0; i < 24; ++i) p[i] = static_cast<float>(i);
  auto cols = t->view().slice(1, 2, 3);
  ASSERT_TRUE(cols.ok());
  EXPECT_EQ(cols->data_as<float>()[0], 2.0f);
  EXPECT_FALSE(cols->is_contiguous());
  EXPECT_EQ(cols->span_bytes(), 3 * 24 + 3 * 4);  // 3 row strides + last row cols 2..4
  EXPECT_FALSE(cols->reshape({12}).ok());  // needs a copy
}

TEST(TensorView, TransposeAndSelect) {
  auto t = Tensor::zeros(DType::kF32, {2, 3});
  ASSERT_TRUE(t.ok());
  float* p = t->data_as<float>();
  for (int i = 0; i < 6; ++i) p[i] = static_cast<float>(i);

  auto tr = t->view().transpose(0, 1);
  ASSERT_TRUE(tr.ok());
  EXPECT_EQ(tr->shape(), (TensorShape{3, 2}));
  EXPECT_FALSE(tr->is_contiguous());
  // element [2][1] of the transpose == original [1][2] == 5
  const auto* base = static_cast<const std::byte*>(tr->data());
  EXPECT_EQ(*reinterpret_cast<const float*>(base + 2 * tr->stride(0) + 1 * tr->stride(1)), 5.0f);

  auto row1 = t->view().select(0, 1);
  ASSERT_TRUE(row1.ok());
  EXPECT_EQ(row1->rank(), 1);
  EXPECT_EQ(row1->dim(0), 3);
  EXPECT_EQ(row1->data_as<float>()[0], 3.0f);

  auto col2 = t->view().select(1, 2);
  ASSERT_TRUE(col2.ok());
  EXPECT_EQ(col2->shape(), (TensorShape{2}));
  EXPECT_EQ(col2->stride(0), 12);
}

TEST(TensorView, QuantizedRestrictions) {
  auto t = Tensor::zeros(DType::kQ8_0, {4, 64});
  ASSERT_TRUE(t.ok());
  const TensorView v = *t;
  EXPECT_FALSE(v.transpose(0, 1).ok());
  EXPECT_FALSE(v.select(1, 3).ok());
  EXPECT_FALSE(v.slice(1, 1, 32).ok());   // not block-aligned
  auto blk = v.slice(1, 32, 32);          // second block of every row
  ASSERT_TRUE(blk.ok());
  EXPECT_EQ(static_cast<std::byte*>(blk->data()) - static_cast<std::byte*>(v.data()), 34);
  auto row = v.select(0, 2);
  ASSERT_TRUE(row.ok());
  EXPECT_EQ(static_cast<std::byte*>(row->data()) - static_cast<std::byte*>(v.data()), 2 * 68);
}

TEST(Tensor, FromStorageValidatesBounds) {
  auto st = Storage::allocate_host(1024);
  ASSERT_TRUE(st.ok());
  auto l = TensorLayout::contiguous(DType::kF32, {16, 16});  // exactly 1024 bytes
  ASSERT_TRUE(l.ok());
  TensorView ok_view((*st)->data(), *l);
  EXPECT_TRUE(Tensor::from_storage(*st, ok_view).ok());
  TensorView bad_view(static_cast<std::byte*>((*st)->data()) + 4, *l);
  EXPECT_FALSE(Tensor::from_storage(*st, bad_view).ok());
}

TEST(Tensor, BorrowedStorageKeepsOwnerAlive) {
  auto owner = std::make_shared<std::vector<float>>(8, 1.5f);
  std::weak_ptr<std::vector<float>> weak = owner;
  auto l = TensorLayout::contiguous(DType::kF32, {8});
  ASSERT_TRUE(l.ok());
  auto storage = Storage::borrow(owner->data(), 32, Device{}, owner);
  auto t = Tensor::from_storage(storage, TensorView(owner->data(), *l));
  ASSERT_TRUE(t.ok());
  owner.reset();
  storage.reset();
  EXPECT_FALSE(weak.expired());
  EXPECT_EQ(t->data_as<float>()[7], 1.5f);
  *t = Tensor();
  EXPECT_TRUE(weak.expired());
}

TEST(Tensor, AllocationFailureIsAnError) {
  set_host_alloc_limit_for_testing(1 << 20);
  auto t = Tensor::empty(DType::kF32, {1024, 1024});  // 4 MiB > limit
  set_host_alloc_limit_for_testing(0);
  ASSERT_FALSE(t.ok());
  EXPECT_EQ(t.status().code(), StatusCode::kOutOfMemory);
}

}  // namespace
}  // namespace engine
