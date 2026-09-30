#pragma once

// Tensor abstractions.
//
//   TensorShape  — up to 4 dims, ordered outermost → innermost (row-major).
//   TensorLayout — dtype + shape + byte strides.
//   TensorView   — non-owning (pointer + layout + device). Trivially copyable;
//                  this is what kernels receive. No refcount traffic.
//   Tensor       — owning handle: shared Storage + a view into it.
//
// Strides are in bytes because block-quantized rows cannot be addressed per
// element. For quantized dtypes the innermost dimension is always packed
// (whole blocks, stride recorded as 0) and may only be sliced on block
// boundaries. Views never copy data; operations that would need a copy
// (reshape of a non-contiguous view, transpose of a quantized row) fail.

#include <array>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>

#include "common/status.h"
#include "dtype/dtype.h"
#include "memory/storage.h"

namespace engine {

inline constexpr int kMaxDims = 4;

class TensorShape {
 public:
  TensorShape() = default;
  TensorShape(std::initializer_list<int64_t> dims);  // asserts rank <= kMaxDims
  static Result<TensorShape> from(std::span<const int64_t> dims);

  int rank() const { return rank_; }
  int64_t operator[](int i) const { return dims_[i]; }
  int64_t& operator[](int i) { return dims_[i]; }
  int64_t numel() const;
  std::span<const int64_t> dims() const { return {dims_.data(), static_cast<size_t>(rank_)}; }
  std::string to_string() const;

  friend bool operator==(const TensorShape& a, const TensorShape& b);

 private:
  std::array<int64_t, kMaxDims> dims_{};
  int rank_ = 0;
};

struct TensorLayout {
  DType dtype = DType::kF32;
  TensorShape shape;
  std::array<int64_t, kMaxDims> strides{};  // bytes; innermost of quantized = 0 (packed)

  // Densely packed row-major layout. Fails if a quantized innermost dim is not
  // a multiple of the block size or any dim is negative.
  static Result<TensorLayout> contiguous(DType dtype, const TensorShape& shape);

  bool is_contiguous() const;
  // Bytes from the first to one-past-the-last addressed byte.
  int64_t span_bytes() const;
};

class TensorView {
 public:
  TensorView() = default;
  TensorView(void* data, TensorLayout layout, Device device = {})
      : data_(data), layout_(layout), device_(device) {}

  void* data() const { return data_; }
  template <typename T>
  T* data_as() const { return static_cast<T*>(data_); }

  const TensorLayout& layout() const { return layout_; }
  DType dtype() const { return layout_.dtype; }
  const TensorShape& shape() const { return layout_.shape; }
  int rank() const { return layout_.shape.rank(); }
  int64_t dim(int i) const { return layout_.shape[i]; }
  int64_t stride(int i) const { return layout_.strides[i]; }
  int64_t numel() const { return layout_.shape.numel(); }
  bool is_contiguous() const { return layout_.is_contiguous(); }
  int64_t span_bytes() const { return layout_.span_bytes(); }
  Device device() const { return device_; }

  // Same elements, new shape. Requires a contiguous view.
  Result<TensorView> reshape(const TensorShape& shape) const;
  // Elements [start, start+length) along `dim`. Quantized innermost slices
  // must be block-aligned.
  Result<TensorView> slice(int dim, int64_t start, int64_t length) const;
  // Index `dim` at `index` and drop that dimension (rank - 1).
  Result<TensorView> select(int dim, int64_t index) const;
  // Swap two dimensions (strides only). Not allowed on a quantized innermost dim.
  Result<TensorView> transpose(int a, int b) const;

 private:
  void* data_ = nullptr;
  TensorLayout layout_;
  Device device_;
};

class Tensor {
 public:
  Tensor() = default;

  // Uninitialized contiguous tensor in freshly allocated host memory.
  static Result<Tensor> empty(DType dtype, const TensorShape& shape,
                              size_t alignment = kDefaultAlignment);
  static Result<Tensor> zeros(DType dtype, const TensorShape& shape);
  // A tensor viewing (part of) existing storage; validates the view's bytes
  // lie inside the storage.
  static Result<Tensor> from_storage(std::shared_ptr<Storage> storage, const TensorView& view);

  bool defined() const { return storage_ != nullptr; }
  const TensorView& view() const { return view_; }
  operator const TensorView&() const { return view_; }  // NOLINT(implicit)
  const std::shared_ptr<Storage>& storage() const { return storage_; }

  void* data() const { return view_.data(); }
  template <typename T>
  T* data_as() const { return view_.data_as<T>(); }
  DType dtype() const { return view_.dtype(); }
  const TensorShape& shape() const { return view_.shape(); }
  int64_t numel() const { return view_.numel(); }

  // View-producing ops that keep the storage alive.
  Result<Tensor> reshape(const TensorShape& shape) const;
  Result<Tensor> slice(int dim, int64_t start, int64_t length) const;

 private:
  Tensor(std::shared_ptr<Storage> storage, TensorView view)
      : storage_(std::move(storage)), view_(view) {}

  std::shared_ptr<Storage> storage_;
  TensorView view_;
};

}  // namespace engine
