#include "dynacore/tensor/tensor.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <utility>

namespace dynacore {

// ---------------------------------------------------------------------------
// TensorShape

TensorShape::TensorShape(std::initializer_list<int64_t> dims) {
  assert(dims.size() <= kMaxDims);
  for (int64_t d : dims) dims_[rank_++] = d;
}

Result<TensorShape> TensorShape::from(std::span<const int64_t> dims) {
  if (dims.size() > kMaxDims) {
    return InvalidArgument("tensor rank " + std::to_string(dims.size()) + " exceeds " +
                           std::to_string(kMaxDims));
  }
  TensorShape s;
  for (int64_t d : dims) {
    if (d < 0) return InvalidArgument("negative dimension");
    s.dims_[s.rank_++] = d;
  }
  return s;
}

int64_t TensorShape::numel() const {
  int64_t n = 1;
  for (int i = 0; i < rank_; ++i) n *= dims_[i];
  return n;
}

std::string TensorShape::to_string() const {
  std::string s = "[";
  for (int i = 0; i < rank_; ++i) {
    if (i) s += ", ";
    s += std::to_string(dims_[i]);
  }
  return s + "]";
}

bool operator==(const TensorShape& a, const TensorShape& b) {
  if (a.rank_ != b.rank_) return false;
  for (int i = 0; i < a.rank_; ++i) {
    if (a.dims_[i] != b.dims_[i]) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// TensorLayout

namespace {

// Stride the innermost dimension has in a packed layout.
int64_t packed_inner_stride(DType t) { return dtype_is_quantized(t) ? 0 : dtype_block_bytes(t); }

// Bytes of `n` innermost elements given the innermost stride.
int64_t inner_span(DType t, int64_t n, int64_t inner_stride) {
  if (n == 0) return 0;
  if (dtype_is_quantized(t)) return dtype_row_bytes(t, n);
  return (n - 1) * inner_stride + dtype_block_bytes(t);
}

}  // namespace

Result<TensorLayout> TensorLayout::contiguous(DType dtype, const TensorShape& shape) {
  TensorLayout l;
  l.dtype = dtype;
  l.shape = shape;
  const int r = shape.rank();
  if (r == 0) return l;
  for (int i = 0; i < r; ++i) {
    if (shape[i] < 0) return InvalidArgument("negative dimension in " + shape.to_string());
  }
  const int64_t row = dtype_row_bytes(dtype, shape[r - 1]);
  if (row < 0) {
    return InvalidArgument("innermost dim " + std::to_string(shape[r - 1]) + " of " +
                           std::string(dtype_name(dtype)) + " tensor is not a multiple of block size " +
                           std::to_string(dtype_block_elems(dtype)));
  }
  l.strides[r - 1] = packed_inner_stride(dtype);
  int64_t stride = row;
  for (int i = r - 2; i >= 0; --i) {
    l.strides[i] = stride;
    stride *= shape[i];
  }
  return l;
}

bool TensorLayout::is_contiguous() const {
  const int r = shape.rank();
  if (r == 0) return true;
  if (strides[r - 1] != packed_inner_stride(dtype)) return false;
  int64_t expect = dtype_row_bytes(dtype, shape[r - 1]);
  for (int i = r - 2; i >= 0; --i) {
    // Size-1 dims can carry any stride without affecting addressing.
    if (shape[i] != 1 && strides[i] != expect) return false;
    expect *= shape[i];
  }
  return true;
}

int64_t TensorLayout::span_bytes() const {
  const int r = shape.rank();
  if (r == 0) return dtype_block_bytes(dtype);
  if (shape.numel() == 0) return 0;
  int64_t span = inner_span(dtype, shape[r - 1], strides[r - 1]);
  for (int i = 0; i < r - 1; ++i) span += (shape[i] - 1) * strides[i];
  return span;
}

// ---------------------------------------------------------------------------
// TensorView

namespace {

Status check_dim(int dim, int rank) {
  if (dim < 0 || dim >= rank) {
    return InvalidArgument("dim " + std::to_string(dim) + " out of range for rank " +
                           std::to_string(rank));
  }
  return Status::Ok();
}

}  // namespace

Result<TensorView> TensorView::reshape(const TensorShape& shape) const {
  if (!is_contiguous()) return InvalidArgument("reshape requires a contiguous view");
  if (shape.numel() != numel()) {
    return InvalidArgument("reshape " + this->shape().to_string() + " -> " + shape.to_string() +
                           " changes element count");
  }
  ENGINE_ASSIGN_OR_RETURN(TensorLayout l, TensorLayout::contiguous(dtype(), shape));
  return TensorView(data_, l, device_);
}

Result<TensorView> TensorView::slice(int dim, int64_t start, int64_t length) const {
  ENGINE_RETURN_IF_ERROR(check_dim(dim, rank()));
  if (start < 0 || length < 0 || start + length > this->dim(dim)) {
    return InvalidArgument("slice [" + std::to_string(start) + ", " +
                           std::to_string(start + length) + ") out of range for dim of size " +
                           std::to_string(this->dim(dim)));
  }
  int64_t offset;
  if (dim == rank() - 1 && dtype_is_quantized(dtype())) {
    offset = dtype_row_bytes(dtype(), start);
    if (offset < 0 || dtype_row_bytes(dtype(), length) < 0) {
      return InvalidArgument("quantized innermost slice must be block-aligned");
    }
  } else {
    offset = start * stride(dim);
  }
  TensorLayout l = layout_;
  l.shape[dim] = length;
  return TensorView(static_cast<std::byte*>(data_) + offset, l, device_);
}

Result<TensorView> TensorView::select(int dim, int64_t index) const {
  ENGINE_RETURN_IF_ERROR(check_dim(dim, rank()));
  if (dim == rank() - 1 && dtype_is_quantized(dtype())) {
    return InvalidArgument("cannot select a single element of a quantized row");
  }
  ENGINE_ASSIGN_OR_RETURN(TensorView s, slice(dim, index, 1));
  TensorLayout l;
  l.dtype = dtype();
  std::array<int64_t, kMaxDims> dims{};
  int r = 0;
  for (int i = 0; i < rank(); ++i) {
    if (i == dim) continue;
    dims[r] = this->dim(i);
    l.strides[r] = stride(i);
    ++r;
  }
  ENGINE_ASSIGN_OR_RETURN(l.shape, TensorShape::from(std::span<const int64_t>(dims.data(), r)));
  return TensorView(s.data(), l, device_);
}

Result<TensorView> TensorView::transpose(int a, int b) const {
  ENGINE_RETURN_IF_ERROR(check_dim(a, rank()));
  ENGINE_RETURN_IF_ERROR(check_dim(b, rank()));
  const int inner = rank() - 1;
  if (dtype_is_quantized(dtype()) && a != b && (a == inner || b == inner)) {
    return InvalidArgument("cannot transpose the packed innermost dim of a quantized tensor");
  }
  TensorLayout l = layout_;
  std::swap(l.shape[a], l.shape[b]);
  std::swap(l.strides[a], l.strides[b]);
  return TensorView(data_, l, device_);
}

// ---------------------------------------------------------------------------
// Tensor

Result<Tensor> Tensor::empty(DType dtype, const TensorShape& shape, size_t alignment) {
  ENGINE_ASSIGN_OR_RETURN(TensorLayout l, TensorLayout::contiguous(dtype, shape));
  // Zero-element tensors still get a (minimal) allocation so data() is valid.
  const size_t bytes = static_cast<size_t>(std::max<int64_t>(l.span_bytes(), 1));
  ENGINE_ASSIGN_OR_RETURN(std::shared_ptr<Storage> storage, Storage::allocate_host(bytes, alignment));
  TensorView view(storage->data(), l, storage->device());
  return Tensor(std::move(storage), view);
}

Result<Tensor> Tensor::zeros(DType dtype, const TensorShape& shape) {
  ENGINE_ASSIGN_OR_RETURN(Tensor t, empty(dtype, shape));
  std::memset(t.data(), 0, t.storage()->size());
  return t;
}

Result<Tensor> Tensor::from_storage(std::shared_ptr<Storage> storage, const TensorView& view) {
  if (!storage) return InvalidArgument("null storage");
  const auto* base = static_cast<const std::byte*>(storage->data());
  const auto* p = static_cast<const std::byte*>(view.data());
  const int64_t span = view.span_bytes();
  if (p < base || span < 0 || p + span > base + storage->size()) {
    return InvalidArgument("view does not fit inside storage");
  }
  if (!(view.device() == storage->device())) return InvalidArgument("view/storage device mismatch");
  return Tensor(std::move(storage), view);
}

Result<Tensor> Tensor::reshape(const TensorShape& shape) const {
  ENGINE_ASSIGN_OR_RETURN(TensorView v, view_.reshape(shape));
  return Tensor(storage_, v);
}

Result<Tensor> Tensor::slice(int dim, int64_t start, int64_t length) const {
  ENGINE_ASSIGN_OR_RETURN(TensorView v, view_.slice(dim, start, length));
  return Tensor(storage_, v);
}

}  // namespace dynacore
