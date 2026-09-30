#include "loader/gguf/gguf.h"

#include <array>
#include <bit>
#include <cstring>
#include <limits>

namespace engine::gguf {

static_assert(std::endian::native == std::endian::little,
              "GGUF reader assumes a little-endian host");

namespace {

constexpr uint32_t kMagic = 0x46554747;  // "GGUF"
constexpr uint64_t kDefaultAlignment = 32;
constexpr int kMaxArrayDepth = 2;

// --- ggml type table --------------------------------------------------------
// Every ggml type we know the geometry of, so tensor extents can be validated
// even for types the engine does not execute.

struct GgmlType {
  std::string_view name;
  uint32_t block_elems;
  uint32_t block_bytes;
  std::optional<DType> dtype;
};

const GgmlType* ggml_type(uint32_t id) {
  static const std::array<GgmlType, 40> kTypes = [] {
    std::array<GgmlType, 40> t{};
    auto set = [&](uint32_t id, std::string_view n, uint32_t be, uint32_t bb,
                   std::optional<DType> d) { t[id] = {n, be, bb, d}; };
    set(0, "f32", 1, 4, DType::kF32);
    set(1, "f16", 1, 2, DType::kF16);
    set(2, "q4_0", 32, 18, DType::kQ4_0);
    set(3, "q4_1", 32, 20, DType::kQ4_1);
    set(6, "q5_0", 32, 22, DType::kQ5_0);
    set(7, "q5_1", 32, 24, DType::kQ5_1);
    set(8, "q8_0", 32, 34, DType::kQ8_0);
    set(9, "q8_1", 32, 36, DType::kQ8_1);
    set(10, "q2_K", 256, 84, DType::kQ2_K);
    set(11, "q3_K", 256, 110, DType::kQ3_K);
    set(12, "q4_K", 256, 144, DType::kQ4_K);
    set(13, "q5_K", 256, 176, DType::kQ5_K);
    set(14, "q6_K", 256, 210, DType::kQ6_K);
    set(15, "q8_K", 256, 292, DType::kQ8_K);
    set(16, "iq2_xxs", 256, 66, std::nullopt);
    set(17, "iq2_xs", 256, 74, std::nullopt);
    set(18, "iq3_xxs", 256, 98, std::nullopt);
    set(19, "iq1_s", 256, 50, std::nullopt);
    set(20, "iq4_nl", 32, 18, std::nullopt);
    set(21, "iq3_s", 256, 110, std::nullopt);
    set(22, "iq2_s", 256, 82, std::nullopt);
    set(23, "iq4_xs", 256, 136, std::nullopt);
    set(24, "i8", 1, 1, DType::kI8);
    set(25, "i16", 1, 2, DType::kI16);
    set(26, "i32", 1, 4, DType::kI32);
    set(27, "i64", 1, 8, std::nullopt);
    set(28, "f64", 1, 8, std::nullopt);
    set(29, "iq1_m", 256, 56, std::nullopt);
    set(30, "bf16", 1, 2, DType::kBF16);
    set(34, "tq1_0", 256, 54, std::nullopt);
    set(35, "tq2_0", 256, 66, std::nullopt);
    set(39, "mxfp4", 32, 17, std::nullopt);
    return t;
  }();
  if (id >= kTypes.size() || kTypes[id].block_elems == 0) return nullptr;
  return &kTypes[id];
}

size_t scalar_size(ValueType t) {
  switch (t) {
    case ValueType::kU8: case ValueType::kI8: case ValueType::kBool: return 1;
    case ValueType::kU16: case ValueType::kI16: return 2;
    case ValueType::kU32: case ValueType::kI32: case ValueType::kF32: return 4;
    case ValueType::kU64: case ValueType::kI64: case ValueType::kF64: return 8;
    default: return 0;
  }
}

// Bounds-checked little-endian cursor over the mapping.
class Reader {
 public:
  Reader(const std::byte* begin, const std::byte* end) : begin_(begin), p_(begin), end_(end) {}

  size_t offset() const { return static_cast<size_t>(p_ - begin_); }
  size_t remaining() const { return static_cast<size_t>(end_ - p_); }
  const std::byte* ptr() const { return p_; }

  template <typename T>
  bool read(T& out) {
    if (remaining() < sizeof(T)) return false;
    std::memcpy(&out, p_, sizeof(T));
    p_ += sizeof(T);
    return true;
  }

  bool skip(uint64_t n) {
    if (n > remaining()) return false;
    p_ += n;
    return true;
  }

  bool read_string(std::string_view& out) {
    uint64_t len;
    if (!read(len) || len > remaining()) return false;
    out = std::string_view(reinterpret_cast<const char*>(p_), static_cast<size_t>(len));
    p_ += len;
    return true;
  }

 private:
  const std::byte* begin_;
  const std::byte* p_;
  const std::byte* end_;
};

Status corrupt_at(const Reader& r, const std::string& what) {
  return Corrupt("GGUF: " + what + " at offset " + std::to_string(r.offset()));
}

bool valid_type(uint32_t t) { return t <= static_cast<uint32_t>(ValueType::kF64); }

// Skips over `count` encoded elements of `type`, validating as it goes.
Status skip_elements(Reader& r, ValueType type, uint64_t count, int depth) {
  if (type == ValueType::kString) {
    for (uint64_t i = 0; i < count; ++i) {
      std::string_view s;
      if (!r.read_string(s)) return corrupt_at(r, "truncated string array");
    }
    return Status::Ok();
  }
  if (type == ValueType::kArray) {
    if (depth >= kMaxArrayDepth) return corrupt_at(r, "array nesting too deep");
    for (uint64_t i = 0; i < count; ++i) {
      uint32_t et;
      uint64_t n;
      if (!r.read(et) || !r.read(n) || !valid_type(et)) return corrupt_at(r, "bad nested array");
      ENGINE_RETURN_IF_ERROR(skip_elements(r, static_cast<ValueType>(et), n, depth + 1));
    }
    return Status::Ok();
  }
  const size_t sz = scalar_size(type);
  if (count > r.remaining() / sz) return corrupt_at(r, "array exceeds file");
  r.skip(count * sz);
  return Status::Ok();
}

Status read_value(Reader& r, ValueType type, Value& v) {
  v.type = type;
  bool ok = true;
  switch (type) {
    case ValueType::kU8: { uint8_t x; ok = r.read(x); v.scalar.u = x; break; }
    case ValueType::kI8: { int8_t x; ok = r.read(x); v.scalar.i = x; break; }
    case ValueType::kU16: { uint16_t x; ok = r.read(x); v.scalar.u = x; break; }
    case ValueType::kI16: { int16_t x; ok = r.read(x); v.scalar.i = x; break; }
    case ValueType::kU32: { uint32_t x; ok = r.read(x); v.scalar.u = x; break; }
    case ValueType::kI32: { int32_t x; ok = r.read(x); v.scalar.i = x; break; }
    case ValueType::kU64: { uint64_t x; ok = r.read(x); v.scalar.u = x; break; }
    case ValueType::kI64: { int64_t x; ok = r.read(x); v.scalar.i = x; break; }
    case ValueType::kF32: { float x; ok = r.read(x); v.scalar.f = x; break; }
    case ValueType::kF64: { double x; ok = r.read(x); v.scalar.f = x; break; }
    case ValueType::kBool: {
      uint8_t x;
      ok = r.read(x) && x <= 1;
      v.scalar.b = x != 0;
      break;
    }
    case ValueType::kString: ok = r.read_string(v.str); break;
    case ValueType::kArray: {
      uint32_t et;
      if (!r.read(et) || !r.read(v.arr.count) || !valid_type(et)) {
        return corrupt_at(r, "bad array header");
      }
      v.arr.elem_type = static_cast<ValueType>(et);
      v.arr.data = r.ptr();
      const size_t start = r.offset();
      ENGINE_RETURN_IF_ERROR(skip_elements(r, v.arr.elem_type, v.arr.count, 1));
      v.arr.bytes = r.offset() - start;
      break;
    }
  }
  if (!ok) return corrupt_at(r, "truncated or invalid value");
  return Status::Ok();
}

bool is_signed(ValueType t) {
  return t == ValueType::kI8 || t == ValueType::kI16 || t == ValueType::kI32 || t == ValueType::kI64;
}

// Decodes one integer element of a numeric array.
int64_t load_int(const std::byte* p, ValueType t) {
  switch (t) {
    case ValueType::kU8: { uint8_t x; std::memcpy(&x, p, 1); return x; }
    case ValueType::kI8: { int8_t x; std::memcpy(&x, p, 1); return x; }
    case ValueType::kU16: { uint16_t x; std::memcpy(&x, p, 2); return x; }
    case ValueType::kI16: { int16_t x; std::memcpy(&x, p, 2); return x; }
    case ValueType::kU32: { uint32_t x; std::memcpy(&x, p, 4); return x; }
    case ValueType::kI32: { int32_t x; std::memcpy(&x, p, 4); return x; }
    case ValueType::kU64: { uint64_t x; std::memcpy(&x, p, 8); return static_cast<int64_t>(x); }
    case ValueType::kI64: { int64_t x; std::memcpy(&x, p, 8); return x; }
    case ValueType::kBool: { uint8_t x; std::memcpy(&x, p, 1); return x; }
    default: return 0;
  }
}

}  // namespace

std::string_view value_type_name(ValueType t) {
  static constexpr std::string_view kNames[] = {"u8",  "i8",  "u16",    "i16",   "u32", "i32", "f32",
                                                "bool", "string", "array", "u64", "i64", "f64"};
  const auto i = static_cast<uint32_t>(t);
  return i < std::size(kNames) ? kNames[i] : "?";
}

std::string_view ggml_type_name(uint32_t id) {
  const GgmlType* t = ggml_type(id);
  return t ? t->name : "unknown";
}

bool Value::is_integer() const {
  switch (type) {
    case ValueType::kU8: case ValueType::kI8: case ValueType::kU16: case ValueType::kI16:
    case ValueType::kU32: case ValueType::kI32: case ValueType::kU64: case ValueType::kI64:
      return true;
    default:
      return false;
  }
}

// ---------------------------------------------------------------------------

Result<std::unique_ptr<GgufFile>> GgufFile::open(const std::string& path) {
  ENGINE_ASSIGN_OR_RETURN(std::shared_ptr<MappedFile> f, MappedFile::open(path));
  return parse(std::move(f));
}

Result<std::unique_ptr<GgufFile>> GgufFile::parse(std::shared_ptr<MappedFile> file) {
  std::unique_ptr<GgufFile> g(new GgufFile());
  g->file_ = std::move(file);
  ENGINE_RETURN_IF_ERROR(g->parse_impl());
  return g;
}

Status GgufFile::parse_impl() {
  Reader r(file_->data(), file_->data() + file_->size());

  uint32_t magic;
  if (!r.read(magic) || magic != kMagic) return Corrupt("not a GGUF file (bad magic): " + file_->path());
  if (!r.read(version_)) return corrupt_at(r, "truncated header");
  if (version_ < 2 || version_ > 3) {
    return Unsupported("GGUF version " + std::to_string(version_) + " (supported: 2, 3)");
  }
  uint64_t n_tensors, n_kv;
  if (!r.read(n_tensors) || !r.read(n_kv)) return corrupt_at(r, "truncated header");
  // Cheap sanity bounds before reserving: each KV needs >= 12 bytes and each
  // tensor info >= 28, so counts beyond that are corrupt.
  if (n_kv > r.remaining() / 12 || n_tensors > r.remaining() / 28) {
    return corrupt_at(r, "implausible tensor/kv counts");
  }

  kvs_.reserve(static_cast<size_t>(n_kv));
  kv_index_.reserve(static_cast<size_t>(n_kv));
  for (uint64_t i = 0; i < n_kv; ++i) {
    KeyValue kv;
    uint32_t type;
    if (!r.read_string(kv.key) || !r.read(type)) return corrupt_at(r, "truncated metadata key");
    if (!valid_type(type)) return corrupt_at(r, "unknown metadata type " + std::to_string(type));
    ENGINE_RETURN_IF_ERROR(read_value(r, static_cast<ValueType>(type), kv.value));
    if (!kv_index_.emplace(kv.key, kvs_.size()).second) {
      return Corrupt("GGUF: duplicate metadata key '" + std::string(kv.key) + "'");
    }
    kvs_.push_back(kv);
  }

  if (const Value* a = find("general.alignment")) {
    if (!a->is_integer() || a->scalar.u == 0 || !std::has_single_bit(a->scalar.u)) {
      return Corrupt("GGUF: general.alignment must be a power of two");
    }
    alignment_ = a->scalar.u;
  } else {
    alignment_ = kDefaultAlignment;
  }

  tensors_.reserve(static_cast<size_t>(n_tensors));
  tensor_index_.reserve(static_cast<size_t>(n_tensors));
  for (uint64_t i = 0; i < n_tensors; ++i) {
    TensorInfo t;
    uint32_t n_dims;
    if (!r.read_string(t.name) || !r.read(n_dims)) return corrupt_at(r, "truncated tensor info");
    if (n_dims == 0 || n_dims > static_cast<uint32_t>(kMaxDims)) {
      return corrupt_at(r, "tensor '" + std::string(t.name) + "' has " + std::to_string(n_dims) + " dims");
    }
    // GGUF stores ne[0] = innermost; the engine orders dims outermost first.
    std::array<int64_t, kMaxDims> dims{};
    uint64_t numel = 1;
    for (uint32_t d = 0; d < n_dims; ++d) {
      uint64_t ne;
      if (!r.read(ne)) return corrupt_at(r, "truncated tensor dims");
      if (ne > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
          (ne != 0 && numel > std::numeric_limits<uint64_t>::max() / ne)) {
        return corrupt_at(r, "tensor dims overflow");
      }
      numel *= ne;
      dims[n_dims - 1 - d] = static_cast<int64_t>(ne);
    }
    ENGINE_ASSIGN_OR_RETURN(t.shape, TensorShape::from(std::span<const int64_t>(dims.data(), n_dims)));
    if (!r.read(t.ggml_type) || !r.read(t.offset)) return corrupt_at(r, "truncated tensor info");

    const GgmlType* gt = ggml_type(t.ggml_type);
    if (!gt) {
      return Unsupported("tensor '" + std::string(t.name) + "' has unknown ggml type " +
                         std::to_string(t.ggml_type));
    }
    t.dtype = gt->dtype;
    const auto inner = static_cast<uint64_t>(t.shape[t.shape.rank() - 1]);
    if (inner % gt->block_elems != 0) {
      return Corrupt("GGUF: tensor '" + std::string(t.name) + "' row not a multiple of block size");
    }
    t.nbytes = numel / gt->block_elems * gt->block_bytes;
    if (!tensor_index_.emplace(t.name, tensors_.size()).second) {
      return Corrupt("GGUF: duplicate tensor '" + std::string(t.name) + "'");
    }
    tensors_.push_back(t);
  }

  // Tensor data starts at the next alignment boundary after the directory.
  data_offset_ = (r.offset() + alignment_ - 1) / alignment_ * alignment_;
  for (TensorInfo& t : tensors_) {
    if (t.offset % alignment_ != 0) {
      return Corrupt("GGUF: tensor '" + std::string(t.name) + "' data is misaligned");
    }
    if (t.offset > file_->size() || data_offset_ > file_->size() - t.offset ||
        t.nbytes > file_->size() - data_offset_ - t.offset) {
      return Corrupt("GGUF: tensor '" + std::string(t.name) + "' extends past end of file");
    }
    t.offset += data_offset_;  // store absolute
  }
  return Status::Ok();
}

const Value* GgufFile::find(std::string_view key) const {
  auto it = kv_index_.find(key);
  return it == kv_index_.end() ? nullptr : &kvs_[it->second].value;
}

namespace {
Status missing(std::string_view key) { return NotFound("GGUF key '" + std::string(key) + "' not found"); }
Status mismatch(std::string_view key, const char* want, ValueType got) {
  return InvalidArgument("GGUF key '" + std::string(key) + "': expected " + want + ", got " +
                         std::string(value_type_name(got)));
}
}  // namespace

Result<uint64_t> GgufFile::get_uint(std::string_view key) const {
  const Value* v = find(key);
  if (!v) return missing(key);
  if (!v->is_integer()) return mismatch(key, "integer", v->type);
  if (is_signed(v->type) && v->scalar.i < 0) return mismatch(key, "non-negative integer", v->type);
  return v->scalar.u;
}

Result<int64_t> GgufFile::get_int(std::string_view key) const {
  const Value* v = find(key);
  if (!v) return missing(key);
  if (!v->is_integer()) return mismatch(key, "integer", v->type);
  if (!is_signed(v->type) && v->scalar.u > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return mismatch(key, "int64-representable integer", v->type);
  }
  return v->scalar.i;
}

Result<double> GgufFile::get_float(std::string_view key) const {
  const Value* v = find(key);
  if (!v) return missing(key);
  if (v->is_float()) return v->scalar.f;
  if (v->is_integer()) {
    return is_signed(v->type) ? static_cast<double>(v->scalar.i) : static_cast<double>(v->scalar.u);
  }
  return mismatch(key, "number", v->type);
}

Result<bool> GgufFile::get_bool(std::string_view key) const {
  const Value* v = find(key);
  if (!v) return missing(key);
  if (v->type != ValueType::kBool) return mismatch(key, "bool", v->type);
  return v->scalar.b;
}

Result<std::string_view> GgufFile::get_string(std::string_view key) const {
  const Value* v = find(key);
  if (!v) return missing(key);
  if (v->type != ValueType::kString) return mismatch(key, "string", v->type);
  return v->str;
}

Result<Array> GgufFile::get_array(std::string_view key) const {
  const Value* v = find(key);
  if (!v) return missing(key);
  if (v->type != ValueType::kArray) return mismatch(key, "array", v->type);
  return v->arr;
}

Result<std::vector<std::string_view>> GgufFile::array_strings(const Array& a) {
  if (a.elem_type != ValueType::kString) return InvalidArgument("array is not a string array");
  std::vector<std::string_view> out;
  out.reserve(static_cast<size_t>(a.count));
  Reader r(a.data, a.data + a.bytes);  // extent validated at parse time
  for (uint64_t i = 0; i < a.count; ++i) {
    std::string_view s;
    if (!r.read_string(s)) return Corrupt("GGUF: string array truncated");
    out.push_back(s);
  }
  return out;
}

Result<std::vector<float>> GgufFile::array_floats(const Array& a) {
  std::vector<float> out(static_cast<size_t>(a.count));
  if (a.elem_type == ValueType::kF32) {
    std::memcpy(out.data(), a.data, out.size() * sizeof(float));
  } else if (a.elem_type == ValueType::kF64) {
    for (size_t i = 0; i < out.size(); ++i) {
      double d;
      std::memcpy(&d, a.data + i * 8, 8);
      out[i] = static_cast<float>(d);
    }
  } else {
    return InvalidArgument("array is not a float array");
  }
  return out;
}

Result<std::vector<int64_t>> GgufFile::array_ints(const Array& a) {
  const size_t sz = scalar_size(a.elem_type);
  if (sz == 0 || a.elem_type == ValueType::kF32 || a.elem_type == ValueType::kF64) {
    return InvalidArgument("array is not an integer array");
  }
  std::vector<int64_t> out(static_cast<size_t>(a.count));
  for (size_t i = 0; i < out.size(); ++i) out[i] = load_int(a.data + i * sz, a.elem_type);
  return out;
}

const TensorInfo* GgufFile::find_tensor(std::string_view name) const {
  auto it = tensor_index_.find(name);
  return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

uint64_t GgufFile::total_tensor_bytes() const {
  uint64_t total = 0;
  for (const TensorInfo& t : tensors_) total += t.nbytes;
  return total;
}

Result<Tensor> GgufFile::load_tensor(const TensorInfo& info) const {
  if (!info.dtype) {
    return Unsupported("tensor '" + std::string(info.name) + "' uses ggml type " +
                       std::string(ggml_type_name(info.ggml_type)) + ", which the engine cannot execute");
  }
  ENGINE_ASSIGN_OR_RETURN(TensorLayout layout, TensorLayout::contiguous(*info.dtype, info.shape));
  // One Storage per tensor, each keeping the mapping alive. Weight data is
  // read-only; kernels take const views.
  void* p = const_cast<std::byte*>(file_->data() + info.offset);
  auto storage = Storage::borrow(p, static_cast<size_t>(info.nbytes), Device{}, file_);
  return Tensor::from_storage(std::move(storage), TensorView(p, layout));
}

}  // namespace engine::gguf
