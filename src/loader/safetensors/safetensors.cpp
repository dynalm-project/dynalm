#include "loader/safetensors/safetensors.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "api/json.h"

namespace engine::safetensors {
namespace {

constexpr uint64_t kMaxHeaderBytes = 100ull << 20;  // the reference implementation's limit
constexpr double kMaxExactInteger = 9007199254740992.0;  // 2^53

// Non-negative integral JSON number that fits a double exactly.
bool as_u64(const json::Value& v, uint64_t& out) {
  if (!v.is_number()) return false;
  const double d = v.as_number();
  if (!(d >= 0) || d > kMaxExactInteger || std::floor(d) != d) return false;
  out = static_cast<uint64_t>(d);
  return true;
}

}  // namespace

std::optional<DType> dtype_from_name(std::string_view name) {
  if (name == "F32") return DType::kF32;
  if (name == "F16") return DType::kF16;
  if (name == "BF16") return DType::kBF16;
  return std::nullopt;
}

int element_bytes(std::string_view n) {
  if (n == "F64" || n == "I64" || n == "U64") return 8;
  if (n == "F32" || n == "I32" || n == "U32") return 4;
  if (n == "F16" || n == "BF16" || n == "I16" || n == "U16") return 2;
  if (n == "I8" || n == "U8" || n == "BOOL" || n == "F8_E4M3" || n == "F8_E5M2") return 1;
  return 0;
}

Result<std::unique_ptr<SafeTensorsFile>> SafeTensorsFile::open(const std::string& path) {
  ENGINE_ASSIGN_OR_RETURN(auto file, MappedFile::open(path));
  return parse(std::move(file));
}

Result<std::unique_ptr<SafeTensorsFile>> SafeTensorsFile::parse(std::shared_ptr<MappedFile> file) {
  const size_t size = file->size();
  if (size < 8) return Corrupt("safetensors: file too small");
  uint64_t header_len = 0;
  for (int i = 7; i >= 0; --i) header_len = (header_len << 8) | static_cast<uint8_t>(file->data()[i]);
  if (header_len < 2 || header_len > kMaxHeaderBytes || header_len > size - 8) {
    return Corrupt("safetensors: invalid header size " + std::to_string(header_len));
  }
  const std::string_view text(reinterpret_cast<const char*>(file->data() + 8), static_cast<size_t>(header_len));
  json::ParseLimits limits;
  limits.max_bytes = static_cast<size_t>(kMaxHeaderBytes);
  limits.max_depth = 8;
  auto header = json::parse(text, limits);
  if (!header.ok()) return Corrupt("safetensors: bad header JSON: " + header.status().message());
  if (!header->is_object()) return Corrupt("safetensors: header is not an object");

  auto st = std::unique_ptr<SafeTensorsFile>(new SafeTensorsFile());
  st->file_ = std::move(file);
  const uint64_t data_start = 8 + header_len;
  const uint64_t data_size = size - data_start;

  for (const auto& [name, entry] : header->as_object()) {
    if (name == "__metadata__") {
      if (!entry.is_object()) return Corrupt("safetensors: __metadata__ is not an object");
      for (const auto& [k, v] : entry.as_object()) {
        if (!v.is_string()) return Corrupt("safetensors: metadata value for '" + k + "' is not a string");
        st->metadata_[k] = v.as_string();
      }
      continue;
    }
    const std::string where = "safetensors: tensor '" + name + "': ";
    const json::Value* dt = entry.find("dtype");
    const json::Value* shape = entry.find("shape");
    const json::Value* offs = entry.find("data_offsets");
    if (!dt || !dt->is_string() || !shape || !shape->is_array() || !offs || !offs->is_array() ||
        offs->as_array().size() != 2) {
      return Corrupt(where + "needs dtype, shape and data_offsets[2]");
    }
    TensorInfo info;
    info.name = name;
    info.dtype_name = dt->as_string();
    const int elem = element_bytes(info.dtype_name);
    if (elem == 0) return Corrupt(where + "unknown dtype '" + info.dtype_name + "'");
    info.dtype = dtype_from_name(info.dtype_name);

    std::vector<int64_t> dims;
    uint64_t numel = 1;
    for (const json::Value& d : shape->as_array()) {
      uint64_t n = 0;
      if (!as_u64(d, n)) return Corrupt(where + "invalid shape");
      if (n != 0 && numel > (uint64_t{1} << 56) / n) return Corrupt(where + "shape too large");
      numel *= n;
      dims.push_back(static_cast<int64_t>(n));
    }
    if (dims.size() > static_cast<size_t>(kMaxDims)) return Unsupported(where + "rank > 4");
    ENGINE_ASSIGN_OR_RETURN(info.shape, TensorShape::from(dims));

    uint64_t begin = 0, end = 0;
    if (!as_u64(offs->as_array()[0], begin) || !as_u64(offs->as_array()[1], end) || end < begin ||
        end > data_size) {
      return Corrupt(where + "data_offsets outside the file");
    }
    if (end - begin != numel * static_cast<uint64_t>(elem)) {
      return Corrupt(where + "byte size " + std::to_string(end - begin) + " does not match dtype x shape");
    }
    info.offset = data_start + begin;
    info.nbytes = end - begin;
    st->tensors_.push_back(std::move(info));
  }

  // No two tensors may share bytes.
  std::vector<const TensorInfo*> by_offset;
  for (const TensorInfo& t : st->tensors_) by_offset.push_back(&t);
  std::sort(by_offset.begin(), by_offset.end(), [](auto* a, auto* b) { return a->offset < b->offset; });
  for (size_t i = 1; i < by_offset.size(); ++i) {
    if (by_offset[i]->offset < by_offset[i - 1]->offset + by_offset[i - 1]->nbytes) {
      return Corrupt("safetensors: tensors '" + by_offset[i - 1]->name + "' and '" + by_offset[i]->name +
                     "' overlap");
    }
  }
  return st;
}

const TensorInfo* SafeTensorsFile::find(std::string_view name) const {
  for (const TensorInfo& t : tensors_) {
    if (t.name == name) return &t;
  }
  return nullptr;
}

uint64_t SafeTensorsFile::total_tensor_bytes() const {
  uint64_t n = 0;
  for (const TensorInfo& t : tensors_) n += t.nbytes;
  return n;
}

Result<Tensor> SafeTensorsFile::load_tensor(const TensorInfo& info) const {
  if (!info.dtype) {
    return Unsupported("tensor '" + info.name + "' has dtype " + info.dtype_name + ", which the engine cannot execute");
  }
  ENGINE_ASSIGN_OR_RETURN(TensorLayout layout, TensorLayout::contiguous(*info.dtype, info.shape));
  const std::byte* src = file_->data() + info.offset;
  const size_t align = static_cast<size_t>(element_bytes(info.dtype_name));
  if (reinterpret_cast<uintptr_t>(src) % align != 0) {
    // The format does not promise element alignment; copy the rare misaligned tensor.
    ENGINE_ASSIGN_OR_RETURN(Tensor t, Tensor::empty(*info.dtype, info.shape));
    std::memcpy(t.data(), src, static_cast<size_t>(info.nbytes));
    return t;
  }
  void* p = const_cast<std::byte*>(src);
  auto storage = Storage::borrow(p, static_cast<size_t>(info.nbytes), Device{}, file_);
  return Tensor::from_storage(std::move(storage), TensorView(p, layout));
}

}  // namespace engine::safetensors
