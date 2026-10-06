#pragma once

// GGUF (v2/v3) reader.
//
// Parses the header, metadata key/values and tensor directory of a memory-
// mapped GGUF file. Zero-copy: strings, arrays and tensor data are views into
// the mapping, which the GgufFile keeps alive. Every length/offset read from
// the file is bounds-checked; a malformed file yields kCorrupt, never UB.
//
// This is the only place GGUF type IDs and layouts are known. Everything
// downstream sees engine DTypes, TensorShapes and Tensors.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "dynacore/base/status.h"
#include "dynacore/tensor/dtype.h"
#include "dynacore/memory/mapped_file.h"
#include "dynacore/tensor/tensor.h"
#include "common/core.h"

namespace dynalm::gguf {

enum class ValueType : uint32_t {
  kU8 = 0, kI8 = 1, kU16 = 2, kI16 = 3, kU32 = 4, kI32 = 5, kF32 = 6, kBool = 7,
  kString = 8, kArray = 9, kU64 = 10, kI64 = 11, kF64 = 12,
};

std::string_view value_type_name(ValueType t);

// Array payload, left encoded in the file until someone asks for it (a vocab
// array can hold 150k strings; most consumers never touch most arrays).
struct Array {
  ValueType elem_type = ValueType::kU8;
  uint64_t count = 0;
  const std::byte* data = nullptr;  // first element
  size_t bytes = 0;                 // encoded size of all elements
};

struct Value {
  ValueType type = ValueType::kU8;
  union {
    uint64_t u;
    int64_t i;
    double f;
    bool b;
  } scalar{0};
  std::string_view str;  // kString
  Array arr;             // kArray

  bool is_integer() const;
  bool is_float() const { return type == ValueType::kF32 || type == ValueType::kF64; }
};

struct KeyValue {
  std::string_view key;
  Value value;
};

struct TensorInfo {
  std::string_view name;
  uint32_t ggml_type = 0;
  std::optional<DType> dtype;  // nullopt: type exists in GGUF but the engine can't run it
  TensorShape shape;           // engine order: outermost -> innermost
  uint64_t offset = 0;         // absolute file offset of the data
  uint64_t nbytes = 0;
};

// Name of a ggml type id (e.g. 12 -> "q4_K"); "unknown" if not recognized.
std::string_view ggml_type_name(uint32_t ggml_type);

class GgufFile {
 public:
  static Result<std::unique_ptr<GgufFile>> open(const std::string& path);
  static Result<std::unique_ptr<GgufFile>> parse(std::shared_ptr<MappedFile> file);

  uint32_t version() const { return version_; }
  uint64_t alignment() const { return alignment_; }
  uint64_t data_offset() const { return data_offset_; }
  const MappedFile& file() const { return *file_; }

  // --- metadata ---
  const std::vector<KeyValue>& metadata() const { return kvs_; }
  const Value* find(std::string_view key) const;

  // Typed getters. Integers of any width/signedness are accepted when the
  // value fits; kNotFound if missing, kInvalidArgument on type mismatch.
  Result<uint64_t> get_uint(std::string_view key) const;
  Result<int64_t> get_int(std::string_view key) const;
  Result<double> get_float(std::string_view key) const;  // accepts ints too
  Result<bool> get_bool(std::string_view key) const;
  Result<std::string_view> get_string(std::string_view key) const;
  Result<Array> get_array(std::string_view key) const;

  // Array decoding.
  static Result<std::vector<std::string_view>> array_strings(const Array& a);
  static Result<std::vector<float>> array_floats(const Array& a);
  static Result<std::vector<int64_t>> array_ints(const Array& a);

  // --- tensors ---
  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  const TensorInfo* find_tensor(std::string_view name) const;
  uint64_t total_tensor_bytes() const;
  // Zero-copy tensor over the mapped bytes. kUnsupported for dtypes the
  // engine can't represent.
  Result<Tensor> load_tensor(const TensorInfo& info) const;

 private:
  GgufFile() = default;
  Status parse_impl();

  std::shared_ptr<MappedFile> file_;
  uint32_t version_ = 0;
  uint64_t alignment_ = 32;
  uint64_t data_offset_ = 0;
  std::vector<KeyValue> kvs_;
  std::unordered_map<std::string_view, size_t> kv_index_;
  std::vector<TensorInfo> tensors_;
  std::unordered_map<std::string_view, size_t> tensor_index_;
};

}  // namespace dynalm::gguf
