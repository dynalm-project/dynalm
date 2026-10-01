#pragma once

// SafeTensors reader (https://github.com/huggingface/safetensors).
//
// Layout: u64 little-endian header size N, N bytes of JSON
//   {"name": {"dtype": "BF16", "shape": [rows, cols], "data_offsets": [begin, end]},
//    ..., "__metadata__": {"k": "v"}}
// then the tensor data; offsets are relative to the data section. Tensors
// are row-major (outermost dimension first), the same order as the engine.
//
// The file is memory-mapped and tensors are zero-copy views. The header is
// treated as hostile: size limits, offsets inside the data section, byte
// counts that match dtype x shape, and no overlapping tensors.

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/status.h"
#include "loader/mapped_file.h"
#include "tensor/tensor.h"

namespace engine::safetensors {

struct TensorInfo {
  std::string name;
  std::string dtype_name;      // as written in the file ("BF16", "F32", "I64", ...)
  std::optional<DType> dtype;  // nullopt: the engine cannot execute this type
  TensorShape shape;
  uint64_t offset = 0;  // absolute file offset
  uint64_t nbytes = 0;
};

class SafeTensorsFile {
 public:
  static Result<std::unique_ptr<SafeTensorsFile>> open(const std::string& path);
  static Result<std::unique_ptr<SafeTensorsFile>> parse(std::shared_ptr<MappedFile> file);

  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  const TensorInfo* find(std::string_view name) const;
  const std::map<std::string, std::string>& metadata() const { return metadata_; }
  uint64_t total_tensor_bytes() const;
  const MappedFile& file() const { return *file_; }

  // Zero-copy view of the mapped data (copied only if the data is not aligned
  // to its element size, which kernels require).
  Result<Tensor> load_tensor(const TensorInfo& info) const;

 private:
  std::shared_ptr<MappedFile> file_;
  std::vector<TensorInfo> tensors_;
  std::map<std::string, std::string> metadata_;
};

// "F32"/"F16"/"BF16" -> DType; other valid SafeTensors dtypes -> nullopt.
std::optional<DType> dtype_from_name(std::string_view name);
// Element size in bytes of any SafeTensors dtype; 0 if the name is unknown.
int element_bytes(std::string_view name);

}  // namespace engine::safetensors
