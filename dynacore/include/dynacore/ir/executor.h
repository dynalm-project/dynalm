#pragma once

// GraphExecutor: runs a standalone IR graph on a Device (stage-1 lowering:
// every op or fused step becomes one call into the existing kernel library).
//
// The recording device replays the calls a model made; the executor runs IR
// that came from anywhere else: the DynaCore language (dynacorec), the text
// form, or a builder. It allocates storage for every activation buffer and KV
// cache, binds symbolic dimensions, and synthesizes weights that are not
// bound (a constant byte pattern: valid scales, no NaNs; timing does not
// depend on values). Supported ops are the ones the Device API executes; the
// others (softmax, mul, transpose, reshape, cast, reduce, broadcast, kv_read)
// report kUnsupported before anything runs.

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "dynacore/base/status.h"
#include "dynacore/device/device.h"
#include "dynacore/ir/ir.h"
#include "dynacore/ir/recording_device.h"

namespace dynacore::ir {

struct ExecutorOptions {
  std::map<std::string, int64_t> symbols;  // e.g. {"M", 4}; unbound symbols are 1
  int32_t context = 64;                    // KV positions already present (attention span = context + row)
  uint32_t seed = 1;                       // activation / index initialization
};

class GraphExecutor {
 public:
  static Result<std::unique_ptr<GraphExecutor>> create(const Graph& g, Device& device, const ExecutorOptions& opts);
  ~GraphExecutor();

  // Binds a weight (or input) to caller memory before run(); the type must match.
  Status bind(ValueId v, const TensorView& view);
  // Runs the graph: in op order (plan == nullptr) or as planned.
  Status run(const ExecPlan* plan = nullptr);
  // The storage of a value after run() (host-accessible devices only).
  TensorView view(ValueId v) const;

 private:
  GraphExecutor(const Graph& g, Device& d);
  Status allocate(const ExecutorOptions& opts);
  Status run_op(int32_t op);

  struct KvState;
  const Graph& g_;
  Device& dev_;
  std::vector<std::shared_ptr<Storage>> storage_;
  std::map<ValueId, TensorView> views_;
  std::map<int32_t, std::unique_ptr<KvState>> kv_;  // by buffer id
  std::map<ValueId, std::vector<int32_t>> index_;   // index arrays (host)
  std::map<ValueId, std::vector<float>> floats_;    // f32 constants
  std::map<std::string, int64_t> symbols_;
};

}  // namespace dynacore::ir
