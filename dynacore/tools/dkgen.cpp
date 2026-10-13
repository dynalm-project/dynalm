// dkgen: writes the generated DynaKernel C++ unit (docs/dynakernel.md).
//
//   dkgen --out FILE [--set test|bench] [--isa scalar|avx2] [--kind fp32|int8]
//   dkgen --out FILE --kind llvm-ir --lowering portable|avx2|avxvnni --spec M,N,K,BIAS
//
// Only built-in spec sets are accepted (no user source text). --kind int8
// (sub-scaled int8 activations, interleaved Q4_K) exists for --isa avx2 only.
// --kind llvm-ir writes the LLVM IR of one INT8 specialization (llvm_jit.h),
// for inspection with LLVM tools; the build does not use it.
// The file is rewritten only when its content changes, so builds stay
// incremental. Exit code: 0 ok, 1 generation or I/O error, 2 usage.

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "dynacore/dynakernel/codegen.h"
#include "dynacore/dynakernel/llvm_jit.h"

namespace {

int usage() {
  std::fprintf(stderr, "usage: dkgen --out FILE [--set test|bench] [--isa scalar|avx2] [--kind fp32|int8]\n"
                       "       dkgen --out FILE --kind llvm-ir --lowering portable|avx2|avxvnni --spec M,N,K,BIAS\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::string out, set = "test", isa_name = "scalar", kind = "fp32", lowering, spec;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--out" && i + 1 < argc) out = argv[++i];
    else if (a == "--set" && i + 1 < argc) set = argv[++i];
    else if (a == "--isa" && i + 1 < argc) isa_name = argv[++i];
    else if (a == "--kind" && i + 1 < argc) kind = argv[++i];
    else if (a == "--lowering" && i + 1 < argc) lowering = argv[++i];
    else if (a == "--spec" && i + 1 < argc) spec = argv[++i];
    else return usage();
  }
  if (kind == "llvm-ir") {
    using dynacore::dk::LlvmLowering;
    int m = 0, b = 0;
    long long n = 0, k = 0;
    if (out.empty() || std::sscanf(spec.c_str(), "%d,%lld,%lld,%d", &m, &n, &k, &b) != 4) return usage();
    LlvmLowering l;
    if (lowering == "portable") l = LlvmLowering::kPortable;
    else if (lowering == "avx2") l = LlvmLowering::kAvx2;
    else if (lowering == "avxvnni") l = LlvmLowering::kAvxVnni;
    else return usage();
    const auto ir = dynacore::dk::emit_q4k_int8_llvm_ir({m, n, k, b != 0}, l);
    if (!ir.ok()) {
      std::fprintf(stderr, "dkgen: %s\n", ir.status().to_string().c_str());
      return 1;
    }
    std::ofstream f(out, std::ios::binary | std::ios::trunc);
    f << *ir;
    if (!f) {
      std::fprintf(stderr, "dkgen: cannot write %s\n", out.c_str());
      return 1;
    }
    std::printf("dkgen: %s (%zu bytes of LLVM IR) -> %s\n",
                dynacore::dk::llvm_kernel_symbol({m, n, k, b != 0}, l).c_str(), ir->size(), out.c_str());
    return 0;
  }
  if (out.empty() || (set != "test" && set != "bench") || (isa_name != "scalar" && isa_name != "avx2") ||
      (kind != "fp32" && kind != "int8") || (kind == "int8" && isa_name != "avx2")) {
    return usage();
  }
  const bool bench = set == "bench";
  const std::string unit = bench ? "q4k_bench" : "q4k";
  dynacore::Result<dynacore::dk::GeneratedSource> src = dynacore::InvalidArgument("unset");
  size_t count = 0;
  if (kind == "int8") {
    const auto specs = bench ? dynacore::dk::q4k_bench_specs() : dynacore::dk::q4k_int8_test_specs();
    count = specs.size();
    src = dynacore::dk::generate_q4k_int8_source(specs, unit);
  } else {
    const auto specs = bench ? dynacore::dk::q4k_bench_specs() : dynacore::dk::q4k_test_specs();
    count = specs.size();
    const auto isa = isa_name == "avx2" ? dynacore::dk::CodegenIsa::kAvx2 : dynacore::dk::CodegenIsa::kScalar;
    src = dynacore::dk::generate_q4k_source(specs, isa, unit);
  }
  if (!src.ok()) {
    std::fprintf(stderr, "dkgen: %s\n", src.status().to_string().c_str());
    return 1;
  }
  {
    std::ifstream in(out, std::ios::binary);
    std::stringstream cur;
    cur << in.rdbuf();
    if (in && cur.str() == src->text) return 0;  // unchanged
  }
  std::ofstream f(out, std::ios::binary | std::ios::trunc);
  f << src->text;
  if (!f) {
    std::fprintf(stderr, "dkgen: cannot write %s\n", out.c_str());
    return 1;
  }
  std::printf("dkgen: %zu kernels -> %s (fingerprint %016llx)\n", count, out.c_str(),
              static_cast<unsigned long long>(src->fingerprint));
  return 0;
}
