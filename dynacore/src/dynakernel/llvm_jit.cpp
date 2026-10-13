// LLVM native compilation for DynaKernel (llvm_jit.h).
//
// LLVM's C API is resolved at run time from LLVM-C.dll / libLLVM.so: this file
// declares the few entry points it uses (opaque handles as void*, C enums as
// int, values as in llvm-c/*.h of LLVM 17-20), so building it needs no LLVM
// headers or libraries. Every entry point must resolve or LLVM counts as
// unavailable.

#include "dynacore/dynakernel/llvm_jit.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace dynacore::dk {

namespace {

using Ref = void*;  // every LLVM*Ref / LLVMOrc*Ref handle
using Bool = int;   // LLVMBool

// llvm-c enum values used here.
constexpr int kCodeGenLevelAggressive = 3;  // LLVMCodeGenLevelAggressive
constexpr int kRelocDefault = 0;            // LLVMRelocDefault
constexpr int kCodeModelJITDefault = 1;     // LLVMCodeModelJITDefault
constexpr int kAssemblyFile = 0;            // LLVMAssemblyFile
constexpr int kReturnStatusAction = 2;      // LLVMReturnStatusAction

struct Api {
  void (*GetVersion)(unsigned*, unsigned*, unsigned*);
  void (*InitializeX86TargetInfo)();
  void (*InitializeX86Target)();
  void (*InitializeX86TargetMC)();
  void (*InitializeX86AsmPrinter)();
  Ref (*ContextCreate)();
  void (*ContextDispose)(Ref);
  Ref (*CreateMemoryBufferWithMemoryRangeCopy)(const char*, size_t, const char*);
  const char* (*GetBufferStart)(Ref);
  size_t (*GetBufferSize)(Ref);
  void (*DisposeMemoryBuffer)(Ref);
  Bool (*ParseIRInContext)(Ref ctx, Ref buf, Ref* mod, char** msg);  // consumes buf
  Bool (*VerifyModule)(Ref mod, int action, char** msg);
  void (*DisposeModule)(Ref);
  void (*DisposeMessage)(char*);
  void (*SetTarget)(Ref mod, const char* triple);
  void (*SetModuleDataLayout)(Ref mod, Ref td);
  void (*DisposeTargetData)(Ref);
  char* (*GetDefaultTargetTriple)();
  char* (*GetHostCPUName)();
  char* (*GetHostCPUFeatures)();
  Bool (*GetTargetFromTriple)(const char*, Ref*, char**);
  Ref (*CreateTargetMachine)(Ref t, const char* triple, const char* cpu, const char* features, int level, int reloc,
                             int code_model);
  void (*DisposeTargetMachine)(Ref);
  Ref (*CreateTargetDataLayout)(Ref tm);
  Bool (*TargetMachineEmitToMemoryBuffer)(Ref tm, Ref mod, int file_type, char** err, Ref* out);
  Ref (*RunPasses)(Ref mod, const char* passes, Ref tm, Ref options);  // LLVMErrorRef
  Ref (*CreatePassBuilderOptions)();
  void (*DisposePassBuilderOptions)(Ref);
  char* (*GetErrorMessage)(Ref err);  // consumes err
  void (*DisposeErrorMessage)(char*);
  Ref (*OrcCreateLLJITBuilder)();
  void (*OrcLLJITBuilderSetJITTargetMachineBuilder)(Ref builder, Ref jtmb);  // takes jtmb
  Ref (*OrcJITTargetMachineBuilderCreateFromTargetMachine)(Ref tm);          // takes tm
  Ref (*OrcCreateLLJIT)(Ref* out, Ref builder);                              // takes builder
  Ref (*OrcDisposeLLJIT)(Ref jit);
  Ref (*OrcLLJITGetMainJITDylib)(Ref jit);
  char (*OrcLLJITGetGlobalPrefix)(Ref jit);
  Ref (*OrcLLJITAddLLVMIRModuleWithRT)(Ref jit, Ref rt, Ref tsm);  // takes tsm
  Ref (*OrcLLJITLookup)(Ref jit, uint64_t* addr, const char* name);
  Ref (*OrcJITDylibCreateResourceTracker)(Ref jd);
  Ref (*OrcResourceTrackerRemove)(Ref rt);
  void (*OrcReleaseResourceTracker)(Ref rt);
  Ref (*OrcCreateNewThreadSafeContext)();
  void (*OrcDisposeThreadSafeContext)(Ref);
  Ref (*OrcThreadSafeContextGetContext)(Ref);
  Ref (*OrcCreateNewThreadSafeModule)(Ref mod, Ref tsctx);  // takes mod
  Ref (*OrcCreateDynamicLibrarySearchGeneratorForProcess)(Ref* out, char prefix, void* filter, void* ctx);
  void (*OrcJITDylibAddGenerator)(Ref jd, Ref gen);  // takes gen
};

struct Loaded {
  Api api{};
  LlvmRuntimeInfo info;
};

#if defined(_WIN32)
void* open_library(const std::string& name) {
  // A full path loads its dependencies from its own directory.
  const bool path = name.find_first_of("\\/") != std::string::npos;
  return LoadLibraryExA(name.c_str(), nullptr, path ? LOAD_WITH_ALTERED_SEARCH_PATH : 0);
}
void* symbol(void* lib, const char* name) { return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib), name)); }
constexpr const char* kDefaultLibrary = "LLVM-C.dll";
#else
void* open_library(const std::string& name) { return dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL); }
void* symbol(void* lib, const char* name) { return dlsym(lib, name); }
constexpr const char* kDefaultLibrary = "libLLVM.so";
#endif

std::string take(const Api& a, char* s) {  // copies and frees an LLVM-allocated message
  std::string r = s != nullptr ? s : "";
  if (s != nullptr) a.DisposeMessage(s);
  return r;
}

std::string error_text(const Api& a, Ref err) {
  char* m = a.GetErrorMessage(err);
  std::string r = m != nullptr ? m : "unknown LLVM error";
  a.DisposeErrorMessage(m);
  return r;
}

Loaded load() {
  Loaded L;
  const char* env = std::getenv("DYNACORE_LLVM_C");
  const std::string name = env != nullptr && *env != '\0' ? env : kDefaultLibrary;
  L.info.library = name;
  void* lib = open_library(name);
  if (lib == nullptr) {
    L.info.error = "LLVM C API library '" + name + "' not found (set DYNACORE_LLVM_C to the path of LLVM-C.dll / libLLVM.so)";
    return L;
  }
  // Resolve every entry point; any miss makes LLVM unavailable. The library
  // stays loaded for the life of the process either way.
  std::string missing;
  auto get = [&](auto& fn, const char* sym) {
    fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(symbol(lib, sym));
    if (fn == nullptr) missing += (missing.empty() ? "" : ", ") + std::string(sym);
  };
  Api& a = L.api;
  get(a.GetVersion, "LLVMGetVersion");
  get(a.InitializeX86TargetInfo, "LLVMInitializeX86TargetInfo");
  get(a.InitializeX86Target, "LLVMInitializeX86Target");
  get(a.InitializeX86TargetMC, "LLVMInitializeX86TargetMC");
  get(a.InitializeX86AsmPrinter, "LLVMInitializeX86AsmPrinter");
  get(a.ContextCreate, "LLVMContextCreate");
  get(a.ContextDispose, "LLVMContextDispose");
  get(a.CreateMemoryBufferWithMemoryRangeCopy, "LLVMCreateMemoryBufferWithMemoryRangeCopy");
  get(a.GetBufferStart, "LLVMGetBufferStart");
  get(a.GetBufferSize, "LLVMGetBufferSize");
  get(a.DisposeMemoryBuffer, "LLVMDisposeMemoryBuffer");
  get(a.ParseIRInContext, "LLVMParseIRInContext");
  get(a.VerifyModule, "LLVMVerifyModule");
  get(a.DisposeModule, "LLVMDisposeModule");
  get(a.DisposeMessage, "LLVMDisposeMessage");
  get(a.SetTarget, "LLVMSetTarget");
  get(a.SetModuleDataLayout, "LLVMSetModuleDataLayout");
  get(a.DisposeTargetData, "LLVMDisposeTargetData");
  get(a.GetDefaultTargetTriple, "LLVMGetDefaultTargetTriple");
  get(a.GetHostCPUName, "LLVMGetHostCPUName");
  get(a.GetHostCPUFeatures, "LLVMGetHostCPUFeatures");
  get(a.GetTargetFromTriple, "LLVMGetTargetFromTriple");
  get(a.CreateTargetMachine, "LLVMCreateTargetMachine");
  get(a.DisposeTargetMachine, "LLVMDisposeTargetMachine");
  get(a.CreateTargetDataLayout, "LLVMCreateTargetDataLayout");
  get(a.TargetMachineEmitToMemoryBuffer, "LLVMTargetMachineEmitToMemoryBuffer");
  get(a.RunPasses, "LLVMRunPasses");
  get(a.CreatePassBuilderOptions, "LLVMCreatePassBuilderOptions");
  get(a.DisposePassBuilderOptions, "LLVMDisposePassBuilderOptions");
  get(a.GetErrorMessage, "LLVMGetErrorMessage");
  get(a.DisposeErrorMessage, "LLVMDisposeErrorMessage");
  get(a.OrcCreateLLJITBuilder, "LLVMOrcCreateLLJITBuilder");
  get(a.OrcLLJITBuilderSetJITTargetMachineBuilder, "LLVMOrcLLJITBuilderSetJITTargetMachineBuilder");
  get(a.OrcJITTargetMachineBuilderCreateFromTargetMachine, "LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine");
  get(a.OrcCreateLLJIT, "LLVMOrcCreateLLJIT");
  get(a.OrcDisposeLLJIT, "LLVMOrcDisposeLLJIT");
  get(a.OrcLLJITGetMainJITDylib, "LLVMOrcLLJITGetMainJITDylib");
  get(a.OrcLLJITGetGlobalPrefix, "LLVMOrcLLJITGetGlobalPrefix");
  get(a.OrcLLJITAddLLVMIRModuleWithRT, "LLVMOrcLLJITAddLLVMIRModuleWithRT");
  get(a.OrcLLJITLookup, "LLVMOrcLLJITLookup");
  get(a.OrcJITDylibCreateResourceTracker, "LLVMOrcJITDylibCreateResourceTracker");
  get(a.OrcResourceTrackerRemove, "LLVMOrcResourceTrackerRemove");
  get(a.OrcReleaseResourceTracker, "LLVMOrcReleaseResourceTracker");
  get(a.OrcCreateNewThreadSafeContext, "LLVMOrcCreateNewThreadSafeContext");
  get(a.OrcDisposeThreadSafeContext, "LLVMOrcDisposeThreadSafeContext");
  get(a.OrcThreadSafeContextGetContext, "LLVMOrcThreadSafeContextGetContext");
  get(a.OrcCreateNewThreadSafeModule, "LLVMOrcCreateNewThreadSafeModule");
  get(a.OrcCreateDynamicLibrarySearchGeneratorForProcess, "LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess");
  get(a.OrcJITDylibAddGenerator, "LLVMOrcJITDylibAddGenerator");
  if (!missing.empty()) {
    L.info.error = "'" + name + "' lacks required LLVM C API entry points: " + missing;
    return L;
  }
  a.GetVersion(&L.info.major, &L.info.minor, &L.info.patch);
  if (L.info.major < 17) {
    L.info.error = "LLVM " + std::to_string(L.info.major) + " is too old (opaque-pointer IR needs LLVM >= 17)";
    return L;
  }
  L.info.host_triple = take(a, a.GetDefaultTargetTriple());
  if (L.info.host_triple.rfind("x86_64", 0) != 0) {
    L.info.error = "host triple " + L.info.host_triple + ": the prototype generates x86-64 code only";
    return L;
  }
  a.InitializeX86TargetInfo();
  a.InitializeX86Target();
  a.InitializeX86TargetMC();
  a.InitializeX86AsmPrinter();
  L.info.host_cpu = take(a, a.GetHostCPUName());
  L.info.host_features = take(a, a.GetHostCPUFeatures());
  L.info.available = true;
  return L;
}

const Loaded& loaded() {
  static const Loaded L = load();  // thread-safe, once per process
  return L;
}

std::atomic<int> g_live{0};

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

const LlvmRuntimeInfo& llvm_runtime() { return loaded().info; }

Status llvm_available() {
  const LlvmRuntimeInfo& i = llvm_runtime();
  return i.available ? Status::Ok() : Unsupported("llvm backend: " + i.error);
}

bool llvm_feature_enabled(std::string_view features, std::string_view name) {
  bool on = false;
  size_t pos = 0;
  while (pos <= features.size()) {
    const size_t end = std::min(features.find(',', pos), features.size());
    const std::string_view f = features.substr(pos, end - pos);
    if (f.size() == name.size() + 1 && f.substr(1) == name && (f[0] == '+' || f[0] == '-')) on = f[0] == '+';
    pos = end + 1;
  }
  return on;
}

Status llvm_lowering_compatible(const LlvmTarget& resolved, LlvmLowering l) {
  const auto need = [&](const char* f) -> Status {
    if (llvm_feature_enabled(resolved.features, f)) return Status::Ok();
    return Unsupported("llvm backend: the " + std::string(llvm_lowering_name(l)) + " lowering needs +" + f +
                       ", which target '" + resolved.cpu + "' does not enable");
  };
  switch (l) {
    case LlvmLowering::kPortable: return Status::Ok();
    case LlvmLowering::kAvx2: return need("avx2");
    case LlvmLowering::kAvxVnni:
      ENGINE_RETURN_IF_ERROR(need("avx2"));
      return need("avxvnni");
  }
  return InvalidArgument("llvm backend: unknown lowering");
}

Status llvm_kernel_compatible(const LlvmTarget& resolved, LlvmLowering l, const LlvmKernelOptions& o) {
  ENGINE_RETURN_IF_ERROR(llvm_lowering_compatible(resolved, l));
  if (packed_layout_info(o.layout).fp16 && !llvm_feature_enabled(resolved.features, "f16c")) {
    return Unsupported("llvm backend: packed layout " + std::string(packed_layout_name(o.layout)) +
                       " stores fp16 scales and needs +f16c, which target '" + resolved.cpu + "' does not enable");
  }
  return Status::Ok();
}

LlvmTarget llvm_tier_target() {
  const CpuIsa tier = select_isa(cpu_info().features);
  if (tier != CpuIsa::kAvx2 && tier != CpuIsa::kAvx512 && tier != CpuIsa::kAmx) return {"x86-64", ""};
  LlvmTarget t;  // host
  if (tier == CpuIsa::kAvx2) t.features = "-avx512f,-amx-tile";  // appended to the host features
  return t;
}

struct LlvmJit::Impl {
  const Api* a = nullptr;
  Ref tm = nullptr;   // optimization and assembly
  Ref jit = nullptr;  // owns all compiled code
  Ref jd = nullptr;
  std::string triple;
  std::vector<std::string> symbols;  // compiled in this JIT
};

Result<std::unique_ptr<LlvmJit>> LlvmJit::create(const LlvmTarget& target) {
  ENGINE_RETURN_IF_ERROR(llvm_available());
  const Loaded& L = loaded();
  const Api& a = L.api;
  std::unique_ptr<LlvmJit> j(new LlvmJit());
  j->impl_ = std::make_unique<Impl>();
  Impl& im = *j->impl_;
  im.a = &a;
  im.triple = L.info.host_triple;
  if (target.cpu.empty()) {  // host, with optional feature overrides appended
    j->target_.cpu = L.info.host_cpu;
    j->target_.features = L.info.host_features + (target.features.empty() ? "" : "," + target.features);
  } else {
    j->target_ = target;
  }
  Ref t = nullptr;
  char* err = nullptr;
  if (a.GetTargetFromTriple(im.triple.c_str(), &t, &err) != 0) {
    return Unsupported("llvm backend: no target for " + im.triple + ": " + take(a, err));
  }
  auto make_tm = [&] {
    return a.CreateTargetMachine(t, im.triple.c_str(), j->target_.cpu.c_str(), j->target_.features.c_str(),
                                 kCodeGenLevelAggressive, kRelocDefault, kCodeModelJITDefault);
  };
  im.tm = make_tm();
  Ref jit_tm = make_tm();
  if (im.tm == nullptr || jit_tm == nullptr) {
    if (jit_tm != nullptr) a.DisposeTargetMachine(jit_tm);
    return Unsupported("llvm backend: cannot create a target machine for cpu '" + j->target_.cpu + "'");
  }
  Ref builder = a.OrcCreateLLJITBuilder();
  a.OrcLLJITBuilderSetJITTargetMachineBuilder(builder, a.OrcJITTargetMachineBuilderCreateFromTargetMachine(jit_tm));
  if (Ref e = a.OrcCreateLLJIT(&im.jit, builder); e != nullptr) {
    im.jit = nullptr;
    return Internal("llvm backend: LLJIT creation failed: " + error_text(a, e));
  }
  g_live.fetch_add(1);  // ~LlvmJit disposes it from here on
  im.jd = a.OrcLLJITGetMainJITDylib(im.jit);
  // Undefined symbols of generated code (e.g. fmaf when the target has no
  // FMA) resolve against the process's loaded libraries.
  Ref gen = nullptr;
  if (Ref e = a.OrcCreateDynamicLibrarySearchGeneratorForProcess(&gen, a.OrcLLJITGetGlobalPrefix(im.jit), nullptr, nullptr);
      e != nullptr) {
    return Internal("llvm backend: process symbol generator: " + error_text(a, e));
  }
  a.OrcJITDylibAddGenerator(im.jd, gen);
  return j;
}

LlvmJit::~LlvmJit() {
  if (!impl_) return;
  const Api& a = *impl_->a;
  if (impl_->jit != nullptr) {
    if (Ref e = a.OrcDisposeLLJIT(impl_->jit); e != nullptr) (void)error_text(a, e);
    g_live.fetch_sub(1);
  }
  if (impl_->tm != nullptr) a.DisposeTargetMachine(impl_->tm);
}

int LlvmJit::live_instances() { return g_live.load(); }

namespace {

// Parse + verify + set target + optimize into `ctx`. On success *out owns the module.
Status build_module(const Api& a, Ref ctx, Ref tm, const std::string& triple, const std::string& ir, Ref* out,
                    LlvmCompileStats* st) {
  auto t0 = std::chrono::steady_clock::now();
  Ref buf = a.CreateMemoryBufferWithMemoryRangeCopy(ir.data(), ir.size(), "dynakernel");
  Ref mod = nullptr;
  char* msg = nullptr;
  if (a.ParseIRInContext(ctx, buf, &mod, &msg) != 0) {
    return Internal("llvm backend: LLVM rejected the generated IR: " + take(a, msg));
  }
  if (st) st->parse_ms = ms_since(t0);
  t0 = std::chrono::steady_clock::now();
  if (a.VerifyModule(mod, kReturnStatusAction, &msg) != 0) {
    a.DisposeModule(mod);
    return Internal("llvm backend: generated IR failed LLVM verification: " + take(a, msg));
  }
  take(a, msg);
  if (st) st->verify_ms = ms_since(t0);
  a.SetTarget(mod, triple.c_str());
  Ref td = a.CreateTargetDataLayout(tm);
  a.SetModuleDataLayout(mod, td);
  a.DisposeTargetData(td);
  t0 = std::chrono::steady_clock::now();
  Ref opts = a.CreatePassBuilderOptions();
  Ref e = a.RunPasses(mod, "default<O3>", tm, opts);
  a.DisposePassBuilderOptions(opts);
  if (e != nullptr) {
    a.DisposeModule(mod);
    return Internal("llvm backend: optimization failed: " + error_text(a, e));
  }
  if (st) st->optimize_ms = ms_since(t0);
  *out = mod;
  return Status::Ok();
}

}  // namespace

Result<LlvmJit::Compiled> LlvmJit::compile(const std::string& ir, const std::string& symbol, LlvmCompileStats* stats) {
  Impl& im = *impl_;
  const Api& a = *im.a;
  for (const std::string& s : im.symbols) {
    if (s == symbol) return InvalidArgument("llvm backend: " + symbol + " is already compiled in this JIT");
  }
  LlvmCompileStats st;
  st.ir_bytes = ir.size();
  const auto t_all = std::chrono::steady_clock::now();
  Ref tsctx = a.OrcCreateNewThreadSafeContext();
  Ref mod = nullptr;
  const Status built = build_module(a, a.OrcThreadSafeContextGetContext(tsctx), im.tm, im.triple, ir, &mod, &st);
  if (!built.ok()) {
    a.OrcDisposeThreadSafeContext(tsctx);
    return built;
  }
  Ref tsm = a.OrcCreateNewThreadSafeModule(mod, tsctx);
  a.OrcDisposeThreadSafeContext(tsctx);  // the module keeps its context alive
  // One resource tracker per module: a failed lookup removes everything it added.
  Ref rt = a.OrcJITDylibCreateResourceTracker(im.jd);
  if (Ref e = a.OrcLLJITAddLLVMIRModuleWithRT(im.jit, rt, tsm); e != nullptr) {
    a.OrcReleaseResourceTracker(rt);
    return Internal("llvm backend: adding the module failed: " + error_text(a, e));
  }
  const auto t0 = std::chrono::steady_clock::now();
  Compiled c;
  uint64_t fn = 0, calls = 0;
  Ref e = a.OrcLLJITLookup(im.jit, &fn, symbol.c_str());  // code generation happens here
  if (e == nullptr) e = a.OrcLLJITLookup(im.jit, &calls, (symbol + "_calls").c_str());
  if (e != nullptr) {
    const std::string why = error_text(a, e);
    if (Ref r = a.OrcResourceTrackerRemove(rt); r != nullptr) (void)error_text(a, r);
    a.OrcReleaseResourceTracker(rt);
    return Internal("llvm backend: native code generation for " + symbol + " failed: " + why);
  }
  a.OrcReleaseResourceTracker(rt);  // the JITDylib keeps the code until the JIT is destroyed
  st.codegen_ms = ms_since(t0);
  st.total_ms = ms_since(t_all);
  c.fn = reinterpret_cast<void*>(static_cast<uintptr_t>(fn));
  c.calls = reinterpret_cast<void*>(static_cast<uintptr_t>(calls));
  im.symbols.push_back(symbol);
  if (stats) *stats = st;
  return c;
}

Result<LlvmJit::Compiled> LlvmJit::compile_kernel(const Q4KMatmulSpec& s, LlvmLowering l, LlvmCompileStats* stats,
                                                  const LlvmKernelOptions& o) {
  ENGINE_RETURN_IF_ERROR(llvm_kernel_compatible(target_, l, o));
  auto ir = emit_q4k_int8_llvm_ir(s, l, o);
  if (!ir.ok()) return ir.status();
  return compile(*ir, llvm_kernel_symbol(s, l, o), stats);
}

Result<std::string> LlvmJit::kernel_assembly(const Q4KMatmulSpec& spec, LlvmLowering l, const LlvmKernelOptions& o) {
  ENGINE_RETURN_IF_ERROR(llvm_kernel_compatible(target_, l, o));
  auto src = emit_q4k_int8_llvm_ir(spec, l, o);
  if (!src.ok()) return src.status();
  const std::string& ir = *src;
  Impl& im = *impl_;
  const Api& a = *im.a;
  Ref ctx = a.ContextCreate();
  Ref mod = nullptr;
  if (const Status s = build_module(a, ctx, im.tm, im.triple, ir, &mod, nullptr); !s.ok()) {
    a.ContextDispose(ctx);
    return s;
  }
  char* err = nullptr;
  Ref buf = nullptr;
  std::string text;
  const bool failed = a.TargetMachineEmitToMemoryBuffer(im.tm, mod, kAssemblyFile, &err, &buf) != 0;
  if (!failed) {
    text.assign(a.GetBufferStart(buf), a.GetBufferSize(buf));
    a.DisposeMemoryBuffer(buf);
  }
  const std::string why = take(a, err);
  a.DisposeModule(mod);
  a.ContextDispose(ctx);
  if (failed) return Internal("llvm backend: assembly emission failed: " + why);
  return text;
}

}  // namespace dynacore::dk
