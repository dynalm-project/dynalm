#include "platform/cpu_info.h"

#include <gtest/gtest.h>

#include "common/platform.h"
#include "platform/isa.h"

namespace engine {
namespace {

TEST(CpuInfo, TopologyIsSane) {
  const CpuInfo& info = cpu_info();
  EXPECT_GE(info.logical_cores, 1);
  EXPECT_GE(info.physical_cores, 1);
  EXPECT_LE(info.physical_cores, info.logical_cores);
  if (info.performance_cores > 0) {
    EXPECT_EQ(info.performance_cores + info.efficiency_cores, info.physical_cores);
  }
}

TEST(CpuInfo, CachedInstanceMatchesFreshDetection) {
  const CpuInfo fresh = detect_cpu_info();
  EXPECT_EQ(fresh.brand, cpu_info().brand);
  EXPECT_EQ(fresh.logical_cores, cpu_info().logical_cores);
  EXPECT_EQ(fresh.features.avx2, cpu_info().features.avx2);
}

TEST(CpuInfo, FeatureImplications) {
  const CpuFeatures& f = cpu_info().features;
  // OS-enabled wider state implies the narrower one.
  if (f.avx2) EXPECT_TRUE(f.avx);
  if (f.avx512f) EXPECT_TRUE(f.avx2);
  if (f.amx_int8) EXPECT_TRUE(f.amx_tile);
#if ENGINE_ARCH_X86_64
  EXPECT_FALSE(cpu_info().vendor.empty());
#endif
}

TEST(MemoryInfo, ReportsPhysicalMemory) {
  const MemoryInfo m = memory_info();
  EXPECT_GT(m.total_bytes, int64_t{256} << 20);
  EXPECT_GT(m.available_bytes, 0);
  EXPECT_LE(m.available_bytes, m.total_bytes);
}

TEST(Isa, GenericAlwaysUsable) {
  CpuFeatures none;
  EXPECT_EQ(select_best_isa(none), CpuIsa::kGeneric);
  EXPECT_TRUE(isa_supported(CpuIsa::kGeneric, none));
  EXPECT_TRUE(isa_compiled(CpuIsa::kGeneric));
}

TEST(Isa, SelectionRequiresFullFeatureSet) {
  CpuFeatures f;
  f.avx = f.avx2 = true;  // AVX2 without FMA/F16C is not our AVX2 tier
  EXPECT_FALSE(isa_supported(CpuIsa::kAvx2, f));
  f.fma = f.f16c = true;
  EXPECT_TRUE(isa_supported(CpuIsa::kAvx2, f));
  EXPECT_EQ(select_best_isa(f), isa_compiled(CpuIsa::kAvx2) ? CpuIsa::kAvx2 : CpuIsa::kGeneric);
}

TEST(Isa, SelectedTierIsSupportedOnThisMachine) {
  const CpuIsa best = select_best_isa(cpu_info().features);
  EXPECT_TRUE(isa_supported(best, cpu_info().features));
  EXPECT_TRUE(isa_compiled(best));
}

TEST(Isa, ParseRoundTrip) {
  for (CpuIsa isa : {CpuIsa::kGeneric, CpuIsa::kAvx2, CpuIsa::kAvx512, CpuIsa::kAmx, CpuIsa::kNeon}) {
    CpuIsa parsed = CpuIsa::kGeneric;
    ASSERT_TRUE(parse_isa(isa_name(isa), parsed));
    EXPECT_EQ(parsed, isa);
  }
  CpuIsa out = CpuIsa::kAvx2;
  EXPECT_FALSE(parse_isa("sse9", out));
}

}  // namespace
}  // namespace engine
