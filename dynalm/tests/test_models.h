#pragma once

// Paths of optional real-model test files. An environment variable of the
// same name overrides the default compiled in by tests/CMakeLists.txt.
// Tests skip when the file does not exist.

#include <cstdlib>
#include <filesystem>
#include <string>
#include "common/core.h"

namespace dynalm::testing {

inline std::string model_path(const char* env, const char* fallback) {
  if (const char* p = std::getenv(env); p && *p) return p;
  return fallback;
}

inline std::string smollm_model() { return model_path("ENGINE_TEST_MODEL", ENGINE_TEST_MODEL_PATH); }
inline std::string qwen_model() { return model_path("ENGINE_TEST_MODEL_QWEN", ENGINE_TEST_MODEL_QWEN_PATH); }

inline std::string qwen_f16_model() {
  return model_path("ENGINE_TEST_MODEL_QWEN_F16", ENGINE_TEST_MODEL_QWEN_F16_PATH);
}
inline std::string gemma_model() { return model_path("ENGINE_TEST_MODEL_GEMMA", ENGINE_TEST_MODEL_GEMMA_PATH); }

inline std::string smollm_q8_model() { return model_path("ENGINE_TEST_MODEL_Q8", ENGINE_TEST_MODEL_Q8_PATH); }

inline std::string qwen_q4_model() { return model_path("ENGINE_TEST_MODEL_QWEN_Q4", ENGINE_TEST_MODEL_QWEN_Q4_PATH); }

inline std::string granite_moe_model() {
  return model_path("ENGINE_TEST_MODEL_GRANITE_MOE", ENGINE_TEST_MODEL_GRANITE_MOE_PATH);
}

inline bool exists(const std::string& p) { return !p.empty() && std::filesystem::exists(p); }

}  // namespace dynalm::testing
