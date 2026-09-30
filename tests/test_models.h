#pragma once

// Paths of optional real-model test files. An environment variable of the
// same name overrides the default compiled in by tests/CMakeLists.txt.
// Tests skip when the file does not exist.

#include <cstdlib>
#include <filesystem>
#include <string>

namespace engine::testing {

inline std::string model_path(const char* env, const char* fallback) {
  if (const char* p = std::getenv(env); p && *p) return p;
  return fallback;
}

inline std::string smollm_model() { return model_path("ENGINE_TEST_MODEL", ENGINE_TEST_MODEL_PATH); }
inline std::string qwen_model() { return model_path("ENGINE_TEST_MODEL_QWEN", ENGINE_TEST_MODEL_QWEN_PATH); }

inline bool exists(const std::string& p) { return !p.empty() && std::filesystem::exists(p); }

}  // namespace engine::testing
