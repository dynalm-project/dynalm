#pragma once

// DynaCore version (DD-068). DynaCore ships inside DynaLM but versions its
// public API on its own: a breaking change to dynacore/include bumps MINOR
// while MAJOR is 0, MAJOR afterwards. No ABI promise (static library).

#define DYNACORE_VERSION_MAJOR 0
#define DYNACORE_VERSION_MINOR 1
#define DYNACORE_VERSION_PATCH 0
#define DYNACORE_VERSION_STRING "0.1.0"

namespace dynacore {
constexpr int kVersionMajor = DYNACORE_VERSION_MAJOR;
constexpr int kVersionMinor = DYNACORE_VERSION_MINOR;
constexpr int kVersionPatch = DYNACORE_VERSION_PATCH;
constexpr const char* kVersionString = DYNACORE_VERSION_STRING;
}  // namespace dynacore
