#pragma once

// DynaLM is written against the DynaCore vocabulary (Status, Tensor, DType,
// Device, ThreadPool, ...) and uses it unqualified. This is the one place
// that imports it; DynaCore itself never sees DynaLM (DD-068).

namespace dynacore {}

namespace dynalm {
using namespace ::dynacore;
}  // namespace dynalm
