#pragma once

// Registry of built-in architecture adapters (one .cpp per family).

#include <span>

#include "model/architecture.h"

namespace engine {

const ModelArchitecture& llama_architecture();
const ModelArchitecture& qwen_architecture();
const ModelArchitecture& gemma_architecture();
const ModelArchitecture& phi_architecture();

std::span<const ModelArchitecture* const> registered_architectures();

}  // namespace engine
