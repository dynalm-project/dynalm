# engine

A CPU-first LLM inference runtime in C++20. It loads GGUF models and executes
them with its own kernels, scheduler, and paged KV cache. The architecture is
designed so GPU backends can be added later without touching the scheduler,
model, or KV subsystems.

**Status:** Phase 0 (foundation). See [TODO.md](TODO.md) for the roadmap and
[docs/architecture.md](docs/architecture.md) for the design.

```sh
cmake --preset msvc-release && cmake --build --preset msvc-release
ctest --preset msvc-release
build/msvc-release/src/engine info
```

See [docs/development.md](docs/development.md) for build requirements.
