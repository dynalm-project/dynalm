# Contributing to DynaLM

Thanks for helping! Bug reports, model requests, benchmarks on your hardware, and pull requests are all
welcome.

## Ways to help

- **Report a bug or a model that does not load:** open an issue with the output of `dynalm version`,
  `dynalm info` and `dynalm inspect <model>`.
- **Request a model:** run `dynalm pull <link> --check` and paste the result into a "Model request" issue.
- **Share benchmarks:** `dynalm benchmark <model> --concurrency 1,4,16` on CPUs we have not measured,
  especially ARM (Graviton, Ampere, Apple M-series, Raspberry Pi).
- **Pick an open item** from [ROADMAP.md](ROADMAP.md). Good first areas:
  - NEON kernels for Q2_K, Q3_K and Q5_K;
  - IQ quantization types;
  - split (sharded) GGUF downloads in `dynalm pull`.

## Building and testing

See [docs/development.md](docs/development.md). In short:

```sh
cmake --preset linux-release          # or macos-release, msvc-release
cmake --build --preset linux-release
ctest --preset linux-release
```

CI runs the same suite on Linux x86-64 (gcc, clang, ASAN/UBSAN, TSAN), Linux ARM64, macOS (Apple Silicon)
and Windows (MSVC). A pull request should keep all of them green.

## Pull request guidelines

- **Keep changes focused.** One topic per pull request.
- **Add a test.**
  - Every fix gets a regression test.
  - Every new kernel or model is checked against a reference: see `tools/ref_model.py` and the tiny-model
    fixtures in `dynalm/tests/data`.
- **No fake implementations.** An unsupported feature must fail with a clear error, never silently produce
  wrong output.
- **Measure performance changes.** A change that claims a speedup includes before and after numbers from
  `benchmarks/` or `dynalm benchmark`.
- **Record design choices.** Non-obvious choices get a short entry in
  [docs/design-decisions.md](docs/design-decisions.md) (Decision / Reason / Alternatives / Tradeoffs /
  Evidence).
- **Follow the code style.** It follows `.clang-format` (C++20; no exceptions in hot paths; `Status` /
  `Result<T>` for errors).

By contributing, you agree that your contributions are licensed under the [Apache License 2.0](LICENSE).

## Code of conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). Be kind and constructive.
