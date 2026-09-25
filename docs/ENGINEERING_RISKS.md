# MuRDoK Inference Engine — Engineering Risks & Mitigation Strategies

**Version**: 0.4.0  
**Focus**: Safety, Portability, Robustness, and Undefined Behavior Prevention

---

## 1. Identified Engineering Risks & Mitigation Matrix

| Risk ID | Category | Severity | Description | Mitigation Strategy | Status |
|---|---|:---:|---|---|:---:|
| **RSK-01** | **Portability / ISA** | **CRITICAL** | Global `/arch:AVX2` or `-mavx2` flags cause illegal instruction (`SIGILL`) crashes on CPUs without AVX2 (older Intel, AMD, ARM64, macOS Apple Silicon). | Remove global compiler flags. Enable AVX2 only on dedicated translation units. Use runtime CPUID feature detection and dynamic function pointer dispatch. | **RESOLVED** |
| **RSK-02** | **Security / CLI** | **HIGH** | `murdok` CLI previously executed commands like `server` and `bench` via `system(cmdline.c_str())`, exposing the system to shell injection, path vulnerabilities, and child process management issues. | Eliminate `system()` calls. Refactor `tools/murdok/main.cpp` to execute server, benchmark, hardware detection, and compilation directly as in-process C++ functions. | **RESOLVED** |
| **RSK-03** | **Transparency / Integrity** | **HIGH** | Calling a packaging format a "native inference format" when the runtime falls back to GGUF creates technical deception and ruins scientific credibility. | Accurately classify `.murdok` as a container/packaging format. Provide transparent diagnostics when loading files and explicit error reporting if corrupt. | **RESOLVED** |
| **RSK-04** | **Concurrency / Thread Safety** | **HIGH** | Concurrent requests to an un-synchronized inference context cause memory corruption, race conditions in KV caches, and sampler crashes. | Enforce request-level serialization using `std::mutex` in `murdok-server`. Document single-threaded context execution rules. | **RESOLVED** |
| **RSK-05** | **Memory / Out of Bounds** | **MEDIUM** | SIMD vector kernels operating on vector lengths not divisible by 8 or unaligned buffer pointers cause heap corruption or segmentation faults. | Implement strict tail element handling ($N \pmod 8 \neq 0$) using scalar accumulation. Add explicit 64-byte alignment checks and unaligned fallbacks. | **RESOLVED** |
| **RSK-06** | **Filesystem / Model Ingestion** | **MEDIUM** | Ingesting corrupted, truncated, or malicious model files with invalid metadata offsets can cause integer overflows or segmentation faults during parsing. | Implement strict bounds checking, file size verification, magic number validation, and corrupt header detection in `murdok_compiler.cpp` and `murdok_format.h`. | **RESOLVED** |
| **RSK-07** | **Network / Exposure** | **MEDIUM** | Binding API servers to `0.0.0.0` by default exposes unauthenticated inference endpoints to local network adversaries. | Default binding is strictly `127.0.0.1` (localhost). Exposing to `0.0.0.0` requires explicit user configuration via `--host 0.0.0.0`. | **RESOLVED** |

---

## 2. Deep-Dive Risk Analysis

### 2.1. RSK-01: Dynamic Instruction Set Dispatch vs Static Flags
Compiling with `/arch:AVX2` allows MSVC to emit AVX2 instructions everywhere, including startup routines and standard library helpers. If run on a non-AVX2 host (such as an old x86 server or in a virtualized container without AVX2 passthrough), the CPU raises an Invalid Opcode fault before `main()` can even check `HardwareDetector`.

**Mitigation Implemented**:
1. Base compilation targets standard x86-64 / portable flags.
2. Vector math kernels are isolated in `src/kernels/simd_avx2.cpp`.
3. The kernel dispatcher queries CPUID at startup. If `hw.simd.avx2 && hw.simd.fma` is false, it returns the portable scalar implementation.

### 2.2. RSK-02: Elimination of Shell Subprocess Invocations
Using `system()` in C++ is a known anti-pattern. If a user provides arguments containing shell meta-characters (`;`, `&`, `|`, `` ` ``), command execution can be hijacked.

**Mitigation Implemented**:
All CLI commands (`run`, `compile`, `server`, `bench`, `hardware`, `optimize`) are integrated directly as linked C++ modules within the single `murdok` executable, eliminating child shell processes entirely.

### 2.3. RSK-04: Multi-Threaded Inference Isolation
In `llama.cpp`, a `llama_context` is stateful: it maintains a single KV cache and sampler state. Calling `llama_decode()` concurrently from multiple threads on the same context corrupts the KV cache.

**Mitigation Implemented**:
In `murdok-server`, access to the shared `Engine` instance is protected by a `std::mutex engine_mutex`. Future scaling will deploy an `EnginePool` with multiple isolated contexts for concurrent multi-user serving.
