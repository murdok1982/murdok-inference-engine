# MuRDoK Inference Engine — Implementation Roadmap

```text
==========================================================================
                     MIE IMPLEMENTATION PHASES
==========================================================================
```

This document tracks the phased development of MuRDoK Inference Engine (MIE). Each phase must strictly follow the project's Golden Rule:
> **Hypothesis $\to$ Baseline Benchmark $\to$ Implementation $\to$ Verification Benchmark $\to$ Analysis $\to$ Accept/Reject.**

---

## Phase Overview

### Milestone M0: Baseline & Infrastructure Setup [IN PROGRESS]
- [x] Workspace and hardware analysis.
- [x] Repository initialization and directory structure layout.
- [x] Submodule integration of upstream `llama.cpp` (`third_party/llama.cpp`).
- [x] Comprehensive architecture documentation (`docs/ARCHITECTURE.md`).
- [x] Hardware detection module & `murdok-hardware` CLI (`src/hardware/`, `include/murdok/hardware.h`).
- [x] Unified benchmarking harness `murdok-bench` (`tools/murdok-bench/`).
- [ ] Reference benchmark run and documentation in `benchmarks/BASELINE.md`.

### Phase 1: MuRDoK Runtime Architecture & Encapsulation
- Implement stable C/C++ public API in `include/murdok/murdok.h`.
- Create `murdok::Runtime` orchestrator wrapping model initialization, context creation, and inference pipeline.
- Build interactive and one-shot `murdok` CLI.
- Verify exact parity with `llama.cpp` baseline before introducing algorithmic changes.

### Phase 2: Memory Allocator & Thread Affinity
- Implement `MuRDoKMemoryPool`: aligned 64-byte arena allocation, elimination of runtime malloc/free.
- Core affinity scheduler: distinguishing physical cores vs logical hyperthreads.
- Benchmark: measure memory fragmentation and cache miss reduction during prompt processing and token generation.

### Phase 3: Weight Memory Layout & Tiling
- Analyze weight tensor layouts for AVX2 and cache-line boundaries.
- Reorganize tensors to ensure sequential DRAM access and SIMD aligned strides.
- Benchmark: compare memory bandwidth saturation and tokens/sec.

### Phase 4: KV Cache Engine (Paged & Quantized)
- Transition KV cache to a paged block system (`kv/`).
- Support FP16, INT8, and INT4 adaptive quantization for long context sequences.
- Benchmark: KV memory reduction and context scaling up to 32k/64k tokens.

### Phase 5: Speculative Decoding
- Draft model + target model pipeline (`speculative/`).
- Dynamic draft token prediction based on historical acceptance rates.
- Benchmark: speedup factor across varying prompt distributions and generation temperatures.

### Phase 6: Static Graph Execution Planner
- Static dependency analysis and buffer allocation before starting inference (`graph/`).
- Operator fusion and optimized execution plan.
- Benchmark: runtime scheduling overhead reduction.

### Phase 7: Auto-Tuning Engine
- Automated calibration tool exploring optimal thread counts, batch sizes, prefetch distance, and KV precision.
- Profile persistence in `~/.murdok/profile.json`.

### Phase 8: MuRDoK Native Model Format (`.murdok`)
- Custom binary format specification with zero-overhead memory mapping (`compiler/`).
- `murdok-compile` tool converting `.gguf` to `.murdok`.

### Phase 9: Native Optimized Runtime
- Progressively reduce external dependencies where native, specialized kernels demonstrate measurable advantages.
