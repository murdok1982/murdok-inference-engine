# MuRDoK Inference Engine — Baseline Benchmarks

This document records the official baseline performance measurements against standard `llama.cpp` upstream on the target test environment.

## 1. Test Environment Specification

- **System OS**: Windows 10/11 (64-bit)
- **CPU**: Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz
- **Physical Cores**: 4
- **Logical Cores**: 8
- **Base / Boost Clock**: 1.60 GHz / 3.40 GHz
- **L1 / L2 / L3 Cache**: 32 KB D-Cache per core / 256 KB per core / 6 MB shared L3
- **SIMD Capabilities**: AVX2, FMA3, SSE4.2 (No AVX-512, No AMX)
- **System Memory**: 15.86 GB RAM (DDR4)
- **Integrated GPU**: Intel(R) UHD Graphics 620
- **Compiler**: Microsoft Visual Studio 2022 BuildTools (MSVC 19.44.35228.0 x64)
- **CMake**: 3.x
- **Build Configuration**: Release, AVX2 enabled (`/arch:AVX2`), OpenMP 2.0.
- **Reference Submodule**: `llama.cpp` commit `b23efaa`

---

## 2. Benchmark Methodology

All benchmarks are performed using `murdok-bench` under controlled conditions:
- **Reference Model**: `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (`qwen2.5-0.5b-instruct-q4_k_m.gguf`, 491.4 MB)
- **Context Size**: 2048 tokens
- **Batch Size**: 512 tokens
- **Warmup**: 1 run prior to measurement
- **Prompt**: Standard CPU cache locality prompt (48 tokens)
- **Generated Tokens**: 128 tokens
- **Sampler**: Greedy (`llama_sampler_init_greedy()`)

---

## 3. Official Baseline Measurements

| Benchmark ID | Model | Threads | Prompt tok/s | Gen tok/s | TTFT (ms) | TPOT (ms) | Peak RAM (MB) |
|---|---|---|---|---|---|---|---|
| **BASE-001 (Physical Cores)** | Qwen2.5-0.5B-Q4_K_M | **4** | 94.56 | **33.27** | 509.71 | **30.06** | 492.90 |
| **BASE-002 (Hyperthreading)** | Qwen2.5-0.5B-Q4_K_M | **8** | **117.10** | 30.75 | **412.16** | 32.52 | 492.83 |

### Architectural Insight from Baseline Data
1. **Autoregressive Generation (Bandwidth-Bound)**:
   - 4 Physical Cores achieved **33.27 tokens/sec**, while 8 Logical Hyperthreads dropped to **30.75 tokens/sec** (-7.6% degradation).
   - *Root cause*: Memory bus contention and L1/L2 cache line thrashing caused by hyperthread sibling contention during single-token decoding.
2. **Prompt Processing (Compute-Bound)**:
   - 8 Threads achieved **117.10 tokens/sec** (+23.8% faster than 4 threads at 94.56 tokens/sec).
   - *Root cause*: Matrix-multiplication during prompt ingest saturates ALU execution ports, benefiting from additional SMT dispatch pipelines.

---

## 4. Comparison Target for MuRDoK Engine (Phase 1 & 2)

| Metric | llama.cpp Baseline (4T Gen / 8T Prompt) | MuRDoK Engine Target | Target Optimization Mechanism |
|---|---|---|---|
| Prompt tok/s | 117.10 | > 125.00 | Thread scheduler dynamic core assignment |
| Generation tok/s | 33.27 | > 35.00 | Physical core pinning, cache prefetch & zero-alloc pool |
| TTFT (ms) | 412.16 | < 400.00 | Static graph pre-allocation |
| Peak Memory (MB) | 492.90 | < 480.00 | Aligned memory pool & paged KV |
