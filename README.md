# SaymTheon

### Lightweight Transformer Inference Engine for Edge AI

**SaymTheon** is a lightweight transformer inference engine built from scratch in **C++20**, designed for environments where memory, compute, latency, and deployment overhead matter.

Rather than placing a large machine-learning framework between the model and the hardware, SaymTheon focuses on the inference runtime itself — including model loading, memory management, transformer execution, KV caching, and CPU-level optimization.

> **The goal is not simply to run a model on an edge device.  
> The goal is to engineer the inference runtime around the device.**

---

## Overview

Modern AI systems are increasingly capable, but their inference stacks are often optimized for environments with abundant compute, memory, and power.

Edge systems operate under a different set of constraints:

- Limited RAM
- Limited CPU resources
- Strict latency requirements
- Restricted power budgets
- Offline or intermittent connectivity
- Hardware-specific optimization requirements
- Reduced deployment flexibility
- Predictable resource usage requirements

A model that performs well in a data center does not automatically translate into an efficient edge deployment.

**SaymTheon explores inference from the systems side.**

Instead of treating the inference engine as a thin execution layer around a model, SaymTheon treats the runtime itself as an engineering problem.

---

# Key Capabilities

| Capability | Description |
|---|---|
| **C++20 Runtime** | Native inference implementation with direct control over memory and execution |
| **GGUF Loading** | Native model loading and tensor mapping |
| **Memory Mapping** | Model data can be mapped directly into the runtime |
| **Arena Allocation** | Dedicated inference-oriented memory management |
| **KV Cache** | Reuse of attention key/value states during autoregressive generation |
| **AVX2 / FMA** | SIMD-oriented CPU optimization on supported hardware |
| **Autoregressive Generation** | Token-by-token transformer inference |
| **Dependency-Free Core** | No PyTorch or heavyweight ML runtime required |
| **Edge-Oriented Design** | Focus on memory, latency, compute, and deployment constraints |

> Feature availability may evolve as the runtime continues to develop.

---

# Architecture

At a high level, SaymTheon's inference stack is structured around several tightly coupled systems:

```text
                         MODEL
                           │
                           ▼
                  ┌─────────────────┐
                  │   GGUF Loader   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Model Metadata  │
                  │ Tensor Mapping  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Memory System   │
                  │ Arena Allocator │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   Transformer   │
                  │     Runtime     │
                  └────────┬────────┘
                           │
                 ┌─────────┴──────────┐
                 │                    │
                 ▼                    ▼
           ┌───────────┐      ┌─────────────────┐
           │ KV Cache  │      │ Optimized       │
           │           │      │ Compute Kernels │
           └───────────┘      └────────┬────────┘
                                       │
                                  AVX2 / FMA
                                       │
                                       ▼
                                      CPU
````

The architecture keeps the critical inference path under the control of the runtime rather than delegating model execution to a heavyweight external framework.

This allows memory behavior, tensor representation, execution flow, and CPU optimization to be considered together.

---

# Core Components

## 1. C++20 Inference Runtime

SaymTheon is implemented in **C++20**.

The choice of C++ is primarily about control.

The runtime can explicitly manage:

* Memory lifetime
* Object ownership
* Allocation behavior
* Tensor storage
* Data layout
* Buffer reuse
* CPU instructions
* Runtime dependencies
* Execution flow

The objective is not to eliminate abstraction entirely.

It is to ensure that the abstractions surrounding the critical inference path remain understandable and controllable.

---

## 2. Dependency-Free Core

SaymTheon does not rely on heavyweight machine-learning frameworks such as PyTorch for inference execution.

The runtime is designed as an independent inference implementation rather than a wrapper around an existing ML framework.

This provides several advantages for an edge-oriented system:

* Smaller dependency surface
* Greater control over runtime behavior
* Simpler deployment
* Reduced framework overhead
* Easier hardware-specific optimization
* More predictable resource usage

The emphasis is on building the execution path itself rather than assembling it from multiple high-level frameworks.

---

# Memory Architecture

Memory is one of the primary constraints in edge inference.

A transformer runtime may need memory for:

* Model weights
* Tensor metadata
* Intermediate activations
* Temporary buffers
* Attention state
* KV cache
* Runtime state
* Generation buffers

For this reason, SaymTheon treats memory management as part of the inference architecture rather than as an implementation detail.

---

## Arena Allocator

SaymTheon uses a dedicated **Arena Allocator** for inference-oriented memory management.

Instead of continuously creating and destroying individual allocations throughout execution, related runtime objects can be placed inside a managed memory region.

Conceptually:

```text
Traditional Allocation

malloc → Tensor
malloc → Buffer
malloc → Temporary
malloc → KV Cache
free
malloc
free
malloc
free
...


Arena Allocation

┌────────────────────────────────────────────┐
│ Tensor │ Buffer │ Temporary │ KV │ Runtime │
└────────────────────────────────────────────┘
                       │
                       ▼
              Managed Memory Region
```

The design targets two important properties:

### Lower Allocation Overhead

Repeated general-purpose allocations can introduce unnecessary work during execution.

An arena allows memory to be acquired from a controlled region instead.

### More Predictable Memory Behavior

For constrained systems, knowing where and how inference memory is allocated can be as important as the raw amount of memory being used.

The allocator therefore forms part of SaymTheon's broader goal of making inference behavior more explicit and predictable.

---

# CPU-Level Optimization

## AVX2 + FMA

SaymTheon includes CPU-level optimization using **AVX2** and **FMA** instructions on supported processors.

Transformer workloads contain operations that can benefit from SIMD execution, particularly arithmetic-heavy operations such as vector and matrix computations.

The execution path can therefore be viewed as:

```text
Transformer Operation
        │
        ▼
   Runtime Kernel
        │
        ▼
 SIMD Optimization
        │
   ┌────┴────┐
   ▼         ▼
 AVX2       FMA
   │         │
   └────┬────┘
        ▼
       CPU
```

The purpose is not simply to use SIMD instructions because they are available.

The larger objective is to bring computation closer to the capabilities of the target processor while keeping the runtime lightweight.

AVX2/FMA execution requires compatible CPU support.

---

# Native GGUF Model Loading

SaymTheon supports loading **GGUF** models directly through the inference runtime.

The loading path is designed around minimizing unnecessary transformation between the model stored on disk and the representation consumed by the runtime.

```text
                GGUF File
                    │
                    ▼
             File Mapping
                    │
                    ▼
               Metadata
                    │
                    ▼
             Tensor Mapping
                    │
                    ▼
             Runtime State
                    │
                    ▼
          Transformer Execution
```

GGUF metadata can provide the information required by the runtime to identify model configuration and tensor data.

Keeping model loading close to the execution layer also reduces the need for an additional model-conversion pipeline during deployment.

---

# KV Cache

SaymTheon implements **KV caching** for autoregressive generation.

During transformer inference, attention repeatedly accesses key and value states from previous tokens.

Without caching, previously computed states may need to be recomputed as the sequence grows.

With KV caching:

```text
Token 1
   │
   ▼
Compute ───────────────► KV Cache
                            │
Token 2                    │
   │                       │
   └──► Compute ───────────┤
                            │
Token 3                    │
   │                       │
   └──► Compute ───────────┤
                            │
Token N                    │
   │                       │
   └──► Compute ───────────┘
```

The fundamental trade-off is:

> **Less recomputation requires additional memory.**

This makes KV-cache design especially important for edge deployments.

A larger context can improve the amount of available conversational or document context while simultaneously increasing memory requirements.

For resource-constrained inference, KV caching is therefore both a performance mechanism and a memory-management problem.

---

# Inference Pipeline

A simplified autoregressive generation path looks like:

```text
                    Input
                      │
                      ▼
                 Tokenization
                      │
                      ▼
                   Token IDs
                      │
                      ▼
              ┌─────────────────┐
              │   Transformer   │
              │     Runtime     │
              └────────┬────────┘
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
        KV Cache          Optimized Kernels
                                 │
                              AVX2/FMA
                                 │
                                 ▼
                               CPU
                                 │
                                 ▼
                               Logits
                                 │
                                 ▼
                           Token Selection
                                 │
                                 ▼
                           Next Token
                                 │
                                 ▼
                             Generation
                                 │
                                 └──────────► Repeat
```

The runtime therefore combines:

**Model Representation + Memory Management + Transformer Execution + Hardware Optimization**

inside a single inference path.

---

# Design Principles

SaymTheon is built around several systems-level principles.

## 1. The Model Is Only One Part of the System

Model architecture and parameter count are only part of the inference problem.

The runtime executing the model can significantly affect:

* Latency
* Memory usage
* Throughput
* Deployment complexity
* Hardware utilization

---

## 2. Memory Is a Compute Constraint

Memory should not be considered separately from compute.

Large model weights, intermediate tensors, and KV cache all compete for the same finite resources.

A memory-efficient runtime can therefore improve the practical operating range of a model even when the underlying model remains unchanged.

---

## 3. Hardware Is Part of the Runtime

The CPU is not an invisible abstraction beneath inference.

Instruction-set support, cache behavior, memory bandwidth, SIMD width, and available RAM all influence real-world inference performance.

SaymTheon therefore considers hardware characteristics part of the runtime design.

---

## 4. Runtime Overhead Matters

A model can be small enough to fit inside an edge device while the surrounding software stack remains unnecessarily large.

The goal is therefore not simply:

> “Can the model run?”

but:

> **“How much infrastructure is required to make the model run?”**

---

## 5. Predictability Matters

Resource-constrained systems benefit from predictable behavior.

This includes:

* Allocation patterns
* Memory lifetime
* Compute paths
* Cache usage
* Model loading
* Runtime dependencies

Predictability becomes particularly important when inference is deployed in systems with strict resource limits.

---

## 6. Local Execution Matters

Edge inference can continue operating when:

* Network connectivity is unavailable
* Cloud latency is unacceptable
* Data should remain local
* Bandwidth is limited
* Remote inference is impractical

Local inference does not eliminate cloud computing.

It provides another execution boundary.

---

# Cloud + Edge

SaymTheon is not built around replacing cloud infrastructure.

Cloud and edge environments solve different classes of problems.

```text
                         CLOUD
              ┌─────────────────────────┐
              │ Large-scale computation │
              │ Centralized services    │
              │ Large model workloads   │
              │ Global infrastructure   │
              └────────────┬────────────┘
                           │
                    Synchronization
                           │
                           ▼
                          EDGE
              ┌─────────────────────────┐
              │ Local inference         │
              │ Low-latency execution   │
              │ Local data processing   │
              │ Offline capability      │
              │ Device-specific logic  │
              └─────────────────────────┘
```

A practical AI architecture can therefore distribute computation according to the capabilities and constraints of each environment.

SaymTheon focuses on the **edge inference layer** of this architecture.

---

# Performance

Performance engineering is an ongoing part of the project.

Current optimization targets include:

* Token generation latency
* Prompt processing latency
* Memory footprint
* Peak runtime memory
* Token generation throughput
* Allocation overhead
* KV-cache efficiency
* SIMD utilization
* Model loading performance
* CPU utilization

## Benchmarking Philosophy

Benchmark numbers are intentionally not presented until they are produced using a reproducible methodology.

A meaningful inference benchmark should specify at least:

| Metric                 | Description                             |
| ---------------------- | --------------------------------------- |
| **Tokens/sec**         | Generation throughput                   |
| **Prompt Processing**  | Input processing performance            |
| **Generation Latency** | Time required per generated token       |
| **Peak Memory**        | Maximum runtime memory usage            |
| **Model Load Time**    | Time required to initialize a model     |
| **Context Length**     | Sequence length used during testing     |
| **Model**              | Exact model and quantization            |
| **CPU**                | Processor and instruction-set support   |
| **RAM**                | Available system memory                 |
| **Build**              | Compiler and optimization configuration |

> **No benchmark numbers are presented here until they are measured under a reproducible test setup.**

This avoids presenting isolated performance numbers without the hardware and workload context required to interpret them.

---

# Hardware Considerations

SaymTheon's performance depends heavily on the target hardware and model configuration.

Important variables include:

* CPU architecture
* CPU frequency
* SIMD instruction support
* Memory bandwidth
* Available RAM
* Model parameter count
* Model quantization
* Context length
* KV-cache size
* Operating system
* Compiler optimization settings

### CPU Support

AVX2/FMA optimized execution requires a compatible processor.

On unsupported hardware, the corresponding optimized execution path cannot be used.

Hardware-specific optimization is intentionally treated as part of the runtime rather than as an afterthought.

---

# Project Structure

The repository is organized around the inference runtime and its supporting build infrastructure.

```text
SaymTheon-AI/
│
├── src/
│   ├── ...
│   └── ...
│
├── build.bat
├── README.md
├── LICENSE
└── .gitignore
```

The exact structure may evolve as additional runtime components are introduced.

The implementation intentionally remains close to the core inference path instead of being divided across a large framework hierarchy.

---

# Building

## Requirements

* **C++20-compatible compiler**
* Windows build environment
* AVX2-compatible CPU for AVX2 optimized execution
* FMA-compatible CPU for FMA optimized execution

## Build

The repository currently provides a Windows build script:

```bat
build.bat
```

The build configuration may evolve as SaymTheon gains additional platform and architecture support.

---

# Usage

A typical SaymTheon workflow follows this general sequence:

```text
1. Build the runtime
        │
        ▼
2. Provide a compatible GGUF model
        │
        ▼
3. Load model metadata
        │
        ▼
4. Map model tensors
        │
        ▼
5. Initialize runtime memory
        │
        ▼
6. Initialize KV cache
        │
        ▼
7. Run transformer inference
        │
        ▼
8. Generate tokens
```

Command-line interfaces and runtime invocation details may evolve alongside the implementation.

---

# Current Status

SaymTheon is an **actively developed inference-engine project**.

The current implementation focuses on:

* Transformer inference
* Native C++20 execution
* GGUF model loading
* Memory-mapped model data
* Custom arena-based memory management
* KV caching
* AVX2/FMA optimization
* Autoregressive generation
* Lightweight local deployment

The project is still under active development, and APIs, supported architectures, and internal runtime components may change.

---

# Roadmap

The roadmap is focused on improving both inference capability and systems-level efficiency.

### Runtime

* [ ] Expanded transformer/model compatibility
* [ ] Improved runtime abstractions
* [ ] More comprehensive inference testing
* [ ] Better error handling and diagnostics

### Performance

* [ ] Additional optimized kernels
* [ ] Improved SIMD utilization
* [ ] More aggressive memory reuse
* [ ] Improved KV-cache management
* [ ] Reduced runtime overhead

### Model Support

* [ ] Expanded GGUF compatibility
* [ ] Additional quantization formats
* [ ] Broader model architecture support
* [ ] Improved tensor compatibility

### Hardware

* [ ] Additional CPU architecture support
* [ ] Improved portability
* [ ] Further edge-device optimization
* [ ] Hardware-specific execution paths

### Benchmarking

* [ ] Reproducible benchmark suite
* [ ] Standardized performance reports
* [ ] Memory profiling
* [ ] Cross-hardware comparisons

### Platform

* [ ] Improved cross-platform build support
* [ ] Linux support improvements
* [ ] Additional compiler/toolchain support

> The roadmap is not a fixed specification. Implementation and benchmarking results may change the development priorities.

---

# Why Build an Inference Engine From Scratch?

Modern AI development often concentrates on the model.

SaymTheon focuses on what happens **after the model exists**.

A trained model is a mathematical representation of learned parameters.

An inference engine is the system responsible for turning those parameters into executable computation.

That distinction becomes particularly important when the target environment is:

* Resource constrained
* Latency sensitive
* Offline capable
* Hardware specific
* Memory limited
* Deployment sensitive

The central question is therefore not simply:

> **Can this model run?**

It is:

> **What does it take to run this model efficiently on the hardware that actually has to execute it?**

That question moves the problem from model selection alone into systems engineering.

---

# Research Direction

SaymTheon is part of a broader exploration of **AI systems, edge inference, and hybrid cloud-edge architectures**.

The project investigates questions such as:

* How much inference should remain local?
* How much runtime overhead is acceptable at the edge?
* How should inference memory be planned?
* How can CPU capabilities be utilized more efficiently?
* When does local inference become more practical than remote inference?
* How should AI systems behave when connectivity is unavailable?
* How can cloud and edge inference cooperate without making either side a single point of failure?
* How much of the inference stack can be optimized without sacrificing maintainability?

These questions sit at the intersection of:

**AI Engineering + Systems Programming + Edge Computing + Distributed Systems**

SaymTheon is primarily an engineering project, but its architecture is also intended to serve as a practical exploration of these systems-level questions.

---

# Project Background

SaymTheon was formerly developed under the name **EDGE-AI**.

The project has evolved from a general edge-inference concept into a dedicated C++ inference-runtime project focused on controlling the execution stack more directly.

---

# Security & Official Project Identity

The official project domain is:

**saymtheon.com**

Project-related communication should be verified against the official project identity before trusting external domains or correspondence claiming to represent SaymTheon.

---

# License

SaymTheon is released under the **MIT License**.

See [`LICENSE`](./LICENSE) for the full license text.

---

# Author

**Rüzgar Albayrak**

AI Systems · Edge AI · C++ Inference Engineering

---

# Final Principle

> **The model defines what the system has learned.**
>
> **The inference engine defines how that computation reaches the hardware.**

SaymTheon exists to explore what happens when the inference engine itself becomes a first-class engineering problem.

**Build the runtime around the hardware — not the other way around.**
