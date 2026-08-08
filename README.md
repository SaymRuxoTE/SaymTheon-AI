# SaymTheon: Bare-Metal C++ Inference Engine

Modern language models don't have to rely on massive frameworks. **SaymTheon** is a lightweight transformer inference engine written entirely in **C++20**, designed for efficient execution on edge hardware.

Built from scratch with a focus on performance, it avoids heavyweight ML frameworks and keeps the runtime as close to the hardware as possible.

## Features

* **Dependency-Free:** No PyTorch or external machine learning libraries required.
* **Custom Memory Management:** Uses a dedicated Arena Allocator to minimize allocations during inference.
* **SIMD Optimizations:** AVX2 and FMA implementations accelerate matrix multiplication on supported CPUs.
* **Native GGUF Support:** Loads and executes GGUF models directly through memory mapping.
* **KV Cache:** Efficient key-value caching enables fast auto-regressive text generation.

---
*Backed by Emergent Ventures. Formerly known as EDGE-AI.*

🔒 **Security Notice:** `saymtheon.com` is the only official domain of this project. Any correspondence from other domains claiming representation is unofficial.
