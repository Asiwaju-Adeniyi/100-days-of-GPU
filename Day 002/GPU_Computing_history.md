## Day 2: History of GPU Computing (from *Programming Massively Parallel Processors*

### What I Studied
Today I explored the **history of GPU computing**, tracing its evolution from fixed-function graphics pipelines to fully programmable parallel processors. Key milestones included:


- Early use of GPUs purely for graphics rendering.
- The shift to programmable shaders with APIs like DirectX and OpenGL.
- The emergence of GPGPU (General-Purpose computing on GPUs).
- NVIDIA’s launch of **CUDA in 2006**, marking a turning point for parallel computing.
- Rise of parallelism for AI, scientific computing, and blockchain.

### Key Learnings
- GPUs were originally designed for **graphics**, but their parallel structure made them ideal for high-throughput computing tasks.
- **GPGPU** was a clever hack — developers used graphics APIs to perform general computations before dedicated frameworks like CUDA existed.
- The **CUDA platform** enabled developers to write C-like code for GPUs, democratizing access to parallelism.
- GPU computing revolutionized fields like **deep learning, simulations**, and more recently **Web3 and AI Agents**.

## From Curiosity to Computation

When I began this challenge, I was just curious: Why are GPUs such a big deal? Over the last 100 days, that question evolved into a deeper understanding of how modern computing is driven by parallelism, architectural tradeoffs, and massive throughput.

---

## The Evolution of Processors: From Speed to Parallelism

### Moore’s Law and the Frequency Wall

For decades, Moore's Law predicted that we'd keep cramming more transistors onto chips. And for a while, that meant faster processors — higher clock speeds, year after year. But physics had other plans.

By the mid-2000s, we hit a wall: pushing clock speeds any further caused too much heat and power draw. Frequency scaling stalled, and single-thread performance plateaued.

### Enter: Multi-core Era

Instead of making one core faster, manufacturers started putting more cores on a chip. This shift kicked off the parallel computing revolution, where performance gains depend not on faster clocks, but smarter, distributed workloads.

---

## CPU vs. GPU: A Tale of Two Designs

### CPUs: Built for Responsiveness

CPUs are great at doing a few things very quickly. They're optimized for latency — fast response times, complex decision-making, and low-thread workloads.

- Sophisticated control logic  
- Large caches for quick access  
- High clock speeds  
- Limited parallelism  

### GPUs: Built for Scale

GPUs, on the other hand, are optimized for throughput — doing thousands of simpler tasks simultaneously. They trade individual task speed for massive parallel capacity.

- Hundreds or thousands of smaller cores  
- Simple control logic  
- Designed to hide latency by flooding the chip with work  
- Perfect for graphics and general-purpose computation  

---

## How GPUs Broke Out of Graphics

Initially, GPUs were specialized for rendering pixels. But developers soon realized that if you could render millions of pixels in parallel, you could compute millions of anything in parallel.

### CUDA: The Turning Point

In 2007, NVIDIA released CUDA, which opened the door to programming GPUs for general-purpose tasks. It was no longer about shaders and polygons — now, you could train neural networks, simulate physics, or crunch big data, all on a GPU.

---

## Parallelism Isn’t Magic: Amdahl’s and Gustafson’s Laws

### Amdahl’s Law: The Wall We Hit

If only part of a task can be parallelized, there’s a limit to how much faster it can go — no matter how many processors you throw at it.

> Example: If 90% of your code can run in parallel, the best you’ll get (even with infinite processors) is a 10× speedup.

### Gustafson’s Law: A More Hopeful View

But in real life, we often deal with growing problems — datasets that keep expanding. And as they grow, the parallel portion usually grows faster than the sequential one. That’s where GPU computing shines.

---

## Lessons from the Front Lines

Over the last 100 days, I worked on everything from simple CUDA kernels to deep learning training loops. Here’s what stood out:

- Thinking in parallel is hard — it requires a shift from step-by-step logic to data-wide operations  
- Memory access matters — bandwidth and locality are often bigger bottlenecks than raw computation  
- Threads are cheap on GPUs — use them liberally, but manage them carefully  
- Optimization is layered — warps, blocks, occupancy, and coalescing all play a role  

---

## Resources That Helped Me

- *Programming Massively Parallel Processors* by Hwu, Kirk, and Hajj  
- *CUDA by Example* – Sanders & Kandrot  
- NVIDIA Developer Blogs  
- Lectures from Stanford, MIT, and Udacity on GPU computing 


