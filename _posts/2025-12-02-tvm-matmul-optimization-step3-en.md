---
title: 'TVM Matrix Multiplication Optimization - Step 3: Shared Memory'
date: 2025-12-02
permalink: /posts/2025/12/tvm-matmul-optimization-step3-en/
excerpt: 'We achieved 101% performance improvement on large matrices (2048x2048) using Shared Memory. This post covers GPU memory hierarchy and caching strategies through Shared Memory, and Cooperative Fetching techniques.'
categories:
  - TVM
tags:
  - TVM
  - GPU
  - Optimization
  - Matrix Multiplication
  - Shared Memory
  - Caching
---

# Step 3: Shared Memory

## Results

| Matrix Size | Step 2 | Step 3 | Improvement |
|----------|--------|--------|--------|
| 512x512 | 466 GFLOPS | 415 GFLOPS | -11% |
| 1024x1024 | 482 GFLOPS | 460 GFLOPS | -5% |
| 2048x2048 | 222 GFLOPS | 446 GFLOPS | +101% |

Average: 440 GFLOPS

## 1. Compiler Theory: Memory Hierarchy Optimization

### GPU Memory Hierarchy and Role of Shared Memory

We can improve performance by maximizing the use of fast-access memory.

**A500 Memory Hierarchy**:

```
Global Memory (4 GB, 192 GB/s)
  - Large but slow
  - Shared by all SMs
    ↓
L2 Cache (2 MB)
  - Hardware managed (automatic)
  - Unpredictable
    ↓
Shared Memory (64 KB/SM)
  - Explicitly managed by programmer
  - Shared by all threads in block
  - 10-100x faster than global memory
    ↓
Registers (256 KB/SM)
  - Fastest
  - Per-thread dedicated
```

### Utilizing Shared Memory

Shared Memory utilizes this spatial locality:
Load data into Shared Memory in tile (32×32) units. By doing so, all threads (32) in the block can reuse it. And each element can be reused 32 times.

## 2. TVM TensorIR Implementation

### Shared Memory Allocation

```python
# Allocate Shared Memory buffers
A_shared = sch.cache_read(block_C, 0, "shared")
B_shared = sch.cache_read(block_C, 1, "shared")
```

- `sch.cache_read` creates a buffer in Shared Memory and loads data from Global Memory.
- `block_C` is the computation block, `0` and `1` are input indices (A and B).

### Cooperative Fetching

Multiple threads cooperate to load data into Shared Memory:

```python
# Cooperative fetching: 32 threads load 32x32 tile together
for k_outer in range(K // BK):
    # All threads cooperate to load A tile
    for i in range(32):
        A_shared[thread_id * 32 + i] = A_global[...]
    
    # Synchronize
    __syncthreads()
    
    # Compute using cached data
    for k_inner in range(BK):
        C_local += A_shared[...] * B_shared[...]
```

## 3. Results Analysis

Using Shared Memory doesn't always improve performance. This is because there are costs for copying from Global Memory to Shared Memory and synchronization costs for `__syncthreads()`. This can be seen in the results for 512x512 and 1024x1024.

However, for 2048×2048, we achieved +101% improvement. This is because the data to be cached exceeds the size of L2 cache (2 MB), increasing cache misses. At this time, Shared Memory acts as cache, significantly reducing access to global memory.

## Execution

```bash
python test_individual/test_step3_with_threads.py
```

Code can be found at [https://github.com/kimm240/matrix-multiplication-optimization-with-tvm](https://github.com/kimm240/matrix-multiplication-optimization-with-tvm).

---

**Series Posts**

- Previous: [Step 2: Tiling + Loop Reordering](/posts/2025/12/tvm-matmul-optimization-step2-en/)
- Next: [Step 4: Vectorization + Local Memory](/posts/2025/12/tvm-matmul-optimization-step4-en/)

**Language**: [한국어 (Korean)](/posts/2025/12/tvm-matmul-optimization-step3/)

