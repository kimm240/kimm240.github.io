---
title: 'TVM Matrix Multiplication Optimization - Step 3: Shared Memory'
date: 2025-12-02
permalink: /posts/2025/12/tvm-matmul-optimization-step3-en/
excerpt: 'This post covers Shared Memory caching and Cooperative Fetching. At 2048×2048 we see 101% improvement over Step 2; it explains GPU memory hierarchy and when Shared Memory pays off for larger matrices.'
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

Average (512, 1024): 437 GFLOPS

**Additional experiment (2048×2048)**:

| Matrix Size | Step 2 | Step 3 | Improvement |
|----------|--------|--------|--------|
| 2048x2048 | 222 GFLOPS | 446 GFLOPS | **+101%** |

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
Data is loaded into Shared Memory in tile (32×32) units. By doing so, all threads (32) in the block can reuse it.

## 2. TVM TensorIR Implementation

### Shared Memory Caching

```python
# Cache tiles in Shared Memory
A_shared = sch.cache_read(block, 0, "shared")
B_shared = sch.cache_read(block, 1, "shared")

# Place tiles in shared memory at k_outer level
sch.compute_at(A_shared, k_outer)
sch.compute_at(B_shared, k_outer)
```

The above TensorIR means the following.

```python
for k_outer:  # 32 tile iterations
    # Load 32×32 tiles into Shared Memory
    A_shared[32×32] = A_global[...]
    B_shared[32×32] = B_global[...]
    
    for i_elem, j_elem, k_inner:
        C[i,j] += A_shared[i,k] * B_shared[k,j]
```

### Cooperative Fetching

When loading 32×32 = 1024 elements into Shared Memory, 32 threads each load 32 elements.

```python
for cache_block in [A_shared, B_shared]:
    fused = sch.fuse(*loops[-2:])
    f_ty, f_tx = sch.split(fused, factors=[threads_y, None])
    sch.bind(f_tx, "threadIdx.x")
    sch.bind(f_ty, "threadIdx.y")
```

- `sch.fuse(*loops[-2:])`: `sch.fuse` merges multiple loops into one. Here it fuses the last two loops of `loops`.
- `sch.split(fused, factors=[threads_y, None])`: `split` divides the fused loop by `threads_y` (the block’s y dimension). This distributes work to match the GPU’s 2D thread layout.

After the above, the TensorIR structure looks like this.

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

However, in [Further experiment] 2048×2048, we improved +101%. The working set of data accessed simultaneously by multiple SMs cannot be reliably maintained in the L2 cache, increasing cache miss.

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

