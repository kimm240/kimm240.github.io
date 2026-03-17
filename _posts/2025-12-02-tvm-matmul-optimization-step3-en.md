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
// "1024 tasks in 8 chunks of 128"
for (f_ty in 0..7) {
    for (f_tx in 0..127) {
        // fused = f_ty * 128 + f_tx
        ...
    }
}
```

- `sch.bind(f_tx, "threadIdx.x")`: `bind` ties the `f_ty`, `f_tx` loops to the GPU’s thread IDs.

Final TensorIR structure (conceptual CUDA code):

```python
// Runs in parallel across 32 threads.

// 1. Each thread gets its hardware ID.
int tx = threadIdx.x; // 0..3
int ty = threadIdx.y; // 0..7

// 2. Compute this thread’s range of work (1024 elements / 32 threads = 32 per thread).
for (i = 0; i < 32; ++i) {
    int element_index = (ty * 4 + tx) * 32 + i;

    int src_ax0 = element_index / 32;
    int src_ax1 = element_index % 32;
    A_shared[src_ax0, src_ax1] = A_global[...];
}
```

## 3. Results Analysis

### Tiling and Shared Memory Usage: Why Occupancy Doesn’t Drop Much

Because we already applied **block-level tiling** in Step 2, each block’s working set is limited to 32×32 tiles. In Step 3 we only put those same tiles into Shared Memory.

- A_shared: 32×32 × 4 bytes = 4 KB  
- B_shared: 32×32 × 4 bytes = 4 KB  
- **Shared Memory per block ≈ 8 KB**

With about 64 KB of Shared Memory per SM on the A500, 8 KB per block means we can fit many blocks per SM in terms of shared memory alone. So **tiling keeps the shared-memory footprint small, and occupancy is not heavily limited by Shared Memory**; it is usually limited by registers or block size instead.

### Why 512/1024 Get Slower vs Why 2048 Gets Faster

Using Shared Memory does not always improve performance. There is extra cost from **explicit copies from Global to Shared Memory** and **`__syncthreads()`** synchronization.

- **512×512, 1024×1024**: Problem sizes are relatively small, so there are fewer blocks, and in Step 2 the L1/L2 caches already handle the tiles reasonably well. Here the **copy and sync overhead** outweighs the benefit of fewer cache misses, so Step 3 is slower.
- **2048×2048 (additional experiment)**: The three matrices (A, B, C) alone are 2048×2048×4×3 ≈ 48 MB, well beyond the L2 cache (2 MB). Required tiles often are not in L2, so cache misses dominate. Here Shared Memory acts as cache and greatly reduces global memory traffic, so the benefit outweighs the copy/sync overhead and we see about **+101%** improvement.

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

