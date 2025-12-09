---
title: 'TVM Matrix Multiplication Optimization - Step 2: Tiling + Loop Reordering'
date: 2025-12-02
permalink: /posts/2025/12/tvm-matmul-optimization-step2-en/
excerpt: 'We achieved 481 GFLOPS through Tiling and Loop Reordering. This shows 5.1x performance improvement over Step 1. This post covers Tiling techniques for cache optimization and Loop Reordering to maximize register reuse.'
categories:
  - TVM
tags:
  - TVM
  - GPU
  - Optimization
  - Matrix Multiplication
  - Tiling
  - Loop Reordering
---

# Step 2: Tiling + Loop Reordering
Achieved 481 GFLOPS (5.1x improvement over Step 1)

## 1. Compiler Theory: Tiling and Loop Transformation

### Tiling (Blocking) - Loop Splitting for Cache

Tiling is a technique that divides large loops into small blocks (tiles). By doing so, we can increase the time data stays in fast-access memory (cache, Shared Memory).

Matrix Multiplication is basically the following triple loop:

```fortran
! Basic form
do i = 1,N
  do j = 1,N
    do k = 1,N
      C(i,j) = C(i,j) + A(i,k) * B(k,j)
    enddo
  enddo
enddo
```

- When `N` is large, matrices `A`, `B`, `C` don't all fit in cache. Therefore, every time data is fetched, it must be read from slow-access memory.

**Applying Tiling** - Split with tile size t:

```fortran
! Split j, k loops into tiles
do j = 1,N,t
  do k = 1,N,t
    do i = 1,N
      do jj = j, min(j+t-1,N)
        do kk = k, min(k+t-1,N)
          C(i,jj) = C(i,jj) + A(i,kk) * B(kk,jj)
        enddo
      enddo
    enddo
  enddo
enddo
```

- When loops are split with tile size `t`, the data processed at once becomes smaller (`N x N` -> `t × t`), so it can fit in cache.
- Transforming the iteration space in this way is called **Affine Transform**.

### Loop Reordering - Maximizing Data Reuse

By changing the loop order along with Tiling, we can maximize reuse in registers.

**Core Principle**:
- Variables in the innermost loop stay in registers continuously
- Placing k in the innermost makes `A[i,k]`, `B[k,j]`, `C[i,j]` all reside in registers

## 2. TVM TensorIR Implementation

GPU requires 2-level tiling:
1. **Block-level Tiling**: GPU block unit (considering Shared Memory size)
2. **Thread-level Tiling**: Workload per thread (considering register size)

### 2-Level Tiling

```python
# Block-level tiling (GPU block)
BM, BN, BK = 32, 32, 32  # Small tiles! (optimal for A500 cache)

# Thread-level tiling (work per thread)
TM, TN = 8, 4  # High ILP

threads_x = BM // TM  # 4
threads_y = BN // TN  # 8

# Create tiles with split
i_block, i_rest = sch.split(i, factors=[None, BM])
j_block, j_rest = sch.split(j, factors=[None, BN])
k_outer, k_inner = sch.split(k, factors=[None, BK])

i_thread, i_elem = sch.split(i_rest, factors=[threads_x, None])
j_thread, j_elem = sch.split(j_rest, factors=[threads_y, None])
```

- `sch.split` is a function that divides a long loop into multiple loops.
    - `i_block, i_rest = sch.split(i, factors=[None, BM])`: Fixes the inner loop size to `BM`(32), and asks the outer loop to be calculated automatically. (`None`)
    - `i_rest` becomes the inner loop that runs `BM` times, and `i_block` becomes the outer loop that runs `M / BM` times.
- At this time, the optimal values of BM, BN, BK, TM, TN were discovered through the process in Section 4.

## 3. Results

### Performance

| Matrix Size | Step 1 | Step 2 | Improvement |
|----------|--------|--------|------|
| 512x512 | 91 GFLOPS | 466 GFLOPS | 5.1x |
| 1024x1024 | 95 GFLOPS | 482 GFLOPS | 5.1x |
| 2048x2048 | - | 222 GFLOPS | - |

Average: 390 GFLOPS

### Optimal Configuration

```python
# Best performance setting (482 GFLOPS)
BM, BN, BK = 32, 32, 32  # Fit in L1 cache
TM, TN = 8, 4            # High ILP
Threads = 32 (4 x 8)     # High ILP with fewer threads
Pattern = "k_innermost"  # Maximize register reuse
```

## Execution

```bash
# Basic execution
python test_individual/test_step2_improved.py

# Parameter search (138 configurations)
python test_individual/step2_parameter_sweep.py
```

Code can be found at [https://github.com/kimm240/matrix-multiplication-optimization-with-tvm](https://github.com/kimm240/matrix-multiplication-optimization-with-tvm).

---

**Series Posts**

- Previous: [Step 1: Simple GPU Binding](/posts/2025/12/tvm-matmul-optimization-step1-en/)
- Next: [Step 3: Shared Memory](/posts/2025/12/tvm-matmul-optimization-step3-en/)

**Language**: [한국어 (Korean)](/posts/2025/12/tvm-matmul-optimization-step2/)

