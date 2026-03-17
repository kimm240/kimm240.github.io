---
title: 'TVM Matrix Multiplication Optimization - Step 7: cuBLAS Comparison'
date: 2025-12-02
permalink: /posts/2025/12/tvm-matmul-optimization-step7-en/
excerpt: 'We compare our TVM-optimized matrix multiplication implementation with NVIDIA cuBLAS. The average 1039 GFLOPS achieved in Step 6 (Unrolling) corresponds to about 50% of cuBLAS, and we achieve around 79% performance for 512x512 size.'
categories:
  - TVM
tags:
  - TVM
  - GPU
  - Optimization
  - Matrix Multiplication
  - cuBLAS
  - Benchmark
---

# Step 7: cuBLAS Comparison

## Results

### Overall Comparison (Average)

| Implementation | Performance | vs cuBLAS |
|------|------|-----------|
| NumPy (CPU) | 13 GFLOPS | 0.6% |
| TVM Step 6 | 1039 GFLOPS | 50.1% |
| cuBLAS (NVIDIA) | 2074 GFLOPS | 100% |

### Size-Specific Details

| Size | TVM Step 6 (Unrolling) | cuBLAS | TVM/cuBLAS |
|------|------------------------|--------|------------|
| 512x512 | 1028 GFLOPS | 1302 GFLOPS | **~79.0%** |
| 1024x1024 | 1050 GFLOPS | 2846 GFLOPS | **~36.9%** |

## Analysis

### Performance Characteristics of TVM Step 6 (Unrolling)

**Around 79% at 512x512 size**:
- For small matrices, the combination of Tiling, Shared Memory, Software Pipelining, and Loop Unrolling allows TVM to reach a high fraction of cuBLAS performance.

**Around 37% at 1024x1024 size**:
- For larger matrices, cuBLAS’s advanced optimizations (e.g., Tensor Core utilization, more aggressive tiling/vectorization) remain more effective.

## Execution

```bash
# cuBLAS benchmark
python benchmarks/cublas_baseline.py

# TVM vs cuBLAS comparison
python benchmarks/compare_all_with_cublas.py
```

Code can be found at [https://github.com/kimm240/matrix-multiplication-optimization-with-tvm](https://github.com/kimm240/matrix-multiplication-optimization-with-tvm).

---

**Series Posts**

- Previous: [Step 6: Loop Unrolling](/posts/2025/12/tvm-matmul-optimization-step6-en/)

**Language**: [한국어 (Korean)](/posts/2025/12/tvm-matmul-optimization-step7/)

