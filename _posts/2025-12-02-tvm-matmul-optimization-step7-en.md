---
title: 'TVM Matrix Multiplication Optimization - Step 7: cuBLAS Comparison'
date: 2025-12-02
permalink: /posts/2025/12/tvm-matmul-optimization-step7-en/
excerpt: 'We compare our TVM-optimized matrix multiplication implementation with NVIDIA cuBLAS. The 1053 GFLOPS achieved in Step 6 corresponds to 50.7% of cuBLAS, and we achieved 85.6% performance for 512x512 size.'
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
| TVM Step 6 | 1053 GFLOPS | 50.7% |
| cuBLAS (NVIDIA) | 2074 GFLOPS | 100% |

### Size-Specific Details

| Size | TVM Step 6 | cuBLAS | TVM/cuBLAS |
|------|-----------|--------|------------|
| 512x512 | 1115 GFLOPS | 1302 GFLOPS | **85.6%** |
| 1024x1024 | 990 GFLOPS | 2846 GFLOPS | 34.8% |

## Analysis

### Performance Characteristics of TVM Step 6

**Achieved 85.6% at 512x512 size**:
- TVM's optimization techniques work effectively on small matrix sizes
- Tiling, Shared Memory, Software Pipelining fit well

**34.8% at 1024x1024 size**:
- cuBLAS's advanced optimization techniques are more effective on large matrices
- cuBLAS includes additional optimizations such as Tensor Core utilization

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

