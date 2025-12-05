---
title: 'TVM 행렬 곱셈 최적화 - Step 7: cuBLAS Comparison'
date: 2025-12-02
permalink: /posts/2025/12/tvm-matmul-optimization-step7/
excerpt: 'TVM으로 최적화한 행렬 곱셈 구현을 NVIDIA cuBLAS와 비교합니다. Step 6에서 달성한 1053 GFLOPS는 cuBLAS의 50.7%에 해당하며, 512x512 크기에서는 85.6%의 성능을 달성했습니다.'
categories:
  - TVM
tags:
  - TVM
  - GPU
  - 최적화
  - 행렬곱셈
  - cuBLAS
  - 벤치마크
---

# Step 7: cuBLAS Comparison

## 결과

### 전체 비교 (평균)

| 구현 | 성능 | cuBLAS 대비 |
|------|------|-----------|
| NumPy (CPU) | 13 GFLOPS | 0.6% |
| TVM Step 6 | 1053 GFLOPS | 50.7% |
| cuBLAS (NVIDIA) | 2074 GFLOPS | 100% |


### 크기별 상세

| 크기 | TVM Step 6 | cuBLAS | TVM/cuBLAS |
|------|-----------|--------|------------|
| 512x512 | 1115 GFLOPS | 1302 GFLOPS | **85.6%** |
| 1024x1024 | 990 GFLOPS | 2846 GFLOPS | 34.8% |

## 분석

### TVM Step 6의 성능 특성

**512x512 크기에서 85.6% 달성**:
- 작은 행렬 크기에서는 TVM의 최적화 기법이 효과적으로 작동
- Tiling, Shared Memory, Software Pipelining이 잘 맞음

**1024x1024 크기에서 34.8%**:
- 큰 행렬에서는 cuBLAS의 고급 최적화 기법이 더 효과적
- cuBLAS는 Tensor Core 활용 등 추가 최적화 포함

## 실행

```bash
# cuBLAS 벤치마크
python benchmarks/cublas_baseline.py

# TVM vs cuBLAS 비교
python benchmarks/compare_all_with_cublas.py
```

코드는 [https://github.com/kimm240/matrix-multiplication-optimization-with-tvm](https://github.com/kimm240/matrix-multiplication-optimization-with-tvm)에서 찾아볼 수 있습니다.

---

**시리즈 포스트**

- 이전: [Step 6: Loop Unrolling](/posts/2025/12/tvm-matmul-optimization-step6/)

**Language**: [English](/posts/2025/12/tvm-matmul-optimization-step7-en/)

