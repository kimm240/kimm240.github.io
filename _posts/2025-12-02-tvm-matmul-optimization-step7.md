---
title: 'TVM 행렬 곱셈 최적화 - Step 7: cuBLAS Comparison'
date: 2025-12-02
permalink: /posts/2025/12/tvm-matmul-optimization-step7/
excerpt: 'TVM으로 최적화한 행렬 곱셈 구현을 NVIDIA cuBLAS와 비교합니다. Step 6(Unrolling)에서 달성한 평균 1039 GFLOPS는 cuBLAS의 약 50%에 해당하며, 512x512 크기에서는 약 79% 수준의 성능을 달성했습니다.'
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
| TVM Step 6 | 1039 GFLOPS | 50.1% |
| cuBLAS (NVIDIA) | 2074 GFLOPS | 100% |


### 크기별 상세

| 크기 | TVM Step 6 (Unrolling) | cuBLAS | TVM/cuBLAS |
|------|------------------------|--------|------------|
| 512x512 | 1028 GFLOPS | 1302 GFLOPS | 약 **79.0%** |
| 1024x1024 | 1050 GFLOPS | 2846 GFLOPS | 약 **36.9%** |

## 분석

### TVM Step 6(Unrolling)의 성능 특성

**512x512 크기에서 약 79% 달성**:
- 작은 행렬 크기에서는 Step 1~5에서 적용한 Tiling, Shared Memory, Software Pipelining 위에 Loop Unrolling까지 더해져, cuBLAS 대비 상당히 근접한 성능을 보입니다.

**1024x1024 크기에서 약 37%**:
- 큰 행렬에서는 여전히 cuBLAS의 고급 최적화 기법(예: Tensor Core 활용, 더 공격적인 타일링/벡터화)이 더 효과적입니다.

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

