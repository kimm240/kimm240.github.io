---
title: 'TVM 행렬 곱셈 최적화 - Step 5: Software Pipelining'
date: 2025-12-02
permalink: /posts/2025/12/tvm-matmul-optimization-step5/
excerpt: 'Software Pipelining을 통해 1029 GFLOPS를 달성했습니다. 메모리 레이턴시를 연산으로 은폐하여 평균 58% 성능 향상을 달성했습니다. 이 포스트에서는 여러 반복을 겹쳐서 실행하는 Software Pipelining 기법을 다룹니다.'
tags:
  - TVM
  - GPU
  - 최적화
  - 행렬곱셈
  - Software Pipelining
---

# Step 5: Software Pipelining

1029 GFLOPS 달성

## 성과

| 행렬 크기 | Step 5 | 개선율 |
|----------|--------|--------|
| 256x256 | 855 GFLOPS | +68% |
| 512x512 | 1020 GFLOPS | +27% |
| 1024x1024 | 1029 GFLOPS | +74% |

평균: 968 GFLOPS

## 1. 컴파일러 이론: Software Pipelining

### 기본 아이디어 - 여러 반복을 동시에

**문제**: 순차 실행은 비효율적

```
반복 0: Load (400 cycles) → Compute (100 cycles) → Store
반복 1:                      Load (400) → Compute (100) → Store
반복 2:                                     Load (400) → Compute (100)

총 시간 = N × (400 + 100) = 500N cycles
```

- 순차적으로 실행할 경우. 처리해야 할 메모리가 오기를 기다리는 동안 GPU 코어가 놀고 있습니다. (IDLE)

**해결**: Software Pipelining - 여러 반복을 겹쳐서 실행합니다.

```
반복 0: Load ────────────────→
반복 1:         Load ──────→ Compute ──→
반복 2:                 Load ──→ Compute ──→ Store
반복 3:                         Load ──→ Compute ──→ Store

총 시간 ≈ max(Load, Compute) × N = 400N cycles
절약: 100N cycles
```

- 메모리 이동 시간에 연산을 중첩하여 처리합니다.

## 2. TVM TensorIR 구현

### Pipeline Annotation

TVM은 annotation으로 파이프라이닝을 구현합니다.

- 파이프라인은 `Load`, `Compute`, `Writeback`을 겹쳐서 실행합니다. 여기서 `Load`는 `0`번 stage, `Compute`는 `1`번 stage, `Writeback`은 `2`번 stage입니다.

```python
# k_outer 안의 4개 블록에 stage 할당
sch.annotate(k_outer, "software_pipeline_stage", [0, 0, 1, 2])

# 실행 순서
sch.annotate(k_outer, "software_pipeline_order", [0, 1, 2, 3])
```

- `sch.annotate(k_outer, "software_pipeling_stage", [0, 1, 2, 3])`: `k_outer` 루프가 파이프라이닝의 대상입니다. `"software_pipeline_stage"` Directive로 각 작업이 어느 stage에 속하는지 지정할 것이라 명시합니다.
   - Stage 0: Load (A_shared, B_shared) - 다음 타일 미리 로드
   - Stage 1: Compute (C_local) - 현재 타일로 연산
   - Stage 2: Writeback (C_local → C) - 결과 쓰기

- `sch.annotate(k_outer, "software_pipeline_order", [0, 1, 2, 3])`: 작업의 순서를 정하는 명령어입니다. `[0, 1, 2, 3]`은 `k_outer` 루프 내 4개 블록을 원래 코드에 있던 순서대로 실행하라는 의미입니다.

- 파이프라이닝은 다음과 같이 이루어집니다.

| 시간 | Stage 0 (L) | Stage 1 (C) | Stage 2 (WB) |
|--------|----------------|--------|--------------|
| **T1** | `k=0` 타일 로드 |||
| **T2** | `k=1` 타일 로드 | `k=0` 타일 연산 | |
| **T3** | `k=2` 타일 로드 | `k=1` 타일 연산 | `k=0` 결과 쓰기 |
| **T4** | `k=3` 타일 로드 | `k=2` 타일 연산 | `k=1` 결과 쓰기 |

## 3. 생성된 실행 패턴

### Prologue-Kernel-Epilogue 구조

```python
# Prologue: 첫 타일 미리 로드
k=0: 
  Load A_tile[0], B_tile[0] to Shared Memory
  __syncthreads()

# Kernel: 여러 stage 동시 실행
for k in 1..31:
  Stage 0: Load A_tile[k], B_tile[k]     # 다음 타일
  Stage 1: Compute using A_tile[k-1], B_tile[k-1]  # 현재 타일
  Stage 2: Writeback C_local[k-2] → C   # 이전 결과
  __syncthreads()

# Epilogue: 마지막 처리
k=32:
  Compute using A_tile[31], B_tile[31]
  Writeback
```

## 4. 실험: Double Buffering

```python
# Double Buffering
sch.annotate(A_shared, "double_buffer_scope", 0)
sch.annotate(B_shared, "double_buffer_scope", 0)
```

결과:

| 크기 | Basic Pipeline | + Double Buffer | 차이 |
|------|---------------|-----------------|------|
| 1024x1024 | 1029 GFLOPS | 1028 GFLOPS | -0.1% |

결론: A500의 작은 shared memory에서는 비효율적입니다. 기존 기법보다 2배의 공간을 사용해야 하므로, Occupancy가 감소합니다.

## 5. 결과

### 성능 분석

Step 4 → Step 5:
- 평균 +58% 향상

왜 이렇게 효과적일까?

A500의 병목:
- 작은 메모리 대역폭 (192 GB/s)
- 메모리 레이턴시가 성능 제한 요소

Software Pipelining의 효과:
- 메모리 레이턴시를 연산으로 완전히 은폐
- 대역폭 완전 활용

## 실행

```bash
python test_individual/test_step5_pipelining.py
```

코드는 [https://github.com/kimm240/matrix-multiplication-optimization-with-tvm](https://github.com/kimm240/matrix-multiplication-optimization-with-tvm)에서 찾아볼 수 있습니다.

---

**교재 참조**: Muchnick Ch.17.4, Dragon Book Ch.11.9

**핵심 기법**: Software Pipelining (4-stage)

**성과**: 614 → 1029 GFLOPS (+68%)

