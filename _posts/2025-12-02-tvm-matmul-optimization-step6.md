---
title: 'TVM 행렬 곱셈 최적화 - Step 6: Loop Unrolling'
date: 2025-12-02
permalink: /posts/2025/12/tvm-matmul-optimization-step6/
excerpt: 'Loop Unrolling을 통해 1050 GFLOPS를 달성했습니다. 루프 오버헤드를 제거하고 Instruction-Level Parallelism을 향상시켜 최종 성능을 끌어올렸습니다.'
categories:
  - TVM
tags:
  - TVM
  - GPU
  - 최적화
  - 행렬곱셈
  - Loop Unrolling
  - ILP
---

# Step 6: Loop Unrolling

1050 GFLOPS 달성

## 성과

| Unroll Factor | 512x512 | 1024x1024 | 개선율 |
|---------------|---------|-----------|-----------|
| 4 | 1034 GFLOPS | 1038 GFLOPS | +0.9% |
| 8 | 940 GFLOPS | 1035 GFLOPS | +0.6% |
| **16** | 1028 GFLOPS | **1050 GFLOPS** | **+2.0%** |
| 32 | 1022 GFLOPS | 1041 GFLOPS | +1.2% |

최고 성능: 1050 GFLOPS (factor=16)

## 1. 컴파일러 이론: Loop Unrolling

### 루프 오버헤드 제거

루프는 매 반복마다 오버헤드가 있습니다.

다음과 같은 루프가 있다고 가정해봅시다.

```python
for (int k = 0; k < 32; k++) {
    C += A[k] * B[k];
}
```

- 루프가 반복될 때마다, 아래 명령어들을 실행합니다.
   - 카운터 증가: `k++`
   - 조건 비교: `k < 32`인가요?
   - 분기(Jump): 조건이 참이면 루프의 시작점으로 돌아가고, 거짓이면 루프를 종료하세요.
- 루프가 총 32번 반복되므로, `32 * 3 = 96`개의 명령어가 추가됩니다.
- Loop Unrolling은 이 반복의 틀을 펼쳐서 없애는 기술입니다.

위 루프에서, 16개의 명령어를 한 번에 펼치면(즉,  `Unroll Factor`가 16인 상황입니다.), 단 2번의 반복, 6개의 명령어만 발생합니다.

```cuda
// 첫 번째 블록 (16개 연산)
C += A[0] * B[0];
C += A[1] * B[1];
// ... (14개 더)
C += A[15] * B[15];

// 두 번째 블록
C += A[16] * B[16];
// ... (15개 더)
C += A[31] * B[31];
```

### Instruction-Level Parallelism (ILP) 향상

- 언롤링을 하여, 의존성이 없는 명령어들이 모인 블록을 만들 수 있습니다. 명령어 스케쥴러는 이 명령어들을 보고, 병렬적으로 작업을 스케쥴링할 수 있습니다.
- 하지만 언롤링된 코드가 크기가 커질 경우, 명령어 캐시(`Intruction Cache`)와 레지스터에 부담을 줘서 성능이 오히려 감소할 수 있습니다.
   - 실험 결과에서도, `Unroll Factor`가 `16`일 때의 성능이 `32`일 때의 성능보다 좋음을 확인할 수 있습니다.

## 2. TVM TensorIR 구현

### Unrolling Annotation

```python
# k_inner 루프 언롤링
sch.annotate(k_inner, "pragma_auto_unroll_max_step", 16)
sch.annotate(k_inner, "pragma_unroll_explicit", 1)
```

- `sch.annotate(k_inner, "pragma_auto_unroll_max_step", 16)`: `k_inner` 루프를 만났을 때, 컴파일러가 최대 `16`단계까지 자동 언롤링(`auto_unrolling`) 기능을 사용하도록 명령합니다.
- `sch.annotate(k_inner, "pragra_unroll_explicit", 1)`: 컴파일러가 자체적으로 언롤링이 비효율적이라 판단하고 지시를 무시할 수 있습니다. 따라서 해당 명령어가 언롤링 실행 명령을 명시적(`explicit`)하게 실행(`1`)하도록 합니다.

## 3. 결과

### 성능

Step 5 → Step 6:
- `1024x1024`: 1029 → 1050 GFLOPS (+2%)

## 실행

```bash
python test_individual/test_step6_unrolling.py
```

코드는 [https://github.com/kimm240/matrix-multiplication-optimization-with-tvm](https://github.com/kimm240/matrix-multiplication-optimization-with-tvm)에서 찾아볼 수 있습니다.

---

**시리즈 포스트**

- 이전: [Step 5: Software Pipelining](/posts/2025/12/tvm-matmul-optimization-step5/)
- 다음: [Step 7: cuBLAS Comparison](/posts/2025/12/tvm-matmul-optimization-step7/)

