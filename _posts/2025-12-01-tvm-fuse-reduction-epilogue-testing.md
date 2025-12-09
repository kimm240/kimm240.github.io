---
title: '[TIR][Schedule] Add FuseReductionEpilogue primitive to fuse epilogue into reduction init - 5. 테스트 전략과 검증'
date: 2025-12-01
permalink: /posts/2025/12/tvm-fuse-reduction-epilogue-testing/
excerpt: 'Part 3까지 구현한 FuseReductionEpilogue 프리미티브가 올바르게 동작하는지 검증하기 위한 테스트 전략을 다룹니다. 구조적 동등성 검사, 수치 정확도 검사, 엣지 케이스 테스트, Trace Roundtrip 검사를 통해 프리미티브의 안정성을 확보합니다.'
categories:
  - TVM
tags:
  - TVM
  - 컴파일러
  - 최적화
  - TIR
  - Schedule
  - C++
  - 테스트
---

[Part 3]까지 FuseReductionEpilogue라는 새로운 스케줄링 프리미티브를 C++로 구현했습니다. 그 후, 스케줄링 프리미티브가 잘 작동하는지 테스트를 하였습니다.

## 1. 테스트 전략

### A. 구조적 동등성 검사 (Structural Equality)

가장 기본이 되는 테스트입니다. "변환된 결과 코드가 내가 예상한 TIR 코드와 100% 일치하는가?"를 확인합니다. 이는 컴파일러가 의도한 대로 AST를 변형했는지 검증합니다.

```python
def test_fuse_reduction_epilogue_basic():  
    # 1. 원본 스케줄 생성
    sch = tir.Schedule(matmul_bias_before, debug_mask="all")  
    
    # 2. 융합 수행
    sch.fuse_reduction_epilogue("multiply", "add")  
    
    # 3. 예상되는 결과(matmul_bias_expected)와 구조적 비교
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], matmul_bias_expected)  
    
    # 4. Trace Roundtrip 검증 (후술)
    verify_trace_roundtrip(sch=sch, mod=matmul_bias_before)
```

또한, int8 양자화 모델뿐만 아니라 일반적인 float32 모델에서도 작동하는지 확인하기 위해 `test_fuse_reduction_epilogue_fp32` 테스트도 추가하여 범용성을 확보했습니다.

### B. 수치 정확도 검사 (Numerical Correctness)

구조가 예쁘게 바뀌었더라도 계산 결과가 틀리면 아무 소용이 없습니다. NumPy를 이용해 실제 연산 결과가 오차 범위 내에서 일치하는지 검증합니다.

```python
def test_fuse_reduction_epilogue_numerical_correctness():  
    # 1. 원본(Before) 컴파일 및 실행 준비
    sch_original = tir.Schedule(matmul_bias_before, debug_mask="all")  
    mod_original = tvm.compile(sch_original.mod["main"], target="llvm")  

    # 2. 융합된(After) 스케줄 컴파일 및 실행 준비
    sch_fused = tir.Schedule(matmul_bias_before, debug_mask="all")  
    sch_fused.fuse_reduction_epilogue("multiply", "add")  
    mod_fused = tvm.compile(sch_fused.mod["main"], target="llvm")  

    # 3. 랜덤 데이터 생성 (Input Generation)
    A_np = np.random.randint(-128, 127, size=(16, 16), dtype="int8")  
    B_np = np.random.randint(-128, 127, size=(16, 16), dtype="int8")  
    C_np = np.random.randint(-1000, 1000, size=(16, 16), dtype="int32")  
    
    # 4. NumPy 정답 계산 (Oracle)
    expected = (A_np.astype("int32") @ B_np.T.astype("int32")) + C_np  

    # 5. TVM 실행 및 결과 비교
    # ... (TVM 실행 코드 생략) ...

    # 6. 검증: 원본 vs 융합본 vs 정답(NumPy)
    np.testing.assert_allclose(D_original, expected, rtol=1e-5)  
    np.testing.assert_allclose(D_fused, expected, rtol=1e-5)  
    np.testing.assert_allclose(D_fused, D_original, rtol=1e-5)
```

### C. 엣지 케이스: 다중 Epilogue 테스트 (Review Feedback)

MatMul의 결과가 여러 곳에서 쓰이는 경우(Multiple Consumers)를 검증하기 위해 `matmul_bias_multiple_epilogue_before` 함수를 정의했습니다. multiply 블록의 결과인 temp가 add 블록과 add2 블록 두 군데서 사용되는 시나리오입니다.

```python
@T.prim_func  
def matmul_bias_multiple_epilogue_before(...):  
    # ... (MatMul 수행) ...
    
    # Consumer 1: Bias Add 1
    for i, j in T.grid(16, 16):  
        with T.block("add"):  
            D[vi, vj] = temp[vi, vj] + C[vi, vj]  
            
    # Consumer 2: Bias Add 2
    for i, j in T.grid(16, 16):  
        with T.block("add2"):  
            E[vi, vj] = temp[vi, vj] + C[vi, vj]
```

이 테스트(`test_fuse_reduction_epilogue_multiple_epilogue`)를 통해, multiply와 add를 융합하더라도 남겨진 add2 블록이 깨지지 않고 정상적으로 동작함을 증명했습니다.

### D. Trace Roundtrip 검사

TVM은 사용자가 적용한 스케줄링 명령어들을 기록(Trace)하여 JSON 형태로 직렬화하거나 다시 복원할 수 있어야 합니다.

모든 테스트 케이스의 마지막에 `verify_trace_roundtrip(sch=sch, ...)`를 호출함으로써, FuseReductionEpilogue 프리미티브가 TVM의 스케줄링 히스토리 시스템에 완벽하게 통합되었음을 확인했습니다.

## 2. 테스트 결과

위의 네 가지 테스트 전략을 통해 FuseReductionEpilogue 프리미티브가 다음과 같이 검증되었습니다:

변환된 TIR 코드가 예상한 구조와 일치하고, 융합 전후의 계산 결과가 동일하며, 다중 Consumer 시나리오에서도 안정적으로 동작하고, TVM의 스케줄링 히스토리 시스템과 완벽하게 통합되었습니다.

이를 통해 프리미티브가 프로덕션 환경에서 사용할 수 있을 만큼 안정적이고 신뢰할 수 있음을 확인했습니다.

---

**시리즈 포스트**

- 이전: [Part 4. 아키텍처 시각화](/posts/2025/12/tvm-fuse-reduction-epilogue-architecture/)

**Language**: [English](/posts/2025/12/tvm-fuse-reduction-epilogue-testing-en/)

