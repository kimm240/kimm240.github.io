---
title: '[TIR][Schedule] Add FuseReductionEpilogue primitive to fuse epilogue into reduction init - 2. TIR 구조 변환 설계'
date: 2025-12-01
permalink: /posts/2025/12/tvm-fuse-reduction-epilogue-design/
excerpt: 'Part 1에서 확인한 기존 스케줄링 프리미티브의 한계를 넘어, Reduction Block의 초기값을 0이 아닌 Bias로 설정하는 새로운 접근 방식을 제안합니다. 이 포스트에서는 TIR 구조 변환 설계와 구현 요구사항을 다룹니다.'
categories:
  - TVM
tags:
  - TVM
  - 컴파일러
  - 최적화
  - TIR
  - Schedule
---

[Part 1]에서 우리는 기존의 스케줄링 프리미티브(compute_inline 등)가 Reduction Block의 특수성 때문에 이 문제를 해결할 수 없음을 확인했습니다. 단순히 두 블록을 물리적으로 이어 붙이려다 보니 충돌이 발생한 것입니다.

따라서 우리는 블록을 단순히 합치는 것을 넘어, 새로운 접근 방식이 필요합니다. 이번 포스트에서는 그 논리적 설계 과정을 다룹니다.

## 1. 개요

## 2. 아이디어

행렬 곱(MatMul)과 편향 덧셈(Bias Add)을 분리해서 처리하는 기존 방식의 흐름은 다음과 같습니다.

1. 임시 버퍼(temp)을 준비한다.
2. 임시 버퍼를 비운다 (temp = 0).
3. 임시 버퍼를 채운다 (temp += A * B).
4. D에 옮겨 담고 C를 추가한다 (D = temp + C).

이 과정에서 1번과 4번 단계가 비효율의 원인입니다. 이를 해결하기 위해, Reduction의 초기값(Accumulator Initializer)을 0이 아닌 Bias로 설정합니다.

### 수식의 변화

Before (임시 버퍼 사용):

1. 초기화:
   $$temp_{i,j} = 0$$

2. 누적:
   $$temp_{i,j} = temp_{i,j} + \sum_{k} A_{i,k} \times B_{j,k}$$

3. 최종 결과:
   $$D_{i,j} = temp_{i,j} + C_{i,j}$$

After (직접 누적):

1. Bias로 초기화:
   $$D_{i,j} = C_{i,j}$$

2. 직접 누적:
   $$D_{i,j} = D_{i,j} + \sum_{k} A_{i,k} \times B_{j,k}$$

## 3. TIR 구조 변환 설계

이 아이디어를 TVM의 중간 표현인 TIR(Tensor IR) 레벨에서 어떻게 구현할지 구체적으로 설계해 봅시다. 우리가 구현할 새로운 프리미티브 FuseReductionEpilogue는 다음 3단계 변환을 수행해야 합니다.

### [Step 1] 타겟 식별 (Pattern Matching)

모든 Reduction 블록을 변환할 수는 없습니다. 정확히 MatMul 결과에 무언가를 더하는(Add) 패턴인 경우에만 동작해야 합니다.

- Producer (Reduction): temp에 값을 누적하는 블록.
- Consumer (Epilogue): D = temp + C 형태의 단순 덧셈 블록.

### [Step 2] Init 구문 수정

가장 중요한 단계입니다. Reduction Block 내부의 T.init() 섹션을 찾아 수정합니다.

변경 전 (AS-IS):

```python
with T.init():
    temp[vi, vj] = 0  # 0으로 초기화
```
변경 후 (TO-BE):

```python
with T.init():
    D[vi, vj] = C[vi, vj]  # Bias(C) 값으로 초기화
```

이때 중요한 점은, 단순히 0을 C로 바꾸는 것뿐만 아니라, 쓰기 대상 버퍼가 temp에서 최종 출력 버퍼인 D로 바뀌어야 한다는 점입니다.

### [Step 3] 버퍼 대체 및 Epilogue 제거

Reduction Block에서도 temp 버퍼를 모두 D 버퍼로 교체해야 합니다.

```
temp[vi, vj] = temp[vi, vj] + ...
→ D[vi, vj] = D[vi, vj] + ...
```

작업이 완료되면, 더 이상 필요 없어진 temp 버퍼 할당과, 할 일이 없어진 Epilogue(add) 블록을 트리에서 완전히 제거합니다.

## 4. 구현을 위한 요구사항 분석

위 설계를 실제 컴파일러 코드로 옮기기 위해 필요한 기능들을 정리했습니다. Part 3에서 C++로 구현할 때 이 리스트가 체크리스트가 됩니다.

### Epilogue 패턴 분석기:

- Epilogue 블록의 수식이 정말 Output = Input + Addend 형태인지 파싱(Parsing) 해야 합니다.
- 덧셈 순서가 Input + Addend이든 Addend + Input이든 상관없이 동작해야 합니다 (교환법칙).

### Reduction Block 검증기:

- Producer가 T.init을 가진 완전한 Reduction Block인지 확인해야 합니다.

### Buffer Replacer (버퍼 교체기):

- AST(Abstract Syntax Tree)를 순회하며 특정 버퍼(temp)에 대한 로드/스토어를 다른 버퍼(D)로 대체하는 모듈이 필요합니다. TVM의 StmtExprMutator를 활용하면 될 것입니다.

### Index Remapping (인덱스 매핑):

- Epilogue 블록의 루프 인덱스(i, j)와 Reduction 블록의 루프 인덱스(i, j, k)가 서로 다르게 매핑되어 있을 수 있습니다. 이를 올바르게 연결해 주는 변수 매핑(Variable Mapping) 로직이 필요합니다.

## 5. 기대되는 결과

이 설계대로 구현된다면, 우리는 다음과 같은 최적화된 TIR 코드를 얻게 됩니다.

```python
# temp 버퍼 제거됨
for i, j, k in T.grid(16, 16, 16):
    with T.block("matmul_fused"):
        # 읽기/쓰기 의존성 명시
        T.reads(C[vi, vj], A[vi, vk], B[vj, vk])
        T.writes(D[vi, vj])
        
        with T.init():
            # 0 대신 Bias로 초기화하여 덧셈 연산 흡수
            D[vi, vj] = C[vi, vj]
            
        # 바로 최종 버퍼에 누적
        D[vi, vj] = D[vi, vj] + A[vi, vk] * B[vj, vk]
```

이것이 바로 제가 fuse_reduction_epilogue를 설계하게 된 배경입니다. 기존 툴을 탓하는 대신, Reduction Block의 특수성을 이해하고 이를 처리할 수 있는 새로운 도구를 만들기로 했습니다. 다음 포스트에서는 0으로 초기화하는 관습을 깨고, "Bias로 초기화하는" 아이디어를 어떻게 구현했는지 구체적인 가이드라인을 다루겠습니다.

---

**시리즈 포스트**

- 이전: [Part 1. 문제 분석과 기존 솔루션의 한계](/posts/2025/12/tvm-fuse-reduction-epilogue-overview/)
- 다음: [Part 3. C++ 구현](/posts/2025/12/tvm-fuse-reduction-epilogue-implementation/)

