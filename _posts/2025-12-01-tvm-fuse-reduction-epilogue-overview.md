---
title: '[TIR][Schedule] Add FuseReductionEpilogue primitive to fuse epilogue into reduction init - 1. 문제 분석과 기존 솔루션의 한계'
date: 2025-12-01
permalink: /posts/2025/12/tvm-fuse-reduction-epilogue-overview/
excerpt: '대부분의 AI 가속기는 Output = Input * Weight + Bias를 한 번의 사이클에 처리하는 MAC(Multiply-Accumulate) 명령어를 지원합니다. 하지만 컴파일러가 생성한 중간 코드(TIR)에서 이 두 연산이 분리되어 있다면, 하드웨어의 성능을 온전히 끌어낼 수 없습니다. 이 포스트에서는 TVM의 MatMul 블록에 Bias Addition을 인라인할 수 없는 문제를 분석하고, 기존 스케줄링 프리미티브의 한계를 살펴봅니다.'
categories:
  - TVM
tags:
  - TVM
  - 컴파일러
  - 최적화
  - TIR
  - Schedule
---

대부분의 AI 가속기는 Output = Input * Weight + Bias를 한 번의 사이클에 처리하는 MAC(Multiply-Accumulate) 명령어를 지원합니다. 하지만 컴파일러가 생성한 중간 코드(TIR)에서 이 두 연산이 분리되어 있다면, 하드웨어의 성능을 온전히 끌어낼 수 없습니다.

## 1. 개요

현재 TVM은 MatMul 블록에 Bias Addition을 인라인 할 수 없습니다. 기존의 강력한 스케줄링 도구들도 이 단순한 패턴 앞에서 무력합니다. 문제를 분석하고, 이를 어떻게 개선하면 될지 정리했습니다.

## 2. 문제 상황

우리가 최적화하고자 하는 코드는 다음과 같이 Reduction(곱셈 누적)과 Epilogue(덧셈)가 분리된 형태입니다.

```python
temp = T.alloc_buffer((16, 16), "int32")

# Block 1: MatMul (Reduction)
for i, j, k in T.grid(16, 16, 16):
    with T.block("multiply"):
        with T.init():
            temp[vi, vj] = 0 
        temp[vi, vj] = temp[vi, vj] + A[vi, vk] * B[vj, vk]

# Block 2: Bias Add (Epilogue)
for i, j in T.grid(16, 16):
    with T.block("add"):
        D[vi, vj] = temp[vi, vj] + C[vi, vj] 
```

temp 버퍼를 없애고, T.init 단계에서 0 대신 Bias(C)를 로드하여 하나의 블록으로 합쳐야 합니다. 하지만 현재 TVM은 해당 Scheduling 기법을 지원하지않습니다.

## 3. 기존 솔루션의 문제

### 시도 1: compute_inline (Producer → Consumer)

MatMul 블록(multiply)을 Bias Add 블록(add) 안으로 밀어 넣는 시도입니다.

```python
def compute_inline(self, block: Union[BlockRV, str]) -> None:
    """Inline a block into its consumer(s). It requires:
    1) The block is a complete non-root block...
    3) The body of the block must be a BufferStore statement in the form of, 
       ``A[i, j, k, ...] = ...`` where the indices of the LHS are all distinct atomic variables...
    """
```

Docstring의 제약 조건 3에  따르면, 인라인 대상 블록의 바디는 단순한 BufferStore 형태여야 합니다.

하지만, MatMul 블록은 Reduction Block입니다. 내부적으로 ```T.init()``` 구문을 포함하고 있으며, 자기 자신을 읽고 쓰는 (temp = temp + ...) 누적 구조를 가집니다. 이는 "단순 할당문(BufferStore statement)"이라는 조건을 만족하지 못합니다.

### 시도 2: reverse_compute_inline (Consumer → Producer)

반대로, Bias Add 블록(add)을 MatMul 블록(multiply) 안으로 가져오는 시도입니다.

```python
def reverse_compute_inline(self, block: Union[BlockRV, str]) -> None:
    """Inline a block into its only producer. It requires:
    3) The only producer of the block is a read-after-write producer and a
       complete non-root block
    4) The body of the block must be a BufferStore statement...
    """
```

Docstring에 따르면, Consumer 블록(add) 자체는 조건 4(단순 BufferStore)를 만족합니다. 하지만 조건 3에서 요구하는 Producer(multiply 블록)의 자격 요건이 문제입니다.

Producer인 MatMul 블록은 Reduction 축(k)을 가지고 있어, 루프가 완전히 끝나기 전에는 출력이 완성되지 않는 불완전한 상태(Incomplete State)를 가집니다. reverse_compute_inline은 Producer가 단순한 'Read-After-Write' 관계이길 기대하지만, Reduction Block은 이보다 훨씬 복잡한 의존성을 가집니다.

### 시도 3: decompose_reduction 후 인라인

그렇다면 Reduction을 init과 update로 쪼갠 뒤에 합치면 어떨까요?

```python
def decompose_reduction(self, block: Union[BlockRV, str], loop: LoopRV) -> BlockRV:
    """Decompose a reduction block into two separate blocks.
    a) The init block... inserted right before the given loop.
    b) The update block... original block without init statement.
    """
```

decompose_reduction을 수행하면 초기화 블록은 루프 바깥으로 나옵니다.

- init: 루프 밖 (0으로 초기화)
- update: 루프 안 (곱셈 누적)
- add: 루프 밖 (Bias 더하기)

add 블록은 update 루프가 모두 끝난 뒤에 실행되어야 합니다. add 블록을 update 루프 안으로 억지로 인라인한다면, 매 반복마다 Bias가 더해지는(수학적으로 틀린) 결과가 나옵니다.

## 4. 결론

위 분석을 통해 기존 프리미티브들은 Reduction Loop의 초기화(Init) 단계에 외부 연산을 주입하도록 설계되지 않았음을 확인했습니다. 각 함수의 Docstring에 적힌 complete non-root block이나 BufferStore statement 같은 제약 조건들은 이러한 복잡한 Reduction 패턴을 배제하고 있었습니다.

우리가 원하는 변환을 달성하려면 다음과 같은 조건을 만족하는 새로운 프리미티브가 필요합니다.

```python
def fuse_reduction_epilogue(self, reduction_block, epilogue_block):
    """
    1) The reduction block is a complete reduction block
    2) The epilogue block only reads from the reduction block's output
    3) The epilogue performs a simple addition: output = reduction_result + bias
    """
```

이것이 바로 제가 fuse_reduction_epilogue를 설계하게 된 배경입니다.

---

**시리즈 포스트**

- 다음: [Part 2. TIR 구조 변환 설계](/posts/2025/12/tvm-fuse-reduction-epilogue-design/)

**Language**: [English](/posts/2025/12/tvm-fuse-reduction-epilogue-overview-en/)
