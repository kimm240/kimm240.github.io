---
title: '[Optimization][Operator] Implement and enable Conv2d-Reshape-Add-ReLU fusion'
date: 2025-12-01
permalink: /posts/2025/12/tvm-conv2d-reshape-add-relu-fusion/
excerpt: 'Conv2d + Bias + ReLU는 딥러닝에서 가장 흔한 패턴입니다. 하지만 PyTorch 모델을 TVM으로 가져올 때 Reshape 노드가 중간에 삽입되어 퓨전이 깨지는 문제가 발생합니다. 이 포스트는 이 문제를 해결하기 위한 패턴 매칭 기반 퓨전 패스 구현 과정을 다룹니다.'
tags:
  - TVM
  - 컴파일러
  - 최적화
  - Relax
  - PyTorch
  - DNNL
  - Fusion
---

이 문서는 Conv2d + Bias + ReLU라는 매우 흔한 패턴이 PyTorch에서 넘어올 때 발생하는 특유의 문제점(Reshape 노드)을 해결하는 과정을 담고 있습니다.

## 1. 개요

딥러닝 모델에서 가장 빈번하게 등장하는 패턴 중 하나는 Convolution -> Bias Add -> Activation(ReLU) 입니다. 대부분의 고성능 딥러닝 라이브러리(DNNL, cuDNN 등)는 이를 하나의 커널로 처리하는 기능을 제공합니다.

하지만 PyTorch 모델을 TVM으로 가져올 때(Import), 예상치 못한 구조적 문제로 인해 이 퓨전(Fusion)이 깨지는 현상이 발견되었습니다. 본 보고서는 Conv2d-Reshape-Add-ReLU 패턴을 하나의 복합 함수(Composite Function)로 묶어, DNNL 백엔드에서 단일 커널로 실행되도록 최적화한 과정을 다룹니다.

## 2. 문제 상황

### 2.1. PyTorch Frontend

일반적으로 우리는 Bias Add를 단순한 덧셈으로 생각합니다. 하지만 PyTorch 프론트엔드가 Conv2d(bias=True)를 Relax IR로 변환할 때, 브로드캐스팅(Broadcasting)을 명시하기 위해 Reshape 연산을 중간에 삽입합니다.

기대: Conv2d -> Add -> ReLU
실제 변환된 IR: Conv2d -> Reshape (Bias) -> Add -> ReLU

### 2.2. 기존 최적화(FuseOps)의 한계

TVM의 범용 퓨전 패스인 `relax.transform.FuseOps`는 일반적인 Conv2d + Add 패턴은 인식하지만, 중간에 **Reshape**이 끼어있는 구조는 인식하지 못했습니다.

이로 인해 다음과 같은 성능 저하가 발생했습니다.

- 커널 오버: 4개의 연산(Conv, Reshape, Add, ReLU)이 각각 별도의 커널로 실행됨.
-  메모리 대역폭 낭비: 각 단계마다 데이터를 VRAM/RAM에 썼다가 다시 읽어오는 I/O 비용 발생.
- 백엔드 가속 불가: DNNL 등은 `conv2d_bias_relu`라는 융합 커널을 제공하지만, TVM이 패턴을 묶어주지 않아 이를 호출할 수 없음.

## 3. 해결 방안

이 문제를 해결하기 위해 패턴 매칭(Pattern Matching) 기반의 새로운 퓨전 패스 `FuseConv2dReshapeAddRelu`를 구현했습니다.

### 3.1. 패턴 정의 (Declarative Pattern Language)

단순한 연결이 아니라, 데이터 흐름의 구조를 정확히 명시하는 패턴을 정의합니다.

- 입력: Data, Weight, Bias
- 흐름:
  1. Conv2d(Data, Weight)
  2. Reshape(Bias) (Shape 변경)
  3. Add(Conv_Output, Reshaped_Bias)
  4. ReLU(Add_Output)

### 3.2. 목표

이 패턴이 발견되면 `dnnl.conv2d_reshape_add_relu`라는 이름의 Composite Function으로 묶습니다. 이후 `MergeCompositeFunctions` 패스가 이를 인식하여 DNNL 백엔드로 오프로딩(Offloading)합니다.

## 4. 구현 상세

### 4.1. 패턴 매칭 및 패스 구현

파일: `python/tvm/relax/transform/fuse_conv2d_reshape_add_relu.py`

`FuseOpsByPattern`을 활용하여 특정 패턴을 감지하고 묶어주는 로직입니다.

```python
def _conv2d_reshape_add_relu_pattern():
    # 1. 와일드카드 정의 (모든 입력 허용)
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    shape = wildcard() # Reshape의 타겟 모양

    # 2. 연산 흐름 정의 (DPL: Declarative Pattern Language)
    # Conv2d 연산 매칭
    conv_out = is_op("relax.nn.conv2d")(data, weight, varg_default_wildcard=True)
    
    # [핵심] Bias에 적용되는 Reshape 연산 매칭
    reshaped_bias = is_op("relax.reshape")(bias, shape)
    
    # Reshape된 Bias와 Conv 결과의 덧셈 매칭
    add_out = is_op("relax.add")(conv_out, reshaped_bias)
    
    # 마지막 ReLU 매칭
    relu_out = is_op("relax.nn.relu")(add_out)
    return relu_out, {...}, _check

@tvm.transform.module_pass(opt_level=0, name="FuseConv2dReshapeAddRelu")
class FuseConv2dReshapeAddRelu:
    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        # 패턴이 매칭되면 "dnnl.conv2d_reshape_add_relu"라는 이름의 Composite 함수로 융합
        return relax.transform.FuseOpsByPattern(
            [("dnnl.conv2d_reshape_add_relu", *_conv2d_reshape_add_relu_pattern())],
            bind_constants=False,
        )(mod)
```

## 5. 검증

### 5.1. 테스트 전략

구현된 패스가 의도한 대로 동작하는지 확인하기 위해 `tests/python/relax/test_conv2d_reshape_add_relu.py`를 작성했습니다.

1. IR 생성: PyTorch가 생성하는 것과 동일한 구조(Conv -> Reshape -> Add -> ReLU)의 Relax IR을 정의합니다.
2. 패스 적용: `FuseConv2dReshapeAddRelu` 패스를 실행합니다.
3. 확인: 결과 IR이 4개의 개별 연산 대신 하나의 Composite Function으로 변환되었는지, 그리고 Codegen 속성이 부여되었는지 확인합니다.

### 5.2. 테스트 코드

```python
def test_transform_pass():
    # ... (초기 IRModule 정의: Conv2d -> Reshape -> Add -> ReLU) ...
    
    # Step 1: 퓨전 패스 적용
    # 이 단계에서 4개의 연산이 하나의 Composite 함수로 묶여야 함
    fused_mod = FuseConv2dReshapeAddRelu()(TestModule)
    
    # Step 2: 백엔드 병합 패스 적용 (MergeCompositeFunctions)
    # Composite 함수가 DNNL 등으로 오프로딩 가능한 형태인지 최종 확인
    final_mod = tvm.ir.transform.Sequential([
        relax.transform.FuseConv2dReshapeAddRelu(),
        relax.transform.MergeCompositeFunctions(),
    ])(TestModule)
    # 출력된 IR을 통해 "dnnl.conv2d_reshape_add_relu" 함수가 생성되었는지 확인
    print(final_mod)
```

