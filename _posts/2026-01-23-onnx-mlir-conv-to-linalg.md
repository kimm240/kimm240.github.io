---
title: 'ONNX Conv를 Linalg로 변환하기: conv_2d_nchw_fchw'
date: 2026-02-04
permalink: /posts/2026/02/onnx-mlir-conv-to-linalg/
excerpt: 'ONNX dialect의 Conv 연산을 Linalg dialect의 conv_2d_nchw_fchw로 변환하는 과정을 단계별로 설명합니다. 입력/속성/출력 매핑 방법과 패턴 구조 설계, 구현 과정을 상세히 다룹니다.'
tags:
  - MLIR
  - ONNX
  - Linalg
  - Compiler
  - Optimization
  - Convolution
categories:
  - MLIR
---

## 개요

ONNX dialect의 `Conv` 연산을 Linalg dialect의 `conv_2d_nchw_fchw`로 변환하는 과정을 단계별로 살펴보겠습니다. 여기서는 변환의 복잡도를 최소화하기 위해, Linalg dialect에서 가장 간단한 형태를 가진, `padding`=0,  `group`=1, `dilations`= [1,1]인 `conv_2d_nchw_fchw`를 변환의 목표로 했습니다.

## 연산자를 어떻게 Mapping할  것인가

### 입력 매핑

| ONNX | Linalg | 변환 방법 |
|------|--------|----------|
| `X` (NCHW) | `inputs[0]` (NCHW) | 직접 매핑 (레이아웃 동일) |
| `W` (FCHW) | `inputs[1]` (FCHW) | 직접 매핑 (레이아웃 동일) |
| `B` (None) | - | 지원하지 않음 (변환 거부) |

### 속성 매핑

| ONNX Attribute | Linalg Attribute | 변환 방법 |
|----------------|------------------|----------|
| `strides` (ArrayAttr) | `strides` (DenseIntElementsAttr) | `ArrayAttr` → `DenseIntElementsAttr` 변환 |
| `dilations` (ArrayAttr) | `dilations` (DenseIntElementsAttr) | `ArrayAttr` → `DenseIntElementsAttr` 변환 (현재는 [1,1] 고정) |
| `pads` | - | padding=0만 지원 (변환 불필요) |
| `group` | - | group=1만 지원 (변환 불필요) |
| `auto_pad` | - | "NOTSET"만 지원 (변환 불필요) |

### 출력 매핑

| ONNX | Linalg | 변환 방법 |
|------|--------|----------|
| `Y` (NCHW) | `outputs[0]` (NCHW) | `tensor.empty` + `linalg.fill`로 초기화 후 전달 |

## 구현 과정

### 패턴 구조 설계

MLIR에서 dialect 변환은 `Pattern Rewriting`을 통해 이루어집니다. `OpRewritePattern`을 상속받아 변환 로직을 구현합니다:

```cpp
struct ONNXConvOpLoweringToLinalg : public OpRewritePattern<ONNXConvOp> {
  LogicalResult matchAndRewrite(
      ONNXConvOp convOp, PatternRewriter &rewriter) const final {
    // 변환 로직
  }
};
```

### 속성 추출

ONNX Conv의 속성을 Linalg 형식으로 변환합니다.

```cpp
// Strides 추출 (기본값 [1, 1])
SmallVector<int64_t> strides = {1, 1};
auto stridesOpt = convOp.getStrides();
if (stridesOpt.has_value()) {
  ArrayAttr stridesAttr = stridesOpt.value();
  strides[0] = ArrayAttrIntVal(stridesAttr, 0);
  strides[1] = ArrayAttrIntVal(stridesAttr, 1);
}
auto stridesDenseAttr = rewriter.getI64TensorAttr(strides);

// Dilations: 고정값 [1, 1] (현재는 dilation=1만 지원)
auto dilationsDenseAttr = rewriter.getI64TensorAttr({1, 1});
```

### 출력 텐서 초기화

```cpp
// 1. 빈 텐서 생성
Value emptyTensor = tensor::EmptyOp::create(
    rewriter, loc, outputShape, outputTensorType.getElementType());

// 2. 0으로 초기화
Value zero = arith::ConstantOp::create(rewriter, loc,
    outputTensorType.getElementType(),
    rewriter.getZeroAttr(outputTensorType.getElementType()));

// 3. Fill 연산으로 0 채우기
Value filledTensor = linalg::FillOp::create(
    rewriter, loc, ValueRange{zero}, ValueRange{emptyTensor})
                         .getResult(0);
```

### Linalg Conv 연산 생성

마지막으로 `linalg.conv_2d_nchw_fchw` 연산을 생성합니다:

```cpp
Value convResult = linalg::Conv2DNchwFchwOp::create(rewriter, loc,
    TypeRange{outputTensorType},  // result type
    ValueRange{X, W},               // inputs: [input, filter]
    ValueRange{filledTensor},     // outputs: [init tensor]
    stridesDenseAttr,              // strides attribute
    dilationsDenseAttr)            // dilations attribute
                       .getResult(0);
rewriter.replaceOp(convOp, convResult);
```

## 결과

### 변환 전 (ONNX)

```mlir
%none = "onnx.NoValue"() : () -> none
%0 = "onnx.Conv"(%arg0, %arg1, %none) {
  dilations = [1, 1],
  group = 1 : si64,
  pads = [0, 0, 0, 0],
  strides = [1, 1]
} : (tensor<1x3x5x5xf32>, tensor<2x3x3x3xf32>, none) -> tensor<1x2x3x3xf32>
```

### 변환 후 (Linalg)

```mlir
[[ZERO:%.+]] = arith.constant 0.000000e+00 : f32
[[EMPTY:%.+]] = tensor.empty() : tensor<1x2x3x3xf32>
[[FILLED:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[EMPTY]] : tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32>
[[RESULT:%.+]] = linalg.conv_2d_nchw_fchw ins(%arg0, %arg1 : tensor<1x3x5x5xf32>, tensor<2x3x3x3xf32>) 
    outs([[FILLED]] : tensor<1x2x3x3xf32>) 
    {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} 
    -> tensor<1x2x3x3xf32>
```

---

**시리즈 포스트**

- 이전: [Mixed Linalg and ONNX Operations를 위한 Bufferization](/posts/2026/01/onnx-mlir-bufferization-mixed-linalg-onnx/)

**Language**: [English](/posts/2026/02/onnx-mlir-conv-to-linalg-en/)

