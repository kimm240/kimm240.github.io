---
title: 'Converting ONNX Conv to Linalg: conv_2d_nchw_fchw'
date: 2026-02-04
permalink: /posts/2026/02/onnx-mlir-conv-to-linalg-en/
excerpt: 'We explain step-by-step how to convert ONNX dialect Conv operations to Linalg dialect conv_2d_nchw_fchw. We detail input/attribute/output mapping methods, pattern structure design, and implementation process.'
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

## Overview

We will examine step-by-step how to convert ONNX dialect `Conv` operations to Linalg dialect `conv_2d_nchw_fchw`. To minimize conversion complexity, we target `conv_2d_nchw_fchw`, which is the simplest form in Linalg dialect, with `padding`=0, `group`=1, and `dilations`=[1,1].

## How to Map Operators

### Input Mapping

| ONNX | Linalg | Conversion Method |
|------|--------|-------------------|
| `X` (NCHW) | `inputs[0]` (NCHW) | Direct mapping (same layout) |
| `W` (FCHW) | `inputs[1]` (FCHW) | Direct mapping (same layout) |
| `B` (None) | - | Not supported (conversion rejected) |

### Attribute Mapping

| ONNX Attribute | Linalg Attribute | Conversion Method |
|----------------|------------------|-------------------|
| `strides` (ArrayAttr) | `strides` (DenseIntElementsAttr) | `ArrayAttr` → `DenseIntElementsAttr` conversion |
| `dilations` (ArrayAttr) | `dilations` (DenseIntElementsAttr) | `ArrayAttr` → `DenseIntElementsAttr` conversion (currently fixed to [1,1]) |
| `pads` | - | Only padding=0 supported (conversion not needed) |
| `group` | - | Only group=1 supported (conversion not needed) |
| `auto_pad` | - | Only "NOTSET" supported (conversion not needed) |

### Output Mapping

| ONNX | Linalg | Conversion Method |
|------|--------|-------------------|
| `Y` (NCHW) | `outputs[0]` (NCHW) | Initialize with `tensor.empty` + `linalg.fill` then pass |

## Implementation Process

### Pattern Structure Design

In MLIR, dialect conversion is performed through `Pattern Rewriting`. We implement the conversion logic by inheriting from `OpRewritePattern`:

```cpp
struct ONNXConvOpLoweringToLinalg : public OpRewritePattern<ONNXConvOp> {
  LogicalResult matchAndRewrite(
      ONNXConvOp convOp, PatternRewriter &rewriter) const final {
    // Conversion logic
  }
};
```

### Attribute Extraction

Convert ONNX Conv attributes to Linalg format.

```cpp
// Extract strides (default [1, 1])
SmallVector<int64_t> strides = {1, 1};
auto stridesOpt = convOp.getStrides();
if (stridesOpt.has_value()) {
  ArrayAttr stridesAttr = stridesOpt.value();
  strides[0] = ArrayAttrIntVal(stridesAttr, 0);
  strides[1] = ArrayAttrIntVal(stridesAttr, 1);
}
auto stridesDenseAttr = rewriter.getI64TensorAttr(strides);

// Dilations: fixed value [1, 1] (currently only dilation=1 supported)
auto dilationsDenseAttr = rewriter.getI64TensorAttr({1, 1});
```

### Output Tensor Initialization

```cpp
// 1. Create empty tensor
Value emptyTensor = tensor::EmptyOp::create(
    rewriter, loc, outputShape, outputTensorType.getElementType());

// 2. Initialize with 0
Value zero = arith::ConstantOp::create(rewriter, loc,
    outputTensorType.getElementType(),
    rewriter.getZeroAttr(outputTensorType.getElementType()));

// 3. Fill with 0 using Fill operation
Value filledTensor = linalg::FillOp::create(
    rewriter, loc, ValueRange{zero}, ValueRange{emptyTensor})
                         .getResult(0);
```

### Linalg Conv Operation Creation

Finally, create the `linalg.conv_2d_nchw_fchw` operation:

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

## Results

### Before Conversion (ONNX)

```mlir
%none = "onnx.NoValue"() : () -> none
%0 = "onnx.Conv"(%arg0, %arg1, %none) {
  dilations = [1, 1],
  group = 1 : si64,
  pads = [0, 0, 0, 0],
  strides = [1, 1]
} : (tensor<1x3x5x5xf32>, tensor<2x3x3x3xf32>, none) -> tensor<1x2x3x3xf32>
```

### After Conversion (Linalg)

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

**Series Posts**

- Previous: [Bufferization for Mixed Linalg and ONNX Operations](/posts/2026/01/onnx-mlir-bufferization-mixed-linalg-onnx-en/)

**Language**: [한국어 (Korean)](/posts/2026/02/onnx-mlir-conv-to-linalg/)

