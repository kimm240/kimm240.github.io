---
title: 'ONNXToLinalg Pipeline Construction: MatMul Operation Conversion Implementation'
date: 2025-12-09
permalink: /posts/2025/12/onnx-mlir-onnx-to-linalg-pipeline-en/
excerpt: 'This post covers the process of building a pipeline to convert ONNX Dialect to Linalg Dialect in ONNX-MLIR. We explain step-by-step from infrastructure setup to the specific conversion logic implementation of MatMul operations, and the detailed IR transformation process.'
tags:
  - MLIR
  - ONNX
  - Linalg
  - Compiler
  - Optimization
  - Deep Learning
categories:
  - MLIR
---

The code changes for building the ONNXToLinalg pipeline went through three main stages: building new conversion infrastructure, implementing the MatMul operation, and integrating it into the entire compiler.

---

## 1. Infrastructure and Pass Registration

This is the work of registering a new conversion pass so that the compiler recognizes it.

### Changes by File

| File | Changes | Role |
|------|---------|------|
| `src/Conversion/CMakeLists.txt` | Add `add_subdirectory(ONNXToLinalg)` | Includes the new ONNXToLinalg directory in the build system. |
| `src/Pass/Passes.hpp` | Declare `createConvertONNXToLinalg()` | Declares the new conversion pass. |
| `src/Tools/.../RegisterPasses.cpp` | Call `mlir::registerPass(...)` | Registers the pass in the system so that the `onnx-mlir-opt` tool can execute it with the `--convert-onnx-to-linalg` option. |
| `src/Conversion/ONNXToLinalg/ConvertONNXToLinalg.cpp` | Define `ConvertONNXToLinalgPass` | Defines an MLIR Pass that traverses functions (`func::FuncOp`) in the ONNX Dialect and greedily applies all registered ONNX → Linalg patterns. |

---

## 2. Core Conversion Logic Implementation: MatMul.cpp

The C++ logic to replace `onnx.MatMul` with linalg operations is implemented in `src/Conversion/ONNXToLinalg/Math/MatMul.cpp`.

### Conversion Logic Steps

```cpp
// 1. [GUARD] 2D rank check: If not 2D x 2D, reject conversion and move to next pattern.
if (aType.getRank() != 2 || bType.getRank() != 2) {
    return rewriter.notifyMatchFailure(matMulOp, 
        "only 2D x 2D MatMul is currently supported...");
}

// 2. Output tensor allocation (tensor.empty)
Value emptyTensor = tensor::EmptyOp::create(
    rewriter, loc, outputShape, outputTensorType.getElementType());

// 3. Zero constant creation (arith.constant)
Value zero = arith::ConstantOp::create(rewriter, loc,
    outputTensorType.getElementType(), 
    rewriter.getZeroAttr(outputTensorType.getElementType()));

// 4. Fill output tensor with zeros (linalg.fill)
Value filledTensor = linalg::FillOp::create(
    rewriter, loc, ValueRange{zero}, ValueRange{emptyTensor})
                         .getResult(0);

// 5. Matrix multiplication operation creation (linalg.matmul)
Value matmulResult = linalg::MatmulOp::create(
    rewriter, loc, ValueRange{A, B}, ValueRange{filledTensor})
                         .getResult(0);

// 6. Replace original ONNX operation with Linalg result
rewriter.replaceOp(matMulOp, matmulResult);
```

### Conversion Process Explanation

First, we check if the input tensors are 2D to handle only the currently supported cases. Then, we create an empty tensor to store the result using the `tensor.empty` operation.
We generate a zero value using `arith.constant` and fill the output tensor with zeros using `linalg.fill`.
Then, we perform the actual matrix multiplication using `linalg.matmul`. This operation accumulates the result into the initialized tensor.
Finally, we replace the original `onnx.MatMul` operation with the generated Linalg operation sequence.

---

## 3. Execution Results

### ONNX IR

Defines the multiplication of matrices $\mathbf{A} \in \mathbb{R}^{2 \times 3}$ and $\mathbf{B} \in \mathbb{R}^{3 \times 4}$.

```
func.func @test_matmul_2d(%arg0 : tensor<2x3xf32>, %arg1 : tensor<3x4xf32>) -> tensor<2x4xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
```

### Converted Linalg

The result after the ONNXToLinalg pass is executed is as follows.

```
// RUN: onnx-mlir-opt --convert-onnx-to-linalg ... | FileCheck %s
// CHECK-LABEL: test_matmul_2d
func.func @test_matmul_2d(...) -> tensor<2x4xf32> {
  // CHECK-DAG: [[ZERO:%.+]] = arith.constant 0.000000e+00 : f32
  // Zero constant creation
  
  // CHECK-DAG: [[EMPTY:%.+]] = tensor.empty() : tensor<2x4xf32>
  // Output space allocation
  
  // CHECK: [[FILLED:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[EMPTY]] : tensor<2x4xf32>) -> tensor<2x4xf32>
  // 1. Initialization: Fill output tensor with zeros using linalg.fill.
  
  // CHECK: [[RESULT:%.+]] = linalg.matmul ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<3x4xf32>) outs([[FILLED]] : tensor<2x4xf32>) -> tensor<2x4xf32>
  // 2. Computation: linalg.matmul accumulates the result into the zero-initialized tensor ([[FILLED]]).
  
  // CHECK: return [[RESULT]] : tensor<2x4xf32>
}
```

---

**Series Posts**

- Previous: [ONNX-MLIR Linalg Dialect Integration: Compilation Flow and Optimization Benefits](/posts/2025/12/onnx-mlir-linalg-dialect-en/)
- Next: [Detailed Pipeline Stages with useLinalgPath Enabled and End-to-End Validation](/posts/2026/01/onnx-mlir-linalg-path-pipeline-en/)

**Language**: [한국어 (Korean)](/posts/2025/12/onnx-mlir-onnx-to-linalg-pipeline/)

