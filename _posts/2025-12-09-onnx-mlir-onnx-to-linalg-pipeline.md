---
title: 'ONNXToLinalg 파이프라인 구축: MatMul 연산 변환 구현'
date: 2025-12-09
permalink: /posts/2025/12/onnx-mlir-onnx-to-linalg-pipeline/
excerpt: 'ONNX-MLIR에서 ONNX Dialect를 Linalg Dialect로 변환하는 파이프라인을 구축하는 과정을 다룹니다. 인프라 구축부터 MatMul 연산의 구체적인 변환 로직 구현, 그리고 IR 변환의 상세 과정까지 단계별로 설명합니다.'
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

ONNXToLinalg 파이프라인 구축을 위한 코드 변경은 변환 인프라를 새로 구축하고, MatMul 연산을 구현하며, 이를 전체 컴파일러에 통합하는 세 가지 주요 단계를 거쳤습니다.

---

## 1. 인프라 및 패스 등록

새로운 변환 패스가 컴파일러에 인식되도록 등록하는 작업입니다.

### 파일별 변경 내용

| 파일 | 변경 내용 | 역할 |
|------|----------|------|
| `src/Conversion/CMakeLists.txt` | `add_subdirectory(ONNXToLinalg)` 추가 | 새로운 ONNXToLinalg 디렉토리를 빌드 시스템에 포함시킵니다. |
| `src/Pass/Passes.hpp` | `createConvertONNXToLinalg()` 선언 | 새로운 변환 패스를 선언합니다. |
| `src/Tools/.../RegisterPasses.cpp` | `mlir::registerPass(...)` 호출 | `onnx-mlir-opt` 도구가 `--convert-onnx-to-linalg` 옵션으로 이 패스를 실행할 수 있도록 시스템에 등록합니다. |
| `src/Conversion/ONNXToLinalg/ConvertONNXToLinalg.cpp` | `ConvertONNXToLinalgPass` 정의 | ONNX Dialect의 함수(`func::FuncOp`)를 순회하며, 등록된 모든 ONNX → Linalg 패턴을 탐욕적(greedily)으로 적용하는 MLIR Pass를 정의합니다. |

---

## 2. 핵심 변환 로직 구현: MatMul.cpp

`onnx.MatMul`을 linalg 연산으로 대체하는 C++ 로직은 `src/Conversion/ONNXToLinalg/Math/MatMul.cpp`에 구현되었습니다.

### 변환 로직 단계

```cpp
// 1. [GUARD] 2D 랭크 확인: 2D x 2D가 아니면 변환을 거부하고 다음 패턴으로 넘어갑니다.
if (aType.getRank() != 2 || bType.getRank() != 2) {
    return rewriter.notifyMatchFailure(matMulOp, 
        "only 2D x 2D MatMul is currently supported...");
}

// 2. 출력 텐서 할당 (tensor.empty)
Value emptyTensor = tensor::EmptyOp::create(
    rewriter, loc, outputShape, outputTensorType.getElementType());

// 3. 0 상수 생성 (arith.constant)
Value zero = arith::ConstantOp::create(rewriter, loc,
    outputTensorType.getElementType(), 
    rewriter.getZeroAttr(outputTensorType.getElementType()));

// 4. 출력 텐서 0으로 채우기 (linalg.fill)
Value filledTensor = linalg::FillOp::create(
    rewriter, loc, ValueRange{zero}, ValueRange{emptyTensor})
                         .getResult(0);

// 5. 행렬 곱셈 연산 생성 (linalg.matmul)
Value matmulResult = linalg::MatmulOp::create(
    rewriter, loc, ValueRange{A, B}, ValueRange{filledTensor})
                         .getResult(0);

// 6. 원래 ONNX 연산을 Linalg 결과로 대체
rewriter.replaceOp(matMulOp, matmulResult);
```

### 변환 과정 설명

입력 텐서가 2D인지 확인하여 현재 지원하는 케이스만 처리합니다. 그 후 `tensor.empty` 연산으로 결과를 저장할 빈 텐서를 생성합니다.
`arith.constant`로 0 값을 생성하고,  `linalg.fill`로 출력 텐서를 0으로 채웁니다.
그리고 `linalg.matmul`로 실제 행렬 곱셈을 수행합니다. 이 연산은 초기화된 텐서에 결과를 누적합니다.
원래 `onnx.MatMul` 연산을 생성된 Linalg 연산 시퀀스로 대체합니다.

---

## 3. 실행 결과
### ONNX IR
행렬 $\mathbf{A} \in \mathbb{R}^{2 \times 3}$와 $\mathbf{B} \in \mathbb{R}^{3 \times 4}$의 곱셈을 정의합니다.

```
func.func @test_matmul_2d(%arg0 : tensor<2x3xf32>, %arg1 : tensor<3x4xf32>) -> tensor<2x4xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
```

### 변환된 Linalg

ONNXToLinalg 패스가 실행된 후의 결과는 다음과 같습니다.

```
// RUN: onnx-mlir-opt --convert-onnx-to-linalg ... | FileCheck %s
// CHECK-LABEL: test_matmul_2d
func.func @test_matmul_2d(...) -> tensor<2x4xf32> {
  // CHECK-DAG: [[ZERO:%.+]] = arith.constant 0.000000e+00 : f32
  // 0 상수 생성
  
  // CHECK-DAG: [[EMPTY:%.+]] = tensor.empty() : tensor<2x4xf32>
  // 출력 공간 할당
  
  // CHECK: [[FILLED:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[EMPTY]] : tensor<2x4xf32>) -> tensor<2x4xf32>
  // 1. 초기화: linalg.fill을 통해 출력 텐서를 0으로 채웁니다.
  
  // CHECK: [[RESULT:%.+]] = linalg.matmul ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<3x4xf32>) outs([[FILLED]] : tensor<2x4xf32>) -> tensor<2x4xf32>
  // 2. 계산: linalg.matmul이 0으로 초기화된 텐서([[FILLED]])에 결과를 누적합니다.
  
  // CHECK: return [[RESULT]] : tensor<2x4xf32>
}
```

---

**시리즈 포스트**

- 이전: [ONNX-MLIR의 Linalg Dialect 도입: 컴파일 흐름과 최적화 이점](/posts/2025/12/onnx-mlir-linalg-dialect/)

**Language**: [English](/posts/2025/12/onnx-mlir-onnx-to-linalg-pipeline-en/)

