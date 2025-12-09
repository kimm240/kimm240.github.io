---
title: '[Optimization][Operator] Implement and enable Conv2d-Reshape-Add-ReLU fusion'
date: 2025-12-01
permalink: /posts/2025/12/tvm-conv2d-reshape-add-relu-fusion-en/
excerpt: 'Conv2d + Bias + ReLU is the most common pattern in deep learning. However, when importing PyTorch models into TVM, Reshape nodes are inserted in the middle, breaking fusion. This post covers the implementation process of a pattern matching-based fusion pass to solve this problem.'
categories:
  - TVM
tags:
  - TVM
  - Compiler
  - Optimization
  - Relax
  - PyTorch
  - DNNL
  - Fusion
---

This document covers the process of solving the problem (Reshape node) that occurs when the pattern Conv2d + Bias + ReLU comes from PyTorch.

## 1. Overview

One of the most frequent patterns in deep learning models is Convolution -> Bias Add -> Activation(ReLU). Most high-performance deep learning libraries (DNNL, cuDNN, etc.) provide functionality to process this as a single kernel.

However, when importing PyTorch models into TVM, it was discovered that this fusion breaks due to unexpected structural issues. This report covers the process of optimizing by bundling the Conv2d-Reshape-Add-ReLU pattern into a single composite function so it executes as a single kernel in the DNNL backend.

## 2. Problem Situation

### 2.1. PyTorch Frontend

Generally, we think of Bias Add as simple addition. However, when the PyTorch frontend converts Conv2d(bias=True) to Relax IR, it inserts a Reshape operation in the middle to explicitly indicate broadcasting.

Expected IR: Conv2d -> Add -> ReLU
Actually converted IR: Conv2d -> Reshape (Bias) -> Add -> ReLU

### 2.2. Limitations of Existing Optimization (FuseOps)

TVM's general fusion pass `relax.transform.FuseOps` recognizes general Conv2d + Add patterns but does not recognize structures with Reshape inserted in the middle.

This caused the following performance degradation. 4 operations (Conv, Reshape, Add, ReLU) executed as separate kernels, causing kernel overhead, I/O costs occurred at each stage writing data to VRAM/RAM and reading it back, and DNNL etc. provide a fused kernel called `conv2d_bias_relu`, but TVM doesn't bundle the pattern, so it cannot be called, making backend acceleration impossible.

## 3. Solution

To solve this problem, we implemented a new fusion pass `FuseConv2dReshapeAddRelu` based on Pattern Matching.

### 3.1. Pattern Definition (Declarative Pattern Language)

Define a pattern that accurately specifies the structure of data flow, not just simple connections.

- Inputs: Data, Weight, Bias
- Flow: Conv2d(Data, Weight) → Reshape(Bias) (Shape change) → Add(Conv_Output, Reshaped_Bias) → ReLU(Add_Output)

### 3.2. Goal

When this pattern is found, bundle it as a Composite Function named `dnnl.conv2d_reshape_add_relu`. Afterwards, the `MergeCompositeFunctions` pass recognizes this and offloads it to the DNNL backend.

## 4. Implementation Details

### 4.1. Pattern Matching and Pass Implementation

File: `python/tvm/relax/transform/fuse_conv2d_reshape_add_relu.py`

This is logic that detects and bundles specific patterns using `FuseOpsByPattern`.

```python
def _conv2d_reshape_add_relu_pattern():
    # 1. Define wildcards (allow all inputs)
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    shape = wildcard() # Target shape of Reshape

    # 2. Define operation flow (DPL: Declarative Pattern Language)
    # Match Conv2d operation
    conv_out = is_op("relax.nn.conv2d")(data, weight, varg_default_wildcard=True)
    
    # [Core] Match Reshape operation applied to Bias
    reshaped_bias = is_op("relax.reshape")(bias, shape)
    
    # Match addition of Reshaped Bias and Conv result
    add_out = is_op("relax.add")(conv_out, reshaped_bias)
    
    # Match final ReLU
    relu_out = is_op("relax.nn.relu")(add_out)
    return relu_out, {...}, _check

@tvm.transform.module_pass(opt_level=0, name="FuseConv2dReshapeAddRelu")
class FuseConv2dReshapeAddRelu:
    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        # When pattern matches, fuse into Composite function named "dnnl.conv2d_reshape_add_relu"
        return relax.transform.FuseOpsByPattern(
            [("dnnl.conv2d_reshape_add_relu", *_conv2d_reshape_add_relu_pattern())],
            bind_constants=False,
        )(mod)
```

## 5. Validation

### 5.1. Testing Strategy

We wrote `tests/python/relax/test_conv2d_reshape_add_relu.py` to verify that the implemented pass works as intended.

Define Relax IR with the same structure (Conv -> Reshape -> Add -> ReLU) that PyTorch generates, execute the `FuseConv2dReshapeAddRelu` pass on that IR, and verify that the generated IR is converted to a single Composite Function instead of 4 individual operations, and that Codegen attributes are assigned.

### 5.2. Test Code

```python
def test_transform_pass():
    # ... (Initial IRModule definition: Conv2d -> Reshape -> Add -> ReLU) ...
    
    # Step 1: Apply fusion pass
    # At this stage, 4 operations should be bundled into one Composite function
    fused_mod = FuseConv2dReshapeAddRelu()(TestModule)
    
    # Step 2: Apply backend merge pass (MergeCompositeFunctions)
    # Final verification that Composite function is in a form that can be offloaded to DNNL etc.
    final_mod = tvm.ir.transform.Sequential([
        relax.transform.FuseConv2dReshapeAddRelu(),
        relax.transform.MergeCompositeFunctions(),
    ])(TestModule)
    # Verify through output IR that "dnnl.conv2d_reshape_add_relu" function is generated
    print(final_mod)
```

---

**Language**: [한국어 (Korean)](/posts/2025/12/tvm-conv2d-reshape-add-relu-fusion/)

