---
title: 'VectorBlox vnnx_tflite.py Fix: Problem 4 - INT8 Constant ADD/SUB Operation Issue Resolution'
date: 2025-12-06
permalink: /posts/2025/12/vectorblox-vnnx-tflite-add-sub-fix-en/
excerpt: 'Fixed quantization parameter uninitialization issues when processing ADD/SUB operations with constant operands in vnnx_tflite.py. Explains how to modify the code to initialize all quantization parameters even when multi_input=False.'
tags:
  - AI Accelerator
  - FPGA
  - VectorBlox
  - VNNX
  - TensorFlow Lite
  - Compiler
  - Bug Fix
categories:
  - VectorBlox
---

## Problem 4: INT8 Constant ADD/SUB Operation

### Compilation Failure Log

```
Assertion failed: (activation_min != 0)

File "vbx_cnn_model.c", line XXXX
```

### Root Cause

In the SPNv2 model, there are cases where one side of ADD or SUB operations is a constant. For example, bias addition like `x + 0.5` or normalization like `x - mean`.

In TFLite, such operations are represented with one input as a constant tensor. VectorBlox SDK determines this as `multi_input=False`.

On the other hand, when both are variables like `x + y`, VectorBlox SDK determines this as `multi_input=True`.

The problem is that when `multi_input=False`, VectorBlox SDK does not initialize quantization parameters (activation_min/max, multiplier, shift, etc.) for ADD/SUB. Therefore, when executing VNNX, the C layer finds that one of the quantization parameters, `activation_min`, is 0, causing an assertion failure during INT8 full quantization.

### Solution Process

We modified the code to initialize all quantization parameters even when `multi_input=False`.

Before modification, parameters were only set when `multi_input=True`:

```python
if subcode in ['ADD', 'SUB']:
    if multi_input:
        # Set quantization parameters
        sn.activation_min = -128
        sn.activation_max = 127
        # Set multiplier, shift, etc.
    # else: do nothing → error in C layer!
```

After modification, parameters are always set regardless of `multi_input`:

```python
if subcode in ['ADD', 'SUB']:
    # Always initialize quantization parameters
    sn.activation_min = -128
    sn.activation_max = 127
    
    # Set input/output scale, offset
    input_scale = i_tensor['quantization']['scale']
    output_scale = o_tensor['quantization']['scale']
    input_offset = i_tensor['quantization']['zero_point'][0]
    output_offset = o_tensor['quantization']['zero_point'][0]
    
    # Calculate and set multiplier, shift
    input_multiplier, input_shift = get_quantized_multiplier(input_scale)
    output_multiplier, output_shift = get_quantized_multiplier(output_scale)
    sn.input_multiplier = len(weights)
    weights += struct.pack(...)
    # ... (initialize all fields)
    
    # Additional settings for multi_input=False
    if not multi_input:
        # Also set filter_data, bias_data, etc.
        sn.eltwise8.left_shift = left_shift
        sn.eltwise8.swap_inputs = 0
        # ...
```

We ensured that all ELTWISE operations have the structure fields required by the C layer.

### Solution Result

After modification, ADD/SUB with constant RHS also execute normally:

```
[ADD with constant RHS]
All quantization parameters initialized
VNNX execution successful ✓
C layer assertion passed ✓
```

### Modification Summary

| Item | Before | After |
|------|--------|-------|
| Constant operation support | Parameters uninitialized when `multi_input=False` | Parameters initialized in all cases |
| Quantization parameters | Conditionally set | Always set |
| C layer compatibility | Assertion failure | Normal execution |
| Runtime stability | Unstable | Stable |

This modification enables all ADD/SUB operations, including constant operations, to execute stably.

---

**Series Posts**

- Previous: [Problem 3 - SLICE/STRIDED_SLICE 5D Tensor Processing](/posts/2025/12/vectorblox-vnnx-tflite-slice-fix-en/)
- Next: [Problem 5 - Constant Tensor Buffer Structure Compatibility](/posts/2025/12/vectorblox-vnnx-tflite-buffer-fix-en/)

**Language**: [한국어 (Korean)](/posts/2025/12/vectorblox-vnnx-tflite-add-sub-fix/)

