---
title: 'VectorBlox VNNX Conversion Issues: Removing Clip and ScatterND Operators'
date: 2025-12-04
permalink: /posts/2025/12/vectorblox-onnx-conversion-issues-en/
excerpt: 'When converting ONNX models to VectorBlox VNNX format, some operators are incompatible. This post covers methods to remove and replace Clip and ScatterND operators.'
tags:
  - AI Accelerator
  - VectorBlox
  - ONNX
  - PyTorch
  - Model Conversion
categories:
  - VectorBlox
---

When converting AI models to VectorBlox's `.vnnx` format, some operators[^1] are not supported in VNNX, causing conversion errors. In such cases, these operators must be replaced with VNNX-supported operators.

## 1. Clip

### Problem
The `Clip` operator generated during PyTorch to ONNX conversion is incompatible with VNNX.

### Solution
- The `Clip` operator clips each element of the input tensor between a specified minimum (`min`) and maximum (`max`) value. The formula is:

$$output = \min(\max(input, min\_val), max\_val)$$

- When converting from PyTorch to ONNX, find the part that creates the `Clip` operator, i.e., find `torch.clamp` in PyTorch. If found, replace it by combining `min` and `max` operations instead of `torch.clamp` in PyTorch.

### Code Before/After Comparison
```python
# Before (problem occurs)
bbox[:,:,0] = torch.clamp(x - w/2, 0, image_size[0] - 1)
```

```python
# After
bbox[:,:,0] = torch.minimum(torch.maximum(x -  w/2, zero), x_max)
```

## 2. ScatterND

### Problem
The `ScatterND` operator generated during ONNX conversion is incompatible with VNNX.

### Solution
- `ScatterND` copies a given `data` tensor and overwrites values at positions specified by `indices` with `updates` to create a new tensor.
- However, ONNX does not have a general assignment operator like `x[:,0] = 1` to modify existing values.
- Instead, it must create a new tensor by copying the original `x` and changing only the 0th column to 1. This is why the `ScatterND` operator appears.
- Instead of modifying specific positions, use `torch.stack` to stack and store operation results.
- The `torch.stack` operation is later converted to `Concat` or `Unsqueeze + Concat` operations.

### Code Before/After Comparison
```python
# Before (problem occurs)
bbox[:,:,0] = ...
bbox[:,:,1] = ...
```

```python
# After
x1 = operation result of bbox[:,:,0]
y1 = operation result of bbox[:,:,1]
...
bbox = torch.stack((x1,y1,x2,y2), dim=-1)
```

---

[^1]: https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/docs/OPS.md

**Language**: [한국어 (Korean)](/posts/2025/12/vectorblox-onnx-conversion-issues/)

