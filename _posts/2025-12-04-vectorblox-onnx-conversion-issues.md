---
title: 'VectorBlox VNNX 변환 이슈: Clip 및 ScatterND 연산자 제거'
date: 2025-12-04
permalink: /posts/2025/12/vectorblox-onnx-conversion-issues/
excerpt: 'ONNX 모델을 VectorBlox VNNX 형식으로 변환할 때 호환되지 않는 연산자들이 발생합니다. 이 포스트에서는 Clip 연산자와 ScatterND 연산자를 제거하고 대체하는 방법을 다룹니다.'
tags:
  - AI Accelerator
  - VectorBlox
  - ONNX
  - PyTorch
  - 모델 변환
categories:
  - VectorBlox
---

AI 모델을 VectorBlox의 `.vnnx` 형식으로 변환할 때, 일부 연산자[^1]가 VNNX에서 지원되지 않아 변환 오류가 발생합니다. 이 경우, 해당 연산자를 VNNX에서 지원하는 연산자로 교환해줘야 합니다. 

## 1. Clip

### 문제
Pytorch에서 ONNX 변환 시 생성되는 `Clip` 연산자는 VNNX에서 호환되지 않습니다.

### 해결 방안
- `Clip` 연산자는 입력 텐서의 각 요소를 지정된 최소값(`min`)과 최대값(`max`) 사이로 자르기(clipping) 위한 연산입니다. 수식으로 쓰면 다음과 같습니다.

$$output = \min(\max(input, min\_val), max\_val)$$

- Pytorch에서 ONNX로 변환할 때, `Clip` 연산자를 만든 부분을 찾습니다. 즉, Pytorch에서 `torch.clamp`를 찾습니다. 찾았다면, 이를 PyTorch에서 `torch.clamp` 대신 `min`과 `max` 연산을 조합하여 대체합니다.

### 코드 수정 전후 비교
```python
# 변환 전 (문제 발생)
bbox[:,:,0] = torch.clamp(x - w/2, 0, image_size[0] - 1)
```

```python
# 변환 후
bbox[:,:,0] = torch.minimum(torch.maximum(x -  w/2, zero), x_max)
```

## 2. ScatterND

### 문제
ONNX 변환 시 생성되는 `ScatterND` 연산자는 VNNX에서 호환되지 않습니다.

### 해결 방안
- `ScatterND`는 주어진 `data` 텐서를 복사한 뒤, `indices`가 지정한 위치에 `updates` 값을 덮어써서 새로운 텐서를 만드는 연산입니다. 
- 한편, ONNX에서 x[:,0] = 1처럼 기존 값을 바꾸는 일반 Assignment 명령어가 없습니다.
- 대신, 기존 x를 복사하여, 0번째 열만 1로 바뀐 새로운 x를 만드는 방식으로 처리해야 합니다. 따라서 `ScatterND` 연산자가 등장하게 됩니다.
- 즉, 특정 위치를 수정하는 방식이 아니고, `torch.stack`을 활용하여, 연산 결과를 쌓아서 저장하는 방식으로 바꿔줍니다.
- `torch.stack` 연산은 이후 `Concat` 또는 `Unsqueeze + Concat`연산으로 변환됩니다.

### 코드 수정 전후 비교
```python
# 변환 전 (문제 발생)
bbox[:,:,0] = ...
bbox[:,:,1] = ...
```

```python
# 변환 후
x1 = bbox[:,:,0]의 연산 결과
y1 = bbox[:,:,1]의 연산 결과
...
bbox = torch.stack((x1,y1,x2,y2), dim=-1)
```

---

[^1]: https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/docs/OPS.md

**Language**: [English](/posts/2025/12/vectorblox-onnx-conversion-issues-en/)