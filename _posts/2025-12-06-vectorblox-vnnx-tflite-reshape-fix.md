---
title: 'VectorBlox vnnx_tflite.py 수정: 문제 2 - RESHAPE 연산 문제 해결'
date: 2025-12-06
permalink: /posts/2025/12/vectorblox-vnnx-tflite-reshape-fix/
excerpt: 'vnnx_tflite.py에서 RESHAPE 연산 처리 시 발생하는 multi-axis squeeze 및 single-axis squeeze 문제를 해결했습니다. VectorBlox SDK가 지원하지 않는 reshape 패턴을 사전에 감지하여 NOP 연산으로 처리하는 방법을 설명합니다.'
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

## 문제 2: RESHAPE 연산

### 컴파일 실패 로그

RESHAPE 연산에서는 두 가지 다른 종류의 에러가 발생했습니다:

에러 1:
```
ValueError: cannot handle multi-axis squeeze
```

에러 2:
```
ValueError: axes don't match array
```

### 문제 원인

첫 번째 에러는 Multi-axis Squeeze 문제입니다. SPNv2 모델은 5차원 텐서 `[1, 4, 4, 9, 4]`를 3차원 `[1, 144, 4]`로 변환하려고 했습니다. 이는 높이, 너비, 앵커 세 개의 축을 동시에 하나의 위치 축으로 병합하는 것입니다. 하지만, VectorBlox SDK의 `channels_first_array_reshape` 함수가 단일 축 squeeze만 지원하여 컴파일에 실패했습니다.

두 번째 에러는 Single-axis Squeeze의 모호성 문제입니다. SPNv2의 분류 헤드는 `[1, 12276, 1]` 형태의 텐서를 `[1, 12276]`로 변환합니다. 마지막 차원(클래스 수)을 제거하는 단순한 연산입니다. 그러나 VectorBlox SDK는 텐서가 `[N, C, H, W]` 형식이라고 가정합니다. 이런 상황 때문에, 축을 재배치하려 할 때 axes 계산이 불가하여 컴파일에 실패했습니다.

### 해결 과정

해결책은 문제가 되는 reshape 패턴들을 `channels_first_array_reshape` 함수에 전달하기 전에 미리 감지하여 처리하는 것입니다.

수정 전에는 모든 RESHAPE를 하나의 함수로 처리하려 했습니다. 코드는 단순히 입력과 출력 shape를 `channels_first_array_reshape`에 전달했고, 이 함수가 지원하지 않는 패턴을 만나면 바로 에러가 발생했습니다.

```python
elif subcode in ['SQUEEZE', 'EXPAND_DIMS', 'RESHAPE']:
    ishape = i_tensor['shape']
    oshape = o_tensor['shape']
    
    # 모든 경우를 한 함수로 처리
    mode = channels_first_array_reshape(ishape, transform)
    # SPNv2 패턴에서 실패
```

수정 후에는 세 가지 문제 패턴을 사전에 감지하는 로직을 추가했습니다.

첫 번째 패턴은 5차원에서 3차원으로의 변환입니다. 입력과 출력의 shape를 분석하여 첫 번째 차원(배치)과 마지막 차원(키포인트)이 보존되고, 중간 세 개 차원의 곱이 출력의 두 번째 차원과 일치하는지 확인합니다(`ishape[1] * ishape[2] * ishape[3] == oshape[1]`). 이는 multi-axis squeeze 패턴입니다.

두 번째 패턴은 3차원에서 2차원으로의 변환입니다. 앞 두 차원이 보존되고 마지막 차원의 크기가 1인 경우(`ishape[2] == 1`)를 감지합니다. 이는 크기 1인 차원을 제거하는 단순한 squeeze입니다.

세 번째 패턴은 반대로 2차원에서 3차원으로 크기 1인 차원을 추가하는 expand 연산입니다(`oshape[2] == 1`).

```python
elif subcode in ['SQUEEZE', 'EXPAND_DIMS', 'RESHAPE']:
    ishape = i_tensor['shape']
    oshape = o_tensor['shape']
    
    # 패턴 1: 5D → 3D
    if (len(ishape) == 5 and len(oshape) == 3 and 
        ishape[0] == oshape[0] and 
        ishape[1] * ishape[2] * ishape[3] == oshape[1] and 
        ishape[4] == oshape[2]):
        mode = 0
        sn.nop = 1
    
    # 패턴 2: 3D → 2D
    elif (len(ishape) == 3 and len(oshape) == 2 and 
          ishape[0] == oshape[0] and 
          ishape[1] == oshape[1] and 
          ishape[2] == 1):
        mode = 0
        sn.nop = 1
    
    # 패턴 3: 2D → 3D
    elif (len(ishape) == 2 and len(oshape) == 3 and 
          ishape[0] == oshape[0] and 
          ishape[1] == oshape[1] and 
          oshape[2] == 1):
        mode = 0
        sn.nop = 1
    else:
        # 일반 reshape → 기존 로직
        mode = channels_first_array_reshape(ishape, transform)
```

이 세 패턴에 해당하면 `channels_first_array_reshape`를 호출하지 않고 직접 처리합니다. 나머지 일반적인 reshape는 기존 로직으로 전달됩니다.

### 해결 결과

수정 후 모든 RESHAPE 패턴이 정상적으로 컴파일됩니다:

```
[RESHAPE [1,4,4,9,4] → [1,144,4]]
DBG: Multi-axis squeeze detected → NOP 처리

[RESHAPE [1,12276,1] → [1,12276]]
DBG: Single-axis squeeze detected → NOP 처리

컴파일 성공
```

이 수정을 통해 VectorBlox SDK가 지원하지 않는 reshape 패턴들을 사전에 감지하고 NOP 연산으로 처리할 수 있게 되었습니다.

---

**시리즈 포스트**

- 이전: [문제 1 - TRANSPOSE 연산](/posts/2025/12/vectorblox-vnnx-tflite-transpose-fix/)
- 다음: [문제 3 - SLICE/STRIDED_SLICE 5D 텐서 처리](/posts/2025/12/vectorblox-vnnx-tflite-slice-fix/)

**Language**: [English](/posts/2025/12/vectorblox-vnnx-tflite-reshape-fix-en/)

