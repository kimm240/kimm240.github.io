---
title: 'VectorBlox vnnx_tflite.py 수정: 문제 4 - INT8 상수 ADD/SUB 연산 문제 해결'
date: 2025-12-06
permalink: /posts/2025/12/vectorblox-vnnx-tflite-add-sub-fix/
excerpt: 'vnnx_tflite.py에서 상수 연산을 포함한 ADD/SUB 연산 처리 시 발생하는 양자화 파라미터 미초기화 문제를 해결했습니다. multi_input=False인 경우에도 모든 양자화 파라미터를 초기화하도록 수정하는 방법을 설명합니다.'
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

## 문제 4: INT8 상수 ADD/SUB 연산

### 컴파일 실패 로그

```
Assertion failed: (activation_min != 0)

File "vbx_cnn_model.c", line XXXX
```

### 문제 원인

SPNv2 모델에서 ADD나 SUB 연산의 한쪽이 상수인 경우가 있습니다. 예를 들어 `x + 0.5` 같은 bias 추가나 `x - mean` 같은 정규화입니다.

TFLite에서 이런 연산은 한쪽 입력이 constant tensor로 표현됩니다. VectorBlox SDK는 이를 `multi_input=False`로 판단합니다.

반면 `x + y`같이, 모두 변수인 경우 VectorBlox SDK는 `multi_input=True`로 판단합니다.

문제는, VectorBlox SDK에서 `multi_input=False`인 경우, ADD/SUB의 양자화 파라미터(activation_min/max, multiplier, shift 등)를 초기화하지 않는다는 것입니다. 따라서 VNNX를 실행할 때 C 레이어에서 양자화 파라미터 중 하나인 `activation_min`이 0이 나와서, INT8 fully quantization 시 assertion을 발생시킵니다.

### 해결 과정

`multi_input=False`인 경우에도 모든 양자화 파라미터를 초기화하도록 수정했습니다.

수정 전에는 `multi_input=True`인 경우에만 파라미터를 설정했습니다:

```python
if subcode in ['ADD', 'SUB']:
    if multi_input:
        # 양자화 파라미터 설정
        sn.activation_min = -128
        sn.activation_max = 127
        # multiplier, shift 등 설정
    # else: 아무것도 안 함 → C 레이어에서 에러!
```

수정 후에는 `multi_input` 여부와 관계없이 항상 파라미터를 설정합니다:

```python
if subcode in ['ADD', 'SUB']:
    # 항상 양자화 파라미터 초기화
    sn.activation_min = -128
    sn.activation_max = 127
    
    # input/output scale, offset 설정
    input_scale = i_tensor['quantization']['scale']
    output_scale = o_tensor['quantization']['scale']
    input_offset = i_tensor['quantization']['zero_point'][0]
    output_offset = o_tensor['quantization']['zero_point'][0]
    
    # multiplier, shift 계산 및 설정
    input_multiplier, input_shift = get_quantized_multiplier(input_scale)
    output_multiplier, output_shift = get_quantized_multiplier(output_scale)
    sn.input_multiplier = len(weights)
    weights += struct.pack(...)
    # ... (모든 필드 초기화)
    
    # multi_input=False인 경우 추가 설정
    if not multi_input:
        # filter_data, bias_data 등도 설정
        sn.eltwise8.left_shift = left_shift
        sn.eltwise8.swap_inputs = 0
        # ...
```

모든 ELTWISE 연산이 C 레이어에서 요구하는 구조체 필드를 갖추도록 보장했습니다.

### 해결 결과

수정 후 상수 RHS를 가진 ADD/SUB도 정상 실행됩니다:

```
[ADD with constant RHS]
양자화 파라미터 모두 초기화
VNNX 실행 성공 ✓
C 레이어 assertion 통과 ✓
```

### 수정 사항 요약

| 항목 | 수정 전 | 수정 후 |
|------|---------|---------|
| 상수 연산 지원 | `multi_input=False`에서 파라미터 미초기화 | 모든 경우에 파라미터 초기화 |
| 양자화 파라미터 | 조건부 설정 | 항상 설정 |
| C 레이어 호환성 | assertion 실패 | 정상 실행 |
| 런타임 안정성 | 불안정 | 안정적 |

이 수정을 통해 상수 연산을 포함한 모든 ADD/SUB 연산이 안정적으로 실행될 수 있게 되었습니다.

---

**시리즈 포스트**

- 이전: [문제 3 - SLICE/STRIDED_SLICE 5D 텐서 처리](/posts/2025/12/vectorblox-vnnx-tflite-slice-fix/)
- 다음: [문제 5 - 상수 텐서 버퍼 구조체 호환성](/posts/2025/12/vectorblox-vnnx-tflite-buffer-fix/)

**Language**: [English](/posts/2025/12/vectorblox-vnnx-tflite-add-sub-fix-en/)

