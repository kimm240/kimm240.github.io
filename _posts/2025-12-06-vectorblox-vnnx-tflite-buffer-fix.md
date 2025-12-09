---
title: 'VectorBlox vnnx_tflite.py 수정: 문제 5 - 상수 텐서 버퍼 구조체 호환성 문제 해결'
date: 2025-12-06
permalink: /posts/2025/12/vectorblox-vnnx-tflite-buffer-fix/
excerpt: 'vnnx_tflite.py에서 상수 텐서의 buffer 필드를 직렬화할 때 발생하는 struct.pack 에러를 해결했습니다. 모든 텐서의 buffer를 [buffer_id, offset] 배열 형식으로 통일하여 C 구조체와의 호환성을 확보하는 방법을 설명합니다.'
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

## 문제 5: 상수 텐서 버퍼 구조체 호환성

### 컴파일 실패 로그

```
struct.pack error: required argument is not an integer

File "vnnx_tflite.py", line XXXX

# 상수 텐서 직렬화 실패
```

### 문제 원인

TFLite 모델에는 가중치나 상수 값이 constant tensor로 저장됩니다. 이런 텐서들은 buffer 필드를 가지는데, VectorBlox의 Tensor 구조체는 이를 2개 정수 배열 `int32_t buffer[2]`로 정의합니다:

```c
// VectorBlox 구조체
typedef struct {
    int32_t buffer[2];  // [buffer_id, offset]
    // ...
} Tensor;
```

그런데 원래 `vnnx_tflite.py`에서는 상수 텐서의 buffer를 단일 정수로 설정하는 경우가 있었습니다:

```python
tn.buffer = buffer_id  # 단일 정수
```

이를 `struct.pack()`으로 직렬화할 때 "2개 정수를 기대했는데 1개만 왔다"는 에러가 발생했습니다.

### 해결 과정

모든 텐서의 buffer를 일관되게 `[buffer_id, offset]` 형식으로 설정해야 했습니다.

수정 전에는 경우에 따라 다른 형식을 사용했습니다:

```python
# 일반 텐서
tn.buffer = [buffer_id, 0]  # 배열

# 상수 텐서
tn.buffer = buffer_id  # 단일 정수
```

수정 후에는 모든 경우를 배열로 통일했습니다:

```python
# 모든 텐서
if isinstance(buffer_info, int):
    # 단일 정수면 배열로 변환
    tn.buffer = [buffer_info, 0]
elif isinstance(buffer_info, dict):
    # dict면 배열로 변환
    tn.buffer = [buffer_info.get('id', 0), buffer_info.get('offset', 0)]
else:
    # 이미 배열이면 그대로
    tn.buffer = buffer_info

# 항상 [buffer_id, offset] 형식 보장
```

Tensor 구조체의 정의에 맞게 Python 코드도 동일한 구조를 유지하도록 수정했습니다.

### 해결 결과

수정 후 상수 텐서가 포함된 모델도 정상 컴파일됩니다:

```
[상수 텐서 처리]
buffer를 [buffer_id, offset] 배열로 통일
struct.pack 성공
컴파일 성공
```

### 수정 사항 요약

| 항목 | 수정 전 | 수정 후 |
|------|---------|---------|
| 버퍼 형식 | 혼재 (단일 정수/배열) | 통일 (배열만) |
| 구조체 호환성 | struct.pack 에러 | 정상 직렬화 |
| 상수 텐서 처리 | 불일치 | 일관성 보장 |
| 컴파일 안정성 | 실패 | 성공 |

### 결론

상수 텐서 버퍼 구조체 호환성 문제는 VectorBlox C 구조체의 `int32_t buffer[2]` 정의와 Python 코드의 불일치에서 발생했습니다.

해결책으로 모든 텐서의 buffer를 `[buffer_id, offset]` 배열 형식으로 통일하여 C 구조체와의 호환성을 확보했습니다.

최종 결과: SPNv2의 상수 텐서들이 VectorBlox 하드웨어에서 정상적으로 직렬화되고 실행 가능하게 되었습니다.

### 핵심 개선사항

- 버퍼 형식의 일관성 보장
- C 구조체와의 완벽한 호환성
- struct.pack 직렬화 안정성
- 다양한 상수 텐서 타입 지원

---

**시리즈 포스트**

- 이전: [문제 4 - INT8 상수 ADD/SUB 연산](/posts/2025/12/vectorblox-vnnx-tflite-add-sub-fix/)

**Language**: [English](/posts/2025/12/vectorblox-vnnx-tflite-buffer-fix-en/)

