---
title: 'VectorBlox vnnx_tflite.py 수정: 문제 3 - SLICE/STRIDED_SLICE 5D 텐서 처리'
date: 2025-12-06
permalink: /posts/2025/12/vectorblox-vnnx-tflite-slice-fix/
excerpt: 'vnnx_tflite.py에서 5차원 텐서를 슬라이싱하는 SLICE/STRIDED_SLICE 연산 처리 시 발생하는 컴파일 실패 문제를 해결했습니다. 5차원을 4차원으로 안전하게 변환하는 방법을 설명합니다.'
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

## 문제 3: SLICE/STRIDED_SLICE 5D 텐서 처리

### 컴파일 실패 로그

```
struct.pack error: unpack requires a buffer of X bytes

File "vnnx_tflite.py", line XXXX

# struct 필드 개수 불일치
```

### 문제 원인

SPNv2 모델에서 5차원 텐서 `[N, H, W, A, K]`를 슬라이싱하는 연산이 있었습니다. 예를 들어 특정 앵커나 키포인트만 선택하는 경우입니다. TFLite는 5차원 SLICE/STRIDED_SLICE를 지원하므로 정상적으로 변환됩니다.

그러나 VectorBlox SDK의 SLICE subnode 구조체는 4차원만 지원합니다. 구조체에서 begin, end, stride가 각각 4개 요소 배열로 정의되어 있습니다:

```c
// VectorBlox 구조체
struct SliceOptions {
    int32_t begin[4];   
    int32_t end[4];     
    int32_t stride[4];  
};
```

5차원 텐서의 파라미터(5개 요소)를 이 구조체에 넣으려 하면, `struct.pack()`에서 "4개를 예상했는데 5개가 왔다"는 에러가 발생합니다.

### 해결 과정

5차원을 4차원으로 안전하게 변환하는 방법이 필요했습니다.

해결 전략: 마지막 두 축(A, K)을 하나의 축(A×K)으로 접어서 4차원으로 만듭니다.

`[N, H, W, A, K]` → `[N, H, W, A×K]`

5D의 (a, k) 인덱스를 4D의 `c = a×K + k`로 매핑해도 메모리 위치가 동일합니다:

```
# 5D: offset = ... + a*K + k
# 4D: offset = ... + c = ... + (a*K + k)
# → 같은 위치!

# 예시:
# 5D [N,H,W,A=9,K=4]에서 (a=2, k=3)
# → 4D [N,H,W,C=36]에서 c = 2*4 + 3 = 11
# 메모리 오프셋 = ... + 2*4 + 3 = ... + 11
# → 동일! ✓
```

따라서 축 접기는 메모리 레이아웃을 유지하며, 단지 표기만 4D로 단순화합니다.

### 수정 후 코드

```python
elif subcode == "SLICE":
    begin = get_numpy_data_from_index(subop['inputs'][1], tensors, buffers)
    size = get_numpy_data_from_index(subop['inputs'][2], tensors, buffers)
    ishape = i_tensor['shape']
    
    # 5D 감지
    if len(begin) > 4 or len(size) > 4 or len(ishape) > 4:
        # 5차원으로 패딩
        begin_pad = pad_list(list(begin), 5, 0)
        size_pad = pad_list(list(size), 5)
        ishape_pad = pad_list(list(ishape), 5)
        
        # 마지막 두 축 접기
        K = ishape_pad[4]
        begin4 = [begin_pad[0], begin_pad[1], begin_pad[2], 
                  begin_pad[3] * K + begin_pad[4]]  # (a, k) → a*K + k
        size4_last = size_pad[3] * K if size_pad[4] == -1 else \
                     size_pad[3] * K + size_pad[4]
        size4 = [size_pad[0], size_pad[1], size_pad[2], size4_last]
        
        # 4차원 파라미터 설정
        sn.SliceOptions.begin = begin4
        sn.SliceOptions.end = [b + s for b, s in zip(begin4, size4)]
        sn.SliceOptions.stride = [1, 1, 1, 1]
    else:
        # 4D 이하는 기존 로직
        [...]
```

인덱스 매핑 `(a, k) → a×K + k`로 5D 인덱스를 4D로 변환합니다.

### 해결 결과

수정 후 5차원 SLICE가 정상적으로 컴파일됩니다:

```
[SLICE 5D 텐서 감지]
5D → 4D 변환 (마지막 두 축 접기)
begin/end/stride 4개로 조정
컴파일 성공 ✓
```

### 결론

SLICE/STRIDED_SLICE 5D 텐서 처리 문제는 VectorBlox SDK의 하드웨어 제약사항(4차원 구조체)과 SPNv2의 5차원 텐서 사용 간의 불일치에서 발생했습니다.

해결책으로 마지막 두 축을 접어서 4차원으로 변환하는 방법을 사용했으며, 이는 메모리 레이아웃을 유지하면서 하드웨어 호환성을 확보하는 안전한 방법입니다.

---

**시리즈 포스트**

- 이전: [문제 2 - RESHAPE 연산](/posts/2025/12/vectorblox-vnnx-tflite-reshape-fix/)
- 다음: [문제 4 - INT8 상수 ADD/SUB 연산](/posts/2025/12/vectorblox-vnnx-tflite-add-sub-fix/)

**Language**: [English](/posts/2025/12/vectorblox-vnnx-tflite-slice-fix-en/)

