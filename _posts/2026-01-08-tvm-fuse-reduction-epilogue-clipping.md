---
title: '[TIR][Schedule] FuseReductionEpilogue: Clipping 패턴 지원 구현'
date: 2026-01-08
permalink: /posts/2026/01/tvm-fuse-reduction-epilogue-clipping/
excerpt: 'TVM의 TIR 스케줄 프리미티브인 fuse_reduction_epilogue의 지원 범위를 확장하여 Clipping(min(max(x, lower), upper)) 패턴을 자동으로 감지하고 최적화하는 기능을 추가했습니다. ReLU6나 Bounded ReLU와 같이 딥러닝 모델에서 빈번하게 사용되는 Clipping 연산을 리덕션 블록과 통합함으로써 메모리 대역폭 효율을 높였습니다.'
categories:
  - TVM
tags:
  - TVM
  - 컴파일러
  - 최적화
  - TIR
  - Schedule
  - Clipping
---

본 작업의 목적은 TVM의 TIR 스케줄 프리미티브인 `fuse_reduction_epilogue`의 지원 범위를 확장하여 Clipping(`min(max(x, lower), upper)`) 패턴을 자동으로 감지하고 최적화(Fusion)하는 기능을 추가하는 것입니다.

Clipping 연산은 ReLU6나 Bounded ReLU와 같이 딥러닝 모델에서 매우 빈번하게 사용되지만, 기존 프리미티브는 Bias(덧셈)와 BiasReLU 패턴만 지원했습니다. 이를 리덕션 블록과 통합함으로써 별도의 에피로그 블록 실행 오버헤드를 줄이고 메모리 대역폭 효율을 높이고자 합니다.

---

## 1. 개요

### 목표

- Clipping 패턴(`min(max(x, lower), upper)`) 자동 감지 및 퓨전
- 교환 법칙이 적용된 다양한 변형 패턴 지원
- 리덕션 루프 내부에 클리핑 연산 통합으로 성능 향상

### 배경

기존 `fuse_reduction_epilogue` 프리미티브는 다음 패턴만 지원했습니다:
- Bias: `temp + C`
- BiasReLU: `max(temp + C, 0)`

하지만 실제 딥러닝 모델에서는 다음과 같은 Clipping 연산이 자주 사용됩니다:
- ReLU6: `min(max(x, 0), 6)`
- Bounded ReLU: `min(max(x, lower), upper)`

이러한 패턴을 지원하지 않아 별도의 에피로그 블록이 생성되어 메모리 대역폭이 낭비되었습니다.

---

## 2. 구현

Clipping 패턴 지원을 위해 다음과 같은 단계로 구현을 진행했습니다.

### 에필로그 타입 확장 및 경계값 저장 구조 마련

열거형을 확장하고, 분석된 경계값을 보관합니다.

```cpp
// src/tir/schedule/primitive/compute_inline.cc

enum class EpilogueType {
  Bias,      // temp + C
  BiasReLU,  // max(temp + C, 0)
  Clipping,  // min(max(temp, lower), upper) // <- 신규 추가
};

// ReductionEpilogueFuser 클래스 멤버 변수 추가
PrimExpr clipping_lower_{nullptr}; // 하한값 저장
PrimExpr clipping_upper_{nullptr}; // 상한값 저장
```

### 유연한 패턴 분석 로직 구현

사용자가 `min(max(x, lo), hi)`뿐만 아니라 `max(hi, min(lo, x))` 등 다양한 순서로 코드를 작성해도 동일한 Clipping 패턴으로 인식하도록 구현했습니다.

```cpp
// 어느 쪽 인자에 리덕션 버퍼(temp)가 있는지 찾아내는 헬퍼
auto match_buffer_in_commutative_op = [this](const PrimExpr& a, const PrimExpr& b, PrimExpr* other) -> bool {
  if (const auto* load_a = a.as<BufferLoadNode>()) {
    if (load_a->buffer.same_as(inlined_buffer_)) { *other = b; return true; }
  }
  if (const auto* load_b = b.as<BufferLoadNode>()) {
    if (load_b->buffer.same_as(inlined_buffer_)) { *other = a; return true; }
  }
  return false;
};

// AnalyzeEpiloguePattern 내 Clipping 감지 로직
if (const auto* min_node = value.as<MinNode>()) {
  const MaxNode* max_node = nullptr;
  // min(max(temp, lower), upper) 또는 min(upper, max(temp, lower)) 감지
  if ((max_node = min_node->a.as<MaxNode>())) { upper = min_node->b; } 
  else if ((max_node = min_node->b.as<MaxNode>())) { upper = min_node->a; }
  
  if (max_node && match_buffer_in_commutative_op(max_node->a, max_node->b, &lower)) {
    clipping_lower_ = lower; clipping_upper_ = upper;
    epilogue_type_ = EpilogueType::Clipping;
    return true;
  }
}
```

### 리덕션 초기화(Init) 및 업데이트(Body) 본문 변환

리덕션의 시작값(0) 자체도 클리핑 범위 내에 들어오도록 초기화하고, 매 반복마다 연산을 수행하도록 BufferReplacer를 수정합니다.

```cpp
// 1. 초기화 단계 수정 (CreateFusedReductionBlock)
if (epilogue_type_ == EpilogueType::Clipping) {
  PrimExpr init_value = tir::make_zero(epilogue_output_buffer_->dtype);
  // 초기값 0에 대해 min(max(0, lower), upper) 적용
  PrimExpr clipped_init = Min(Max(init_value, Substitute(clipping_lower_, var_map)),
                              Substitute(clipping_upper_, var_map));
  new_init_store = BufferStore(epilogue_output_buffer_, clipped_init,
                               Substitute(epilogue_output_indices_, var_map));
}

// 2. 업데이트 단계 수정 (BufferReplacer::VisitStmt_)
if (store->buffer.same_as(old_buffer_)) {
  PrimExpr new_value = store->value;
  if (epilogue_type_ == EpilogueType::Clipping) {
    // 매 반복(per-iteration)마다 클리핑 적용하여 세만틱 유지
    new_value = Min(Max(new_value, clipping_lower_), clipping_upper_);
  }
  return BufferStore(new_buffer_, new_value, store->indices);
}
```

## 검증

총 8가지 케이스(기본 퓨전, 수치 정확도, 다중 에피로그, 5가지 교환 법칙 변형)를 통해 구현의 견고함을 확인했습니다.

```python
# tests/python/tir-schedule/test_tir_schedule_fuse_reduction_epilogue_clipping.py

@pytest.mark.parametrize("pattern_func", [
    lambda temp, lower, upper: T.min(T.max(temp, lower), upper),
    lambda temp, lower, upper: T.min(upper, T.max(temp, lower)),
    lambda temp, lower, upper: T.min(T.max(lower, temp), upper),
    lambda temp, lower, upper: T.max(T.min(temp, upper), lower),
    lambda temp, lower, upper: T.max(lower, T.min(temp, upper)),
])
def test_matmul_clipping_commutative_variants(pattern_func):
    # 모든 교환 법칙 조합이 Clipping 패턴으로 정상 인식 및 퓨전되는지 테스트
    ...
```

---

**시리즈 포스트**

- 이전: [FuseReductionEpilogue: 테스트 및 검증](/posts/2025/12/tvm-fuse-reduction-epilogue-testing/)

**Language**: [English](/posts/2026/01/tvm-fuse-reduction-epilogue-clipping-en/)

