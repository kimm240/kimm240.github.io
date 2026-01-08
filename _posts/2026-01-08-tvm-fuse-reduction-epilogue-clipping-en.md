---
title: '[TIR][Schedule] FuseReductionEpilogue: Clipping Pattern Support Implementation'
date: 2026-01-08
permalink: /posts/2026/01/tvm-fuse-reduction-epilogue-clipping-en/
excerpt: 'We extended the support scope of TVM TIR schedule primitive fuse_reduction_epilogue to automatically detect and optimize Clipping (min(max(x, lower), upper)) patterns. By integrating Clipping operations, which are frequently used in deep learning models like ReLU6 and Bounded ReLU, with reduction blocks, we improved memory bandwidth efficiency'
categories:
  - TVM
tags:
  - TVM
  - Compiler
  - Optimization
  - TIR
  - Schedule
  - Clipping
---

The goal of this work is to extend the support scope of TVM's TIR schedule primitive `fuse_reduction_epilogue` to automatically detect and optimize (fuse) Clipping (`min(max(x, lower), upper)`) patterns.

Clipping operations are very frequently used in deep learning models like ReLU6 and Bounded ReLU, but the existing primitive only supported Bias (addition) and BiasReLU patterns. By integrating these with reduction blocks, we aim to reduce the overhead of separate epilogue block execution and improve memory bandwidth efficiency.

---

## 1. Overview

### Goals

- Automatic detection and fusion of Clipping patterns (`min(max(x, lower), upper)`)
- Support for various variant patterns with commutative law applied
- Performance improvement by integrating clipping operations inside reduction loops

### Background

The existing `fuse_reduction_epilogue` primitive only supported the following patterns:
- Bias: `temp + C`
- BiasReLU: `max(temp + C, 0)`

However, the following Clipping operations are commonly used in actual deep learning models:
- ReLU6: `min(max(x, 0), 6)`
- Bounded ReLU: `min(max(x, lower), upper)`

Since these patterns were not supported, separate epilogue blocks were created, wasting memory bandwidth.

---

## 2. Implementation

We implemented Clipping pattern support through the following steps.

### Epilogue Type Extension and Boundary Value Storage Structure

We extended the enumeration and stored the analyzed boundary values.

```cpp
// src/tir/schedule/primitive/compute_inline.cc

enum class EpilogueType {
  Bias,      // temp + C
  BiasReLU,  // max(temp + C, 0)
  Clipping,  // min(max(temp, lower), upper) // <- newly added
};

// ReductionEpilogueFuser class member variable addition
PrimExpr clipping_lower_{nullptr}; // store lower bound
PrimExpr clipping_upper_{nullptr}; // store upper bound
```

### Flexible Pattern Analysis Logic Implementation

We implemented it so that users can write code in various orders such as `min(max(x, lo), hi)` as well as `max(hi, min(lo, x))`, and it will be recognized as the same Clipping pattern.

```cpp
// Helper to find which argument contains the reduction buffer (temp)
auto match_buffer_in_commutative_op = [this](const PrimExpr& a, const PrimExpr& b, PrimExpr* other) -> bool {
  if (const auto* load_a = a.as<BufferLoadNode>()) {
    if (load_a->buffer.same_as(inlined_buffer_)) { *other = b; return true; }
  }
  if (const auto* load_b = b.as<BufferLoadNode>()) {
    if (load_b->buffer.same_as(inlined_buffer_)) { *other = a; return true; }
  }
  return false;
};

// Clipping detection logic in AnalyzeEpiloguePattern
if (const auto* min_node = value.as<MinNode>()) {
  const MaxNode* max_node = nullptr;
  // Detect min(max(temp, lower), upper) or min(upper, max(temp, lower))
  if ((max_node = min_node->a.as<MaxNode>())) { upper = min_node->b; } 
  else if ((max_node = min_node->b.as<MaxNode>())) { upper = min_node->a; }
  
  if (max_node && match_buffer_in_commutative_op(max_node->a, max_node->b, &lower)) {
    clipping_lower_ = lower; clipping_upper_ = upper;
    epilogue_type_ = EpilogueType::Clipping;
    return true;
  }
}
```

### Reduction Initialization (Init) and Update (Body) Transformation

We modified BufferReplacer so that the reduction's starting value (0) is initialized to be within the clipping range, and operations are performed on each iteration.

```cpp
// 1. Initialization step modification (CreateFusedReductionBlock)
if (epilogue_type_ == EpilogueType::Clipping) {
  PrimExpr init_value = tir::make_zero(epilogue_output_buffer_->dtype);
  // Apply min(max(0, lower), upper) to initial value 0
  PrimExpr clipped_init = Min(Max(init_value, Substitute(clipping_lower_, var_map)),
                              Substitute(clipping_upper_, var_map));
  new_init_store = BufferStore(epilogue_output_buffer_, clipped_init,
                               Substitute(epilogue_output_indices_, var_map));
}

// 2. Update step modification (BufferReplacer::VisitStmt_)
if (store->buffer.same_as(old_buffer_)) {
  PrimExpr new_value = store->value;
  if (epilogue_type_ == EpilogueType::Clipping) {
    // Apply clipping per-iteration to maintain semantics
    new_value = Min(Max(new_value, clipping_lower_), clipping_upper_);
  }
  return BufferStore(new_buffer_, new_value, store->indices);
}
```

## Validation

We verified the robustness of the implementation through a total of 8 cases (basic fusion, numerical accuracy, multiple epilogues, 5 commutative law variants).

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
    # Test that all commutative law combinations are correctly recognized and fused as Clipping patterns
    ...
```

---

**Series Posts**

- Previous: [FuseReductionEpilogue: Testing and Validation](/posts/2025/12/tvm-fuse-reduction-epilogue-testing-en/)

**Language**: [한국어 (Korean)](/posts/2026/01/tvm-fuse-reduction-epilogue-clipping/)

