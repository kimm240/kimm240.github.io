---
title: 'TVM Matrix Multiplication Optimization - Step 5: Software Pipelining'
date: 2025-12-02
permalink: /posts/2025/12/tvm-matmul-optimization-step5-en/
excerpt: 'We achieved 1029 GFLOPS through Software Pipelining. We achieved an average 58% performance improvement by hiding memory latency with computation. This post covers Software Pipelining techniques that execute multiple iterations overlapped.'
categories:
  - TVM
tags:
  - TVM
  - GPU
  - Optimization
  - Matrix Multiplication
  - Software Pipelining
---

# Step 5: Software Pipelining

Achieved 1029 GFLOPS

## Results

| Matrix Size | Step 5 | Improvement |
|----------|--------|--------|
| 256x256 | 855 GFLOPS | +68% |
| 512x512 | 1020 GFLOPS | +27% |
| 1024x1024 | 1029 GFLOPS | +74% |

Average: 968 GFLOPS

## 1. Compiler Theory: Software Pipelining

### Basic Idea - Multiple Iterations Simultaneously

**Problem**: Sequential execution is inefficient

```
Iteration 0: Load (400 cycles) → Compute (100 cycles) → Store
Iteration 1:                      Load (400) → Compute (100) → Store
Iteration 2:                                     Load (400) → Compute (100)

Total time = N × (400 + 100) = 500N cycles
```

- When executed sequentially, GPU cores are idle (IDLE) while waiting for memory to arrive.

**Solution**: Software Pipelining - Execute multiple iterations overlapped.

```
Iteration 0: Load ────────────────→
Iteration 1:         Load ──────→ Compute ──→
Iteration 2:                 Load ──→ Compute ──→ Store
Iteration 3:                         Load ──→ Compute ──→ Store

Total time ≈ max(Load, Compute) × N = 400N cycles
Savings: 100N cycles
```

- We overlap computation with memory transfer time.

## 2. TVM TensorIR Implementation

```python
# Software pipelining: overlap memory loads with computation
sch.annotate(k_outer, "software_pipeline_stage", [0, 0, 0, 1, 1])
sch.annotate(k_outer, "software_pipeline_order", [0, 1, 2, 3, 4])
```

- `software_pipeline_stage` specifies which stage each operation belongs to.
- `software_pipeline_order` specifies the execution order within each stage.

## Execution

```bash
python test_individual/test_step5.py
```

Code can be found at [https://github.com/kimm240/matrix-multiplication-optimization-with-tvm](https://github.com/kimm240/matrix-multiplication-optimization-with-tvm).

---

**Series Posts**

- Previous: [Step 4: Vectorization + Local Memory](/posts/2025/12/tvm-matmul-optimization-step4-en/)
- Next: [Step 6: Loop Unrolling](/posts/2025/12/tvm-matmul-optimization-step6-en/)

**Language**: [한국어 (Korean)](/posts/2025/12/tvm-matmul-optimization-step5/)

