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

### Pipeline Annotation

TVM implements pipelining through annotations.

- The pipeline overlaps execution of `Load`, `Compute`, and `Writeback`. Here, `Load` is stage `0`, `Compute` is stage `1`, and `Writeback` is stage `2`.

```python
# Assign stages to 4 blocks inside k_outer
sch.annotate(k_outer, "software_pipeline_stage", [0, 0, 1, 2])

# Execution order
sch.annotate(k_outer, "software_pipeline_order", [0, 1, 2, 3])
```

- `sch.annotate(k_outer, "software_pipeline_stage", [0, 1, 2, 3])`: The `k_outer` loop is the target of pipelining. The `"software_pipeline_stage"` directive specifies which stage each operation belongs to.
  - Stage 0: Load (A_shared, B_shared) - Preload next tile
  - Stage 1: Compute (C_local) - Compute with current tile
  - Stage 2: Writeback (C_local → C) - Write result

- `sch.annotate(k_outer, "software_pipeline_order", [0, 1, 2, 3])`: This command determines the execution order. `[0, 1, 2, 3]` means to execute the 4 blocks inside the `k_outer` loop in the original code order.

- Pipelining works as follows:

| Time | Stage 0 (L) | Stage 1 (C) | Stage 2 (WB) |
|------|-------------|-------------|--------------|
| **T1** | Load `k=0` tile |||
| **T2** | Load `k=1` tile | Compute `k=0` tile | |
| **T3** | Load `k=2` tile | Compute `k=1` tile | Writeback `k=0` result |
| **T4** | Load `k=3` tile | Compute `k=2` tile | Writeback `k=1` result |

## 3. Generated Execution Pattern

### Prologue-Kernel-Epilogue Structure

```python
# Prologue: Preload first tile
k=0: 
  Load A_tile[0], B_tile[0] to Shared Memory
  __syncthreads()

# Kernel: Execute multiple stages simultaneously
for k in 1..31:
  Stage 0: Load A_tile[k], B_tile[k]     # Next tile
  Stage 1: Compute using A_tile[k-1], B_tile[k-1]  # Current tile
  Stage 2: Writeback C_local[k-2] → C   # Previous result
  __syncthreads()

# Epilogue: Final processing
k=32:
  Compute using A_tile[31], B_tile[31]
  Writeback
```

## 4. Experiment: Double Buffering

```python
# Double Buffering
sch.annotate(A_shared, "double_buffer_scope", 0)
sch.annotate(B_shared, "double_buffer_scope", 0)
```

Results:

| Size | Basic Pipeline | + Double Buffer | Difference |
|------|---------------|-----------------|------------|
| 1024x1024 | 1029 GFLOPS | 1028 GFLOPS | -0.1% |

Conclusion: It is inefficient on A500's small shared memory. Since it requires 2x the space compared to the existing technique, Occupancy decreases.

## 5. Results

### Performance Analysis

Step 4 → Step 5:
- Average +58% improvement

Why is it so effective?

A500 bottlenecks:
- Small memory bandwidth (192 GB/s)
- Memory latency is the performance limiting factor

Software Pipelining effects:
- Completely hides memory latency with computation
- Fully utilizes bandwidth

## Execution

```bash
python test_individual/test_step5_pipelining.py
```

Code can be found at [https://github.com/kimm240/matrix-multiplication-optimization-with-tvm](https://github.com/kimm240/matrix-multiplication-optimization-with-tvm).

---

**Series Posts**

- Previous: [Step 4: Vectorization + Local Memory](/posts/2025/12/tvm-matmul-optimization-step4-en/)
- Next: [Step 6: Loop Unrolling](/posts/2025/12/tvm-matmul-optimization-step6-en/)

**Language**: [한국어 (Korean)](/posts/2025/12/tvm-matmul-optimization-step5/)

