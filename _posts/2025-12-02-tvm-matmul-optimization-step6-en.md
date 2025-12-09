---
title: 'TVM Matrix Multiplication Optimization - Step 6: Loop Unrolling'
date: 2025-12-02
permalink: /posts/2025/12/tvm-matmul-optimization-step6-en/
excerpt: 'We achieved 1050 GFLOPS through Loop Unrolling. We improved final performance by removing loop overhead and enhancing Instruction-Level Parallelism.'
categories:
  - TVM
tags:
  - TVM
  - GPU
  - Optimization
  - Matrix Multiplication
  - Loop Unrolling
  - ILP
---

# Step 6: Loop Unrolling

Achieved 1050 GFLOPS

## Results

| Unroll Factor | 512x512 | 1024x1024 | Improvement |
|---------------|---------|-----------|-----------|
| 4 | 1034 GFLOPS | 1038 GFLOPS | +0.9% |
| 8 | 940 GFLOPS | 1035 GFLOPS | +0.6% |
| **16** | 1028 GFLOPS | **1050 GFLOPS** | **+2.0%** |
| 32 | 1022 GFLOPS | 1041 GFLOPS | +1.2% |

Best performance: 1050 GFLOPS (factor=16)

## 1. Compiler Theory: Loop Unrolling

### Removing Loop Overhead

Loops have overhead every iteration.

Suppose we have the following loop:

```python
for (int k = 0; k < 32; k++) {
    C += A[k] * B[k];
}
```

- Every time the loop iterates, it executes the following instructions:
   - Counter increment: `k++`
   - Condition comparison: Is `k < 32`?
   - Branch (Jump): If condition is true, return to loop start, if false, exit loop.
- Since the loop iterates 32 times total, `32 * 3 = 96` additional instructions are added.
- Loop Unrolling is a technique that unfolds and removes this iteration framework.

In the above loop, if we unfold 16 instructions at once (i.e., `Unroll Factor` is 16), only 2 iterations and 6 instructions occur.

```cuda
// First block (16 operations)
C += A[0] * B[0];
C += A[1] * B[1];
// ... (14 more)
C += A[15] * B[15];

// Second block (16 operations)
C += A[16] * B[16];
// ... (14 more)
C += A[31] * B[31];
```

## 2. TVM TensorIR Implementation

```python
# Unroll inner loop
sch.unroll(k_inner)
```

- `sch.unroll` replicates the loop body to reduce loop overhead and expose Instruction-Level Parallelism (ILP).

## Execution

```bash
python test_individual/test_step6.py
```

Code can be found at [https://github.com/kimm240/matrix-multiplication-optimization-with-tvm](https://github.com/kimm240/matrix-multiplication-optimization-with-tvm).

---

**Series Posts**

- Previous: [Step 5: Software Pipelining](/posts/2025/12/tvm-matmul-optimization-step5-en/)
- Next: [Step 7: cuBLAS Comparison](/posts/2025/12/tvm-matmul-optimization-step7-en/)

**Language**: [한국어 (Korean)](/posts/2025/12/tvm-matmul-optimization-step6/)

