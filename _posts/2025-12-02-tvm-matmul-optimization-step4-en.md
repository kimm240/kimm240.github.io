---
title: 'TVM Matrix Multiplication Optimization - Step 4: Vectorization + Local Memory'
date: 2025-12-02
permalink: /posts/2025/12/tvm-matmul-optimization-step4-en/
excerpt: 'We achieved an average of 614 GFLOPS through Vectorization and Local Memory (register) caching. This post covers register optimization through Scalar Replacement techniques and memory bandwidth utilization through vectorization.'
categories:
  - TVM
tags:
  - TVM
  - GPU
  - Optimization
  - Matrix Multiplication
  - Vectorization
  - Scalar Replacement
---

# Step 4: Vectorization + Local Memory

## Results

| Matrix Size | Step 4 | Improvement |
|----------|--------|--------|
| 512x512 | 804 GFLOPS | +94% |
| 1024x1024 | 593 GFLOPS | +29% |
| 2048x2048 | 549 GFLOPS | +23% |

Average: 614 GFLOPS

## 1. Compiler Theory: Scalar Replacement (Register Pipelining)

### Basic Idea

Replace repeated array accesses with scalar variables and assign them to registers.

In matrix multiplication, `C[i,j]` is accumulated K times (here, 1024 times):

```python
# Original: Read and write C[i,j] from memory every time
for k in range(1024):
    C[i,j] = C[i,j] + A[i,k] * B[k,j]
```

- This way, we make 1024 read requests and 1024 write requests to `C[i, j]`. That is, 2048 memory accesses.
- By assigning the value to be stored in `C[i, j]` to a scalar variable and storing the scalar variable in `local memory` (in GPU, `register`), we can create the following situation:

```python
# Replace C[i,j] with scalar variable temp
temp = C[i,j]  # 1 read
for k in range(1024):
    temp = temp + A[i,k] * B[k,j]
    # Accumulated in register
C[i,j] = temp  # 1 write
```

- Under the premise that `register` access is not counted as memory access, we can reduce memory access count from 2048 to 2.

## 2. TVM TensorIR Implementation

### C_local: GPU's Scalar Replacement

```python
# Cache C to local memory (register)
C_local = sch.cache_write(block_C, 0, "local")
```

- `sch.cache_write` creates a buffer in local memory (register) and writes the final result to global memory only once.

### Vectorization

```python
# Vectorize memory loads
sch.vectorize(i_elem)
```

- Vectorization allows loading multiple elements at once, improving memory bandwidth utilization.

## Execution

```bash
python test_individual/test_step4_final.py
```

Code can be found at [https://github.com/kimm240/matrix-multiplication-optimization-with-tvm](https://github.com/kimm240/matrix-multiplication-optimization-with-tvm).

---

**Series Posts**

- Previous: [Step 3: Shared Memory](/posts/2025/12/tvm-matmul-optimization-step3-en/)
- Next: [Step 5: Software Pipelining](/posts/2025/12/tvm-matmul-optimization-step5-en/)

**Language**: [한국어 (Korean)](/posts/2025/12/tvm-matmul-optimization-step4/)

