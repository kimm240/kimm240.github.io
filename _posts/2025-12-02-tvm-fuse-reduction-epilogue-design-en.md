---
title: '[TIR][Schedule] Add FuseReductionEpilogue primitive to fuse epilogue into reduction init - 2. TIR Structure Transformation Design'
date: 2025-12-01
permalink: /posts/2025/12/tvm-fuse-reduction-epilogue-design-en/
excerpt: 'Beyond the limitations of existing scheduling primitives confirmed in Part 1, we propose a new approach that sets the initial value of Reduction Block to Bias instead of 0. This post covers TIR structure transformation design and implementation requirements.'
categories:
  - TVM
tags:
  - TVM
  - Compiler
  - Optimization
  - TIR
  - Schedule
---

In [Part 1], we confirmed that existing scheduling primitives (compute_inline, etc.) cannot solve this problem due to the special nature of Reduction Blocks. Simply trying to physically connect the two blocks caused conflicts.

Therefore, we need a new approach beyond simply merging blocks. This post covers the logical design process.

## 1. Overview

## 2. Idea

The flow of the existing approach that processes matrix multiplication (MatMul) and bias addition (Bias Add) separately is as follows:

1. Prepare a temporary buffer (temp).
2. Clear the temporary buffer (temp = 0).
3. Fill the temporary buffer (temp += A * B).
4. Move to D and add C (D = temp + C).

Steps 1 and 4 in this process are the cause of inefficiency. To solve this, we set the initial value (Accumulator Initializer) of Reduction to Bias instead of 0.

### Formula Changes

Before (using temporary buffer):

1. Initialization:
   $$temp_{i,j} = 0$$

2. Accumulation:
   $$temp_{i,j} = temp_{i,j} + \sum_{k} A_{i,k} \times B_{j,k}$$

3. Final result:
   $$D_{i,j} = temp_{i,j} + C_{i,j}$$

After (direct accumulation):

1. Initialize with Bias:
   $$D_{i,j} = C_{i,j}$$

2. Direct accumulation:
   $$D_{i,j} = D_{i,j} + \sum_{k} A_{i,k} \times B_{j,k}$$

## 3. TIR Structure Transformation Design

Let's specifically design how to implement this idea at the TIR (Tensor IR) level, TVM's intermediate representation. The new primitive FuseReductionEpilogue we will implement must perform the following 3-step transformation.

### [Step 1] Target Identification (Pattern Matching)

Not all Reduction blocks can be transformed. It should only work when it's exactly a pattern that adds something to the MatMul result (Add).

- Producer (Reduction): Block that accumulates values in temp.
- Consumer (Epilogue): Simple addition block of the form D = temp + C.

### [Step 2] Init Statement Modification

This is the most important step. Find and modify the T.init() section inside the Reduction Block.

Before (AS-IS):

```python
with T.init():
    temp[vi, vj] = 0  # Initialize to 0
```

After (TO-BE):

```python
with T.init():
    D[vi, vj] = C[vi, vj]  # Initialize with Bias(C) value
```

The important point here is that not only should 0 be changed to C, but the write target buffer must change from temp to the final output buffer D.

### [Step 3] Buffer Replacement and Epilogue Removal

The temp buffer must also be replaced with the D buffer in the Reduction Block.

```
temp[vi, vj] = temp[vi, vj] + ...
→ D[vi, vj] = D[vi, vj] + ...
```

Once the work is complete, the temp buffer allocation that is no longer needed and the Epilogue (add) block that has nothing to do are completely removed from the tree.

## 4. Requirements Analysis for Implementation

We've organized the functions needed to move the above design into actual compiler code. This list will serve as a checklist when implementing in C++ in Part 3.

### Epilogue Pattern Analyzer:

- Must parse whether the Epilogue block's expression is really of the form Output = Input + Addend.
- Must work regardless of whether the addition order is Input + Addend or Addend + Input (commutative property).

### Reduction Block Validator:

- Must verify that the Producer is a complete Reduction Block with T.init.

### Buffer Replacer:

- A module is needed that traverses the AST (Abstract Syntax Tree) and replaces loads/stores for a specific buffer (temp) with another buffer (D). TVM's StmtExprMutator can be utilized.

### Index Remapping:

- The loop indices (i, j) of the Epilogue block and the loop indices (i, j, k) of the Reduction block may be mapped differently. Variable mapping logic is needed to correctly connect them.

## 5. Expected Results

If implemented according to this design, we will get optimized TIR code as follows:

```python
# temp buffer removed
for i, j, k in T.grid(16, 16, 16):
    with T.block("matmul_fused"):
        # Explicit read/write dependencies
        T.reads(C[vi, vj], A[vi, vk], B[vj, vk])
        T.writes(D[vi, vj])
        
        with T.init():
            # Initialize with Bias instead of 0 to absorb addition operation
            D[vi, vj] = C[vi, vj]
            
        # Accumulate directly to final buffer
        D[vi, vj] = D[vi, vj] + A[vi, vk] * B[vj, vk]
```

This is the background for why I designed fuse_reduction_epilogue. Instead of blaming existing tools, I decided to understand the special nature of Reduction Blocks and create a new tool that can handle them. The next post will cover specific guidelines on how we implemented the idea of breaking the convention of initializing to 0 and "initializing with Bias".

---

**Series Posts**

- Previous: [Part 1. Problem Analysis and Limitations of Existing Solutions](/posts/2025/12/tvm-fuse-reduction-epilogue-overview-en/)
- Next: [Part 3. C++ Implementation](/posts/2025/12/tvm-fuse-reduction-epilogue-implementation-en/)

**Language**: [한국어 (Korean)](/posts/2025/12/tvm-fuse-reduction-epilogue-design/)

