---
title: '[TIR][Schedule] Add FuseReductionEpilogue primitive to fuse epilogue into reduction init - 5. Testing Strategy and Validation'
date: 2025-12-01
permalink: /posts/2025/12/tvm-fuse-reduction-epilogue-testing-en/
excerpt: 'We cover testing strategies to verify that the FuseReductionEpilogue primitive implemented up to Part 3 works correctly. We ensure the stability of the primitive through structural equality checks, numerical accuracy checks, edge case tests, and Trace Roundtrip checks.'
categories:
  - TVM
tags:
  - TVM
  - Compiler
  - Optimization
  - TIR
  - Schedule
  - C++
  - Testing
---

Up to [Part 3], we implemented a new scheduling primitive called FuseReductionEpilogue in C++. After that, we tested whether the scheduling primitive works well.

## 1. Testing Strategy

### A. Structural Equality Check

This is the most basic test. It checks "Does the transformed result code match 100% with the expected TIR code?" This verifies whether the compiler transformed the AST as intended.

```python
def test_fuse_reduction_epilogue_basic():  
    # 1. Create original schedule
    sch = tir.Schedule(matmul_bias_before, debug_mask="all")  
    
    # 2. Perform fusion
    sch.fuse_reduction_epilogue("multiply", "add")  
    
    # 3. Structural comparison with expected result (matmul_bias_expected)
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], matmul_bias_expected)  
    
    # 4. Trace Roundtrip validation (described later)
    verify_trace_roundtrip(sch=sch, mod=matmul_bias_before)
```

Additionally, to ensure it works not only with int8 quantized models but also with general float32 models, we added the `test_fuse_reduction_epilogue_fp32` test to ensure generality.

### B. Numerical Accuracy Check

Even if the structure changed nicely, it's useless if the calculation results are wrong. We verify using NumPy whether the actual operation results match within the error range.

```python
def test_fuse_reduction_epilogue_numerical_correctness():  
    # 1. Compile and prepare execution of original (Before)
    sch_original = tir.Schedule(matmul_bias_before, debug_mask="all")  
    mod_original = tvm.compile(sch_original.mod["main"], target="llvm")  

    # 2. Compile and prepare execution of fused (After) schedule
    sch_fused = tir.Schedule(matmul_bias_before, debug_mask="all")  
    sch_fused.fuse_reduction_epilogue("multiply", "add")  
    mod_fused = tvm.compile(sch_fused.mod["main"], target="llvm")  

    # 3. Generate random data (Input Generation)
    A_np = np.random.randint(-128, 127, size=(16, 16), dtype="int8")  
    B_np = np.random.randint(-128, 127, size=(16, 16), dtype="int8")  
    C_np = np.random.randint(-1000, 1000, size=(16, 16), dtype="int32")  
    
    # 4. Calculate NumPy ground truth (Oracle)
    expected = (A_np.astype("int32") @ B_np.T.astype("int32")) + C_np  

    # 5. Execute TVM and compare results
    # ... (TVM execution code omitted) ...

    # 6. Validation: original vs fused vs ground truth (NumPy)
    np.testing.assert_allclose(D_original, expected, rtol=1e-5)  
    np.testing.assert_allclose(D_fused, expected, rtol=1e-5)  
    np.testing.assert_allclose(D_fused, D_original, rtol=1e-5)
```

### C. Edge Case: Multiple Epilogue Test (Review Feedback)

To verify cases where MatMul results are used in multiple places (Multiple Consumers), we defined the `matmul_bias_multiple_epilogue_before` function. This is a scenario where temp, the result of the multiply block, is used in two places: the add block and the add2 block.

```python
@T.prim_func  
def matmul_bias_multiple_epilogue_before(...):  
    # ... (Perform MatMul) ...
    
    # Consumer 1: Bias Add 1
    for i, j in T.grid(16, 16):  
        with T.block("add"):  
            D[vi, vj] = temp[vi, vj] + C[vi, vj]  
            
    # Consumer 2: Bias Add 2
    for i, j in T.grid(16, 16):  
        with T.block("add2"):  
            E[vi, vj] = temp[vi, vj] + C[vi, vj]
```

Through this test (`test_fuse_reduction_epilogue_multiple_epilogue`), we proved that even if multiply and add are fused, the remaining add2 block does not break and works normally.

### D. Trace Roundtrip Check

TVM must be able to record (Trace) the scheduling commands applied by users and serialize them in JSON format or restore them again.

By calling `verify_trace_roundtrip(sch=sch, ...)` at the end of all test cases, we confirmed that the FuseReductionEpilogue primitive is perfectly integrated into TVM's scheduling history system.

## 2. Test Results

Through the four testing strategies above, the FuseReductionEpilogue primitive was verified as follows:

The transformed TIR code matches the expected structure, calculation results before and after fusion are identical, it works stably even in multiple Consumer scenarios, and it is perfectly integrated with TVM's scheduling history system.

This confirmed that the primitive is stable and reliable enough to be used in production environments.

---

**Series Posts**

- Previous: [Part 4. Architecture Visualization](/posts/2025/12/tvm-fuse-reduction-epilogue-architecture-en/)
- Next: [FuseReductionEpilogue: Clipping Pattern Support Implementation](/posts/2026/01/tvm-fuse-reduction-epilogue-clipping-en/)

**Language**: [한국어 (Korean)](/posts/2025/12/tvm-fuse-reduction-epilogue-testing/)

