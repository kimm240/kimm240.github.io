---
title: '[TIR][Schedule] Add FuseReductionEpilogue primitive to fuse epilogue into reduction init - 3. C++ Implementation'
date: 2025-12-01
permalink: /posts/2025/12/tvm-fuse-reduction-epilogue-implementation-en/
excerpt: 'Based on the plan established in Part 2, we implement it in C++ code that the TVM compiler can understand. We complete a primitive that fuses Reduction Block and Epilogue Block through three stages: pattern analysis, AST transformation, and tree reconstruction.'
categories:
  - TVM
tags:
  - TVM
  - Compiler
  - Optimization
  - TIR
  - Schedule
  - C++
---

Based on the plan established in [Part 2], it's time to implement it in C++ code that the TVM compiler can understand.

TVM provides a Python API (tir.Schedule), but most of the heavy computation and tree transformation logic behind it is written in C++. This implementation is done in the `src/tir/schedule/primitive/compute_inline.cc` file.

## 2. Implementation Structure (Architecture)

The implementation is divided into three main stages.

1. Analysis: `ReductionEpilogueFuser` class
   - Checks whether the two blocks meet the conditions for fusion.

2. Transformation: `CreateFusedReductionBlock` function
   - Modifies the T.init of the Reduction Block and replaces the intermediate buffer.

3. Substitution: `SingleBlockFusionReplacer` class
   - Removes the existing two blocks and grafts the newly created fused block.

## 3. Step 1: Pattern Analyzer (ReductionEpilogueFuser)

The first thing to do is determine "Is it safe to merge these blocks?" For this, we defined the `ReductionEpilogueFuser` class. The `BodyPatternAllowFusion` method oversees overall validation (predicate checking, BufferStore checking, etc.), and core pattern matching is performed in `AnalyzeEpiloguePattern`.

### Epilogue Pattern Validation

We need to verify that the Epilogue block's operation is of the form D = temp + C (or C + temp). To check this in TVM's AST, we need to look at `AddNode`.

```cpp
bool ReductionEpilogueFuser::AnalyzeEpiloguePattern(const PrimExpr& value) {
  // Check if expression is addition (AddNode)
  if (const auto* add = value.as<AddNode>()) {
    const auto* load_a = add->a.as<BufferLoadNode>();
    const auto* load_b = add->b.as<BufferLoadNode>();
    
    // Check if one operand is the result of Reduction Block (temp)
    bool a_is_target = load_a && load_a->buffer.same_as(inlined_buffer_);
    bool b_is_target = load_b && load_b->buffer.same_as(inlined_buffer_);
    
    // XOR condition: exactly one must be temp (temp + temp is not allowed)
    if (a_is_target != b_is_target) {
      // The side that is not temp becomes Bias(C)
      epilogue_addend_ = a_is_target ? add->b : add->a;
      return true;
    }
  }
  return false;
}
```

Thanks to this logic, we can extract the Bias term (`epilogue_addend_`) without problems even if it's in the order `C + temp` (commutative property), not just `temp + C`.

## 4. Step 2: Block Reassembly (CreateFusedReductionBlock)

Once validation is complete, the `CreateFusedReductionBlock` function executes. This is the work of cloning the Reduction Block and then replacing the internal code.

### Core: Init Statement Modification

As designed in Part 2, we change the statement that initialized to 0 to a statement that loads Bias(C).

```cpp
// 2. Change init to epilogue value: D[vi, vj] = C[vi, vj]
BufferStore new_init_store(
    epilogue_output_buffer_,                       // Final output buffer D
    Substitute(epilogue_addend_, var_map),         // Variable-mapped Bias value C
    Substitute(epilogue_output_indices_, var_map)  // Variable-mapped indices
);
new_block->init = new_init_store;
```

### Buffer Replacement

Not only T.init, but also all code using the temp buffer in the block body must be changed to use the D buffer. For this, we implemented the `BufferReplacer` class that inherits from `StmtExprMutator`.

```cpp
class BufferReplacer : public StmtExprMutator {
 public:
  BufferReplacer(Buffer old_buf, Buffer new_buf) 
      : old_buffer_(old_buf), new_buffer_(new_buf) {}

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    // When writing to old buffer (temp) -> change to write to new buffer (D)
    if (store->buffer.same_as(old_buffer_)) {
      return BufferStore(new_buffer_, store->value, store->indices);
    }
    return store;
  }
  // ... (BufferLoad is handled the same way)
};
```

Additionally, the Read/Write Region information specified at the top of the block must also be updated. Missing this will cause errors in TVM's IR validation stage (Validator).

## 5. Step 3: Tree Grafting (SingleBlockFusionReplacer)

The new fused block (`new_fused_block_`) is complete. Now, through the `SingleBlockFusionReplacer` class, we remove the old blocks (multiply, add) from the entire tree (Scope) and insert the new block.

```cpp
Stmt VisitStmt_(const BlockRealizeNode* realize) final {
  if (realize->block.same_as(old_reduction_block_)) {
    // 1. Graft new fused block at existing Reduction block location
    ObjectPtr<BlockRealizeNode> new_realize = make_object<BlockRealizeNode>(*realize);
    new_realize->block = new_fused_block_;
    return BlockRealize(new_realize);
  } else if (realize->block.same_as(old_epilogue_block_)) {
    // 2. Existing Epilogue block location -> delete (return No-Op)
    return Evaluate(0);
  }
  return StmtMutator::VisitStmt_(realize);
}
```

The `Evaluate(0)` left in the deleted location is cleanly organized through the subsequent `SeqStmt::Flatten` process. Finally, if we find and remove the `Allocate` node of the temp buffer that is no longer used, the TIR tree becomes clean.

## 6. Python API Connection (FFI Binding)

The C++ implementation is complete, but users want to use this feature from Python. We connect through TVM's FFI (Foreign Function Interface).

### src/tir/schedule/schedule.cc (C++ Registration)

```cpp
.def_method("tir.schedule.ScheduleFuseReductionEpilogue",
            &ScheduleNode::FuseReductionEpilogue);
```

### python/tvm/tir/schedule/schedule.py (Python Wrapper)

A wrapper function is provided to normalize input values for user convenience.

```python
def fuse_reduction_epilogue(
    self,
    reduction_block: Union[BlockRV, str],
    epilogue_block: Union[BlockRV, str],
) -> None:
    """Fuse an epilogue block into a reduction block."""
    reduction_block = self._normalize_block_arg(reduction_block)
    epilogue_block = self._normalize_block_arg(epilogue_block)
    
    # Call C++ backend
    _ffi_api.ScheduleFuseReductionEpilogue(
        self, reduction_block, epilogue_block
    )
```

## 7. Conclusion

This completes the implementation of the FuseReductionEpilogue primitive.

Pattern analysis: Check if they can be merged (`ReductionEpilogueFuser`), AST manipulation: Change Init from 0 to Bias and replace buffers (`CreateFusedReductionBlock`, `BufferReplacer`), Tree reconstruction: Replaced existing blocks with new blocks (`SingleBlockFusionReplacer`).

Through these three stages, we were able to fuse MatMul and Bias Add into a single Reduction Block, enabling efficient use of the hardware's MAC instruction.

---

**Series Posts**

- Previous: [Part 2. TIR Structure Transformation Design](/posts/2025/12/tvm-fuse-reduction-epilogue-design-en/)
- Next: [Part 4. Architecture Visualization](/posts/2025/12/tvm-fuse-reduction-epilogue-architecture-en/)

**Language**: [한국어 (Korean)](/posts/2025/12/tvm-fuse-reduction-epilogue-implementation/)

