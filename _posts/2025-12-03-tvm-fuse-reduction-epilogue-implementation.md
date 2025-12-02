---
title: '[TIR][Schedule] Add FuseReductionEpilogue primitive to fuse epilogue into reduction init - 3. C++ 구현'
date: 2025-12-01
permalink: /posts/2025/12/tvm-fuse-reduction-epilogue-implementation/
excerpt: 'Part 2에서 세운 계획을 바탕으로 TVM 컴파일러가 이해할 수 있는 C++ 코드로 구현합니다. 패턴 분석, AST 변환, 트리 재구성의 세 단계를 통해 Reduction Block과 Epilogue Block을 융합하는 프리미티브를 완성합니다.'
tags:
  - TVM
  - 컴파일러
  - 최적화
  - TIR
  - Schedule
  - C++
---

[Part 2]에서 세운 계획을 바탕으로, TVM 컴파일러가 이해할 수 있는 C++ 코드로 구현할 차례입니다.

TVM은 Python API(tir.Schedule)를 제공하지만, 그 뒷단의 무거운 연산과 트리 변환(Transformation) 로직은 대부분 C++로 작성되어 있습니다. 이번 구현은 `src/tir/schedule/primitive/compute_inline.cc` 파일에서 진행됩니다.

## 2. 구현 구조 (Architecture)

구현은 크게 세 단계로 나뉩니다.

1. Analysis: `ReductionEpilogueFuser` 클래스
   - 두 블록이 융합 가능한 조건인지 검사합니다.

2. Transformation: `CreateFusedReductionBlock` 함수
   - Reduction Block의 T.init을 수정하고, 중간 버퍼를 교체합니다.

3. Substitution: `SingleBlockFusionReplacer` 클래스
   - 기존의 두 블록을 도려내고, 새로 만든 융합 블록을 이식합니다.

## 3. Step 1: 패턴 분석기 (ReductionEpilogueFuser)

가장 먼저 해야 할 일은 "이 블록들을 합쳐도 안전한가?"를 판단하는 것입니다. 이를 위해 `ReductionEpilogueFuser` 클래스를 정의했습니다. `BodyPatternAllowFusion` 메서드가 전체적인 검증(predicate 확인, BufferStore 확인 등)을 총괄하며, 핵심 패턴 매칭은 `AnalyzeEpiloguePattern`에서 수행합니다.

### Epilogue 패턴 검증

Epilogue 블록의 연산이 D = temp + C (혹은 C + temp) 형태인지 확인해야 합니다. TVM의 AST에서 이를 확인하려면 `AddNode`를 들여다봐야 합니다.

```cpp
bool ReductionEpilogueFuser::AnalyzeEpiloguePattern(const PrimExpr& value) {
  // 수식이 덧셈(AddNode)인지 확인
  if (const auto* add = value.as<AddNode>()) {
    const auto* load_a = add->a.as<BufferLoadNode>();
    const auto* load_b = add->b.as<BufferLoadNode>();
    
    // 피연산자 중 하나가 Reduction Block의 결과(temp)인지 확인
    bool a_is_target = load_a && load_a->buffer.same_as(inlined_buffer_);
    bool b_is_target = load_b && load_b->buffer.same_as(inlined_buffer_);
    
    // XOR 조건: 둘 중 하나만 temp여야 함 (temp + temp는 불가)
    if (a_is_target != b_is_target) {
      // temp가 아닌 쪽이 바로 Bias(C)가 됨
      epilogue_addend_ = a_is_target ? add->b : add->a;
      return true;
    }
  }
  return false;
}
```

이 로직 덕분에 `temp + C`뿐만 아니라 `C + temp` 순서로 되어 있어도(교환법칙) 문제없이 Bias 항(`epilogue_addend_`)을 추출해낼 수 있습니다.

## 4. Step 2: 블록 재조립 (CreateFusedReductionBlock)

검증이 끝났다면, `CreateFusedReductionBlock` 함수가 실행됩니다. Reduction Block을 복제한 뒤 내부 코드를 교체하는 작업입니다.

### 핵심: Init 구문 개조

Part 2에서 설계한 대로, 0으로 초기화하던 구문을 Bias(C)를 로드하는 구문으로 바꿉니다.

```cpp
// 2. Change init to epilogue value: D[vi, vj] = C[vi, vj]
BufferStore new_init_store(
    epilogue_output_buffer_,                       // 최종 출력 버퍼 D
    Substitute(epilogue_addend_, var_map),         // 변수 매핑된 Bias 값 C
    Substitute(epilogue_output_indices_, var_map)  // 변수 매핑된 인덱스
);
new_block->init = new_init_store;
```

### 버퍼 교체 (Buffer Replacement)

T.init 뿐만 아니라, 블록 본문(Body)에서도 temp 버퍼를 사용하는 모든 코드를 D 버퍼를 사용하도록 바꿔야 합니다. 이를 위해 `StmtExprMutator`를 상속받은 `BufferReplacer` 클래스를 구현했습니다.

```cpp
class BufferReplacer : public StmtExprMutator {
 public:
  BufferReplacer(Buffer old_buf, Buffer new_buf) 
      : old_buffer_(old_buf), new_buffer_(new_buf) {}

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    // 기존 버퍼(temp)에 쓰는 경우 -> 새 버퍼(D)에 쓰도록 변경
    if (store->buffer.same_as(old_buffer_)) {
      return BufferStore(new_buffer_, store->value, store->indices);
    }
    return store;
  }
  // ... (BufferLoad도 동일하게 처리)
};
```

또한, 블록 상단에 명시된 Read/Write Region 정보도 함께 업데이트해주어야 합니다. 이를 놓치면 TVM의 IR 검증 단계(Validator)에서 에러가 발생합니다.

## 5. Step 3: 트리 이식 (SingleBlockFusionReplacer)

새로운 융합 블록(`new_fused_block_`)이 완성되었습니다. 이제 `SingleBlockFusionReplacer` 클래스를 통해 전체 트리(Scope)에서 옛날 블록들(multiply, add)을 제거하고 새 블록을 끼워 넣습니다.

```cpp
Stmt VisitStmt_(const BlockRealizeNode* realize) final {
  if (realize->block.same_as(old_reduction_block_)) {
    // 1. 기존 Reduction 블록 위치에 -> 새 융합 블록 이식
    ObjectPtr<BlockRealizeNode> new_realize = make_object<BlockRealizeNode>(*realize);
    new_realize->block = new_fused_block_;
    return BlockRealize(new_realize);
  } else if (realize->block.same_as(old_epilogue_block_)) {
    // 2. 기존 Epilogue 블록 위치 -> 삭제 (No-Op 반환)
    return Evaluate(0);
  }
  return StmtMutator::VisitStmt_(realize);
}
```

삭제된 자리에 남은 `Evaluate(0)`는 이후 `SeqStmt::Flatten` 과정을 통해 깔끔하게 정리됩니다. 마지막으로 더 이상 사용되지 않는 temp 버퍼의 `Allocate` 노드도 찾아서 제거해 주면 TIR 트리가 깨끗해집니다.

## 6. Python API 연결 (FFI Binding)

C++ 구현이 끝났지만, 사용자는 Python에서 이 기능을 쓰고 싶어 합니다. TVM의 FFI(Foreign Function Interface)를 통해 길을 뚫어줍니다.

### src/tir/schedule/schedule.cc (C++ 등록)

```cpp
.def_method("tir.schedule.ScheduleFuseReductionEpilogue",
            &ScheduleNode::FuseReductionEpilogue);
```

### python/tvm/tir/schedule/schedule.py (Python 래퍼)

사용자 편의를 위해 입력값을 정규화하는 래퍼 함수를 제공합니다.

```python
def fuse_reduction_epilogue(
    self,
    reduction_block: Union[BlockRV, str],
    epilogue_block: Union[BlockRV, str],
) -> None:
    """Fuse an epilogue block into a reduction block."""
    reduction_block = self._normalize_block_arg(reduction_block)
    epilogue_block = self._normalize_block_arg(epilogue_block)
    
    # C++ 백엔드 호출
    _ffi_api.ScheduleFuseReductionEpilogue(
        self, reduction_block, epilogue_block
    )
```

## 7. 결론

이로써 FuseReductionEpilogue 프리미티브의 구현이 모두 완료되었습니다.

패턴 분석: 합쳐도 되는지 확인하고 (`ReductionEpilogueFuser`), AST 조작: Init을 0에서 Bias로 바꾸고 버퍼를 교체한 뒤 (`CreateFusedReductionBlock`, `BufferReplacer`), 트리 재구성: 기존 블록을 새 블록으로 대체했습니다 (`SingleBlockFusionReplacer`).

이 세 단계를 통해 MatMul과 Bias Add를 하나의 Reduction Block으로 융합하여, 하드웨어의 MAC 명령어를 효율적으로 활용할 수 있게 되었습니다.

---

**시리즈 포스트**

- 이전: [Part 2. TIR 구조 변환 설계](/posts/2025/12/tvm-fuse-reduction-epilogue-design/)
- 다음: [Part 4. 아키텍처 시각화](/posts/2025/12/tvm-fuse-reduction-epilogue-architecture/)

