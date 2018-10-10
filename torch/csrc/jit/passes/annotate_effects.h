#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

TORCH_API void AnnotateEffects(std::shared_ptr<Graph>& graph);
TORCH_API bool CanRelocate(
    const Block* block,
    const Node* insertPoint,
    const std::unordered_set<const Node*>& group);

} // namespace jit
} // namespace torch
