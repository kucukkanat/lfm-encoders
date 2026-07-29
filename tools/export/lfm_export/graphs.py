"""Torch wrappers that give each exported model a flat, ONNX-friendly signature.

The upstream classes return dataclasses / dicts and take pooling matrices as
inputs. Neither survives an ONNX export cleanly, and the pooling matrices are
`(batch, rules, seq)` dense tensors that would have to be built and uploaded on
every call from JS. So each wrapper below returns *per-token* tensors and leaves
pooling to the caller.

That is not an approximation. Both task heads pool with a mean and then apply an
affine projection, and affine maps commute with means:

    W @ mean(h_i) + b == mean(W @ h_i + b)

so projecting every token first and mean-pooling the projections afterwards is
exactly the reference computation, just reassociated. It also means the graph is
independent of the number of rules/categories, which keeps one static ONNX file
usable for any zero-shot label set.
"""

from __future__ import annotations

import torch
from torch import nn


class MaskedLmGraph(nn.Module):
    """Base encoder: vocabulary logits plus the pooled-ready hidden states."""

    outputs = ("logits", "last_hidden_state")

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.model.lfm2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        return self.model.lm_head(hidden), hidden


class TwoTowerGraph(nn.Module):
    """Prompt router / policy linter: both projection towers, per token.

    `token_proj` is the "query" tower (the text being classified) and
    `rule_proj` is the "key" tower (the free-text categories or rules). The
    router and the linter share this graph exactly; they differ only in how the
    two towers are combined afterwards, which is a handful of scalars recorded
    in `head_config` rather than anything structural.
    """

    outputs = ("token_proj", "rule_proj")

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.model.lfm2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        return self.model.tok_proj(hidden), self.model.rule_proj(hidden)
