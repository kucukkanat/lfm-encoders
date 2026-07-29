"""What gets exported, and what the browser needs to know about each head."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .graphs import MaskedLmGraph, TwoTowerGraph

# transformers.js resolves an architecture from `model_type`. Upstream `lfm2` is
# the *causal* decoder there, which expects KV-cache inputs this graph does not
# have, so a load would fail on missing inputs. Declaring our own type makes
# `AutoModel` fall through to its generic session runner, which is exactly the
# "feed the named inputs, return the named outputs" behaviour we want.
JS_MODEL_TYPE = "lfm2-bidirectional"

# Hub account the exports are published under. Output directories mirror
# `<owner>/<name>` so the same `modelRoot` works for a local export and for a
# plain Hub id, and the library's default model constants resolve either way.
HUB_OWNER = "kucukkanat"


@dataclass(frozen=True)
class ModelSpec:
    """One exportable model."""

    name: str
    repo: str
    auto_class: str
    graph: type[torch.nn.Module]
    task: str
    head_config: Callable[[Any], dict[str, Any]] = field(default=lambda _model: {})
    # Extra config blocks a consumer needs that are not derivable from the graph
    # (the diffusion decode schedule, for instance). Takes (model, tokenizer).
    extra_config: Callable[[Any, Any], dict[str, Any]] = field(default=lambda _m, _t: {})
    doc: str = ""

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.graph.outputs

    @property
    def hub_id(self) -> str:
        return f"{HUB_OWNER}/{self.name}"


def _router_head(model: Any) -> dict[str, Any]:
    """Scalars the JS side needs to reproduce `Lfm2BidirForSequenceRouting`.

    `logit_scale` is stored pre-exponential and the reference clamps the
    exponential at 30, so the clamp is applied here once rather than trusted to
    every consumer.
    """
    scale = float(torch.clamp(model.logit_scale.detach().exp(), max=30.0))
    return {
        "kind": "cosine",
        "normalize": True,
        "scale": scale,
        "bias": float(model.score_bias.detach()),
        "activation": "softmax",
        "prefix_heading": "Categories",
    }


def _linter_head(model: Any) -> dict[str, Any]:
    """Scalars for `Lfm2BidirForRuleMatching`.

    Its scale is the fixed 1/sqrt(d) of a dot-product attention score rather
    than a learned temperature, and it never L2-normalises the towers.
    """
    dim = model.tok_proj.out_features
    return {
        "kind": "dot",
        "normalize": False,
        "scale": 1.0 / (dim**0.5),
        "bias": float(model.score_bias.detach()),
        "activation": "sigmoid",
        "prefix_heading": "Policy",
    }


def _diffusion_config(_model: Any, tokenizer: Any) -> dict[str, Any]:
    """The MDLM decode schedule, recorded so consumers don't hard-code it.

    Nothing here is structural — the graph is the same bidirectional masked-LM
    forward as the base encoder. What makes this checkpoint a *chatbot* is the
    loop wrapped around it: a canvas of `<|mask|>` tokens is filled in over
    several confidence-ordered passes instead of one. The prompt layout is not
    cosmetic either; the diffusion-SFT run used exactly these markers, so the
    template is shipped with the weights rather than left to the caller.
    """
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("diffusion checkpoint's tokenizer has no mask token")
    return {
        "diffusion": {
            "mask_token_id": int(mask_id),
            "mask_token": tokenizer.mask_token,
            # Defaults mirroring LiquidAI/masked-diffusion's app.py.
            "max_new_tokens": 64,
            "steps": 32,
            "block_size": 16,
            "temperature": 0.0,
            # Confidence above which a token is committed early, letting a block
            # finish in fewer steps than its budget.
            "tau": 0.9,
            "template": {
                "system": "[SYS]\n{system}\n[/SYS]\n\n",
                "user": "[Question]\n{user}\n[/Question]\n\n[Answer]\n",
                "stop": ["[/Answer]", "[Question]", "[SYS]"],
            },
        }
    }


SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="LFM2.5-Encoder-350M-ONNX",
        repo="LiquidAI/LFM2.5-Encoder-350M",
        auto_class="AutoModelForMaskedLM",
        graph=MaskedLmGraph,
        task="fill-mask",
        doc="Bidirectional LFM2.5 encoder with its tied masked-LM head.",
    ),
    ModelSpec(
        name="LFM2.5-Encoder-350M-Prompt-Router-ONNX",
        repo="LiquidAI/LFM2.5-Encoder-350M-Prompt-Router",
        auto_class="AutoModel",
        graph=TwoTowerGraph,
        task="zero-shot-routing",
        head_config=_router_head,
        doc="Zero-shot prompt router: scores one text against free-text categories.",
    ),
    ModelSpec(
        name="LFM2.5-Encoder-350M-Policy-Linter-ONNX",
        repo="LiquidAI/LFM2.5-Encoder-350M-Policy-Linter",
        auto_class="AutoModel",
        graph=TwoTowerGraph,
        task="zero-shot-token-matching",
        head_config=_linter_head,
        doc="Zero-shot policy linter: scores every token against every free-text rule.",
    ),
    ModelSpec(
        name="LFM2.5-Encoder-350M-Diffusion-ONNX",
        repo="LiquidAI/LFM2.5-Encoder-350M-Diffusion",
        auto_class="AutoModelForMaskedLM",
        graph=MaskedLmGraph,
        task="masked-diffusion",
        extra_config=_diffusion_config,
        doc="Diffusion-SFT encoder: generates text by iteratively unmasking a canvas.",
    ),
)

BY_NAME = {s.repo.rsplit("/", 1)[-1]: s for s in SPECS}
