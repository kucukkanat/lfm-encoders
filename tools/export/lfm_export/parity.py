"""Measure what quantization cost us, and freeze golden fixtures for the TS tests.

Every number the browser shows is produced by two things: the ONNX graph, and a
pooling/scoring step reimplemented in TypeScript. This module pins both.

  * It runs the *unmodified* upstream PyTorch model (remote code, fp32) to get a
    reference answer, then runs each exported ONNX variant through an identical
    NumPy reimplementation of the pooling and compares.
  * It writes `fixtures.json` next to the model repos: token ids, character
    offsets from the HF fast tokenizer, and reference scores. The TypeScript
    integration tests replay those, which is how we catch a JS tokenizer or
    offset-reconstruction drift that no Python-side check could see.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from .spec import BY_NAME, ModelSpec

ROUTING_CASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "My invoice from last month charged me twice, can I get a refund?",
        ("billing and payments", "technical support", "account management", "sales enquiry"),
    ),
    (
        "The deployment keeps crashing with an out-of-memory error on startup.",
        ("billing and payments", "technical support", "account management", "sales enquiry"),
    ),
    (
        "Ich möchte mein Abonnement kündigen und eine Rückerstattung erhalten.",
        ("billing and payments", "technical support", "account management", "sales enquiry"),
    ),
    (
        "Write a haiku about the ocean at dawn.",
        ("creative writing", "code generation", "factual question", "small talk"),
    ),
)

_POLICY = (
    "no guarantees about financial returns",
    "no medical or health claims",
    "no disparaging competitors by name",
)

LINTING_CASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("We guarantee a 300% return within six months, risk free.", _POLICY),
    ("Unlike Acme Corp, our product actually cures chronic back pain.", _POLICY),
    ("Our quarterly report will be published on the investor relations page.", _POLICY),
    ("This supplement is clinically proven to reverse type 2 diabetes.", _POLICY),
    ("Competitor X ships a slower, buggier product at twice the price.", _POLICY),
    ("Nous garantissons un rendement de 20 % par an, sans aucun risque.", _POLICY),
)

MLM_CASES: tuple[str, ...] = (
    "The capital of France is <|mask|>.",
    "Paris is the capital of <|mask|>.",
    "El <|mask|> es un animal doméstico.",
)


# --------------------------------------------------------------------------- pooling


def build_prefix(heading: str, labels: tuple[str, ...]) -> str:
    body = "\n".join(f"- {label}" for label in labels) if labels else "- (none)"
    return f"{heading}:\n{body}\n\nText:\n"


def label_ranges(heading: str, labels: tuple[str, ...]) -> list[tuple[int, int]]:
    """Character span of each label inside the prefix, matching the reference."""
    ranges, pos = [], len(f"{heading}:\n")
    for label in labels:
        start = pos + 2  # skip the "- " bullet
        ranges.append((start, start + len(label)))
        pos = start + len(label) + 1  # + the newline
    return ranges


def pool_matrix(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> np.ndarray:
    """Row-normalised mean-pooling weights, one row per span.

    A token joins a span when it overlaps it; zero-width tokens (the specials)
    never do. This is the `text_pool` / `category_pool` construction from
    `Lfm2BidirForSequenceRouting.route`, lifted out so both the reference and
    the ONNX path provably share it.
    """
    pool = np.zeros((len(spans), len(offsets)), dtype=np.float32)
    for row, (start, end) in enumerate(spans):
        idxs = [
            i
            for i, (a, b) in enumerate(offsets)
            if a < end and b > start and a != b
        ]
        if idxs:
            pool[row, idxs] = 1.0 / len(idxs)
    return pool


def text_span(prefix: str, full: str) -> tuple[int, int]:
    return (len(prefix), len(full))


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def score_from_projections(
    token_proj: np.ndarray,  # (T, d)
    rule_proj: np.ndarray,  # (T, d)
    text_pool: np.ndarray,  # (1, T)
    rule_pool: np.ndarray,  # (R, T)
    head: dict[str, Any],
) -> np.ndarray:
    """Reproduce a head from per-token projections.

    Pooling after projecting is exact because both towers are affine: for an
    affine f, mean(f(h_i)) == f(mean(h_i)). See tools/export/README.md.
    """
    rules = rule_pool @ rule_proj  # (R, d)
    if head["kind"] == "cosine":
        query = _normalize(text_pool @ token_proj)[0]  # (d,)
        return (_normalize(rules) @ query) * head["scale"] + head["bias"]
    tokens = token_proj  # (T, d), scored per token
    return (tokens @ rules.T) * head["scale"] + head["bias"]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- runners


@dataclass
class Encoded:
    ids: list[int]
    offsets: list[tuple[int, int]]


def encode(tokenizer: Any, text: str) -> Encoded:
    enc = tokenizer(text, return_offsets_mapping=True)
    return Encoded(list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]])


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _run(session: ort.InferenceSession, ids: list[int]) -> list[np.ndarray]:
    array = np.asarray([ids], dtype=np.int64)
    return session.run(None, {"input_ids": array, "attention_mask": np.ones_like(array)})


def reference_two_tower(model: Any, ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        hidden = model.lfm2(
            input_ids=torch.tensor([ids]),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        return (
            model.tok_proj(hidden)[0].numpy(),
            model.rule_proj(hidden)[0].numpy(),
        )
