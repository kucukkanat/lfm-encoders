"""CLI: `python -m lfm_export.check` — quantization parity report + TS fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from . import parity as P
from .spec import BY_NAME, ModelSpec

from .onnx_export import DTYPE_FILES as FILES


def _fmt(x: float) -> str:
    return f"{x:.2e}" if x < 1e-3 else f"{x:.5f}"


def _sessions(onnx_dir: Path) -> dict[str, Any]:
    found = {}
    for dtype, name in FILES.items():
        path = onnx_dir / name
        if path.exists():
            found[dtype] = P._session(path)
    return found


def check_two_tower(spec: ModelSpec, out_root: Path) -> dict[str, Any]:
    repo = out_root / spec.name
    head = json.loads((repo / "config.json").read_text())["head"]
    heading = head["prefix_heading"]
    cases = P.ROUTING_CASES if head["kind"] == "cosine" else P.LINTING_CASES

    tokenizer = AutoTokenizer.from_pretrained(spec.repo, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        spec.repo, trust_remote_code=True, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    sessions = _sessions(repo / "onnx")

    errors: dict[str, list[float]] = {d: [] for d in sessions}
    flips: dict[str, int] = {d: 0 for d in sessions}
    fixtures = []

    for text, labels in cases:
        prefix = P.build_prefix(heading, labels)
        full = prefix + text
        enc = P.encode(tokenizer, full)
        text_pool = P.pool_matrix(enc.offsets, [P.text_span(prefix, full)])
        rule_pool = P.pool_matrix(enc.offsets, P.label_ranges(heading, labels))

        # The linter scores every token, but only the ones in the draft region are
        # ever surfaced; prefix tokens (the rules restating themselves) are noise
        # the UI drops. Measuring them would overstate the quantization cost.
        start = len(prefix)
        shown = [i for i, (a, b) in enumerate(enc.offsets) if b > start and a != b]

        def scores(tok_proj: np.ndarray, rule_proj: np.ndarray) -> np.ndarray:
            raw = P.score_from_projections(tok_proj, rule_proj, text_pool, rule_pool, head)
            if head["activation"] == "softmax":
                return P.softmax(raw)
            return P.sigmoid(raw[shown])

        ref = scores(*P.reference_two_tower(model, enc.ids))
        for dtype, session in sessions.items():
            got = scores(*[o[0] for o in P._run(session, enc.ids)])
            errors[dtype].append(float(np.abs(got - ref).max()))
            if head["activation"] == "softmax":
                flips[dtype] += int(got.argmax() != ref.argmax())
            else:
                flips[dtype] += int(((got > 0.5) != (ref > 0.5)).sum())

        fixtures.append(
            {
                "text": text,
                "labels": list(labels),
                "prefix": prefix,
                "input_ids": enc.ids,
                "offsets": [list(o) for o in enc.offsets],
                "reference": np.asarray(ref, dtype=np.float64).round(6).tolist(),
            }
        )

    (repo / "fixtures.json").write_text(
        json.dumps({"task": spec.task, "head": head, "cases": fixtures}, indent=2) + "\n"
    )
    return {"errors": errors, "flips": flips, "n": len(cases), "metric": head["activation"]}


def check_mlm(spec: ModelSpec, out_root: Path, topk: int = 5) -> dict[str, Any]:
    repo = out_root / spec.name
    tokenizer = AutoTokenizer.from_pretrained(spec.repo, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        spec.repo, trust_remote_code=True, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    sessions = _sessions(repo / "onnx")

    errors: dict[str, list[float]] = {d: [] for d in sessions}
    flips: dict[str, int] = {d: 0 for d in sessions}
    fixtures = []

    for text in P.MLM_CASES:
        enc = P.encode(tokenizer, text)
        pos = enc.ids.index(tokenizer.mask_token_id)
        with torch.no_grad():
            ref_logits = model(input_ids=torch.tensor([enc.ids])).logits[0, pos].numpy()
        ref_top = ref_logits.argsort()[::-1][:topk]
        ref_probs = P.softmax(ref_logits)

        for dtype, session in sessions.items():
            logits = P._run(session, enc.ids)[0][0, pos]
            errors[dtype].append(float(np.abs(P.softmax(logits) - ref_probs).max()))
            flips[dtype] += topk - len(set(logits.argsort()[::-1][:topk]) & set(ref_top))

        fixtures.append(
            {
                "text": text,
                "input_ids": enc.ids,
                "mask_position": pos,
                "top_k": [
                    {"id": int(i), "token": tokenizer.decode([int(i)]), "prob": round(float(ref_probs[i]), 6)}
                    for i in ref_top
                ],
            }
        )

    (repo / "fixtures.json").write_text(
        json.dumps({"task": spec.task, "top_k": topk, "cases": fixtures}, indent=2) + "\n"
    )
    return {"errors": errors, "flips": flips, "n": len(P.MLM_CASES), "metric": "top-5"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lfm_export.check", description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--only", action="append", choices=sorted(BY_NAME))
    args = parser.parse_args(argv)

    specs = [BY_NAME[n] for n in args.only] if args.only else list(BY_NAME.values())
    rows = []
    for spec in specs:
        if not (args.out / spec.name / "config.json").exists():
            print(f"[check] {spec.name}: not exported, skipping", flush=True)
            continue
        print(f"[check] {spec.name}: comparing against fp32 PyTorch ...", flush=True)
        report = check_mlm(spec, args.out) if spec.task == "fill-mask" else check_two_tower(spec, args.out)
        for dtype, errs in report["errors"].items():
            rows.append(
                (
                    spec.name.removesuffix("-ONNX"),
                    dtype,
                    max(errs),
                    float(np.mean(errs)),
                    report["flips"][dtype],
                    report["n"],
                    report["metric"],
                )
            )

    width = max((len(r[0]) for r in rows), default=10)
    print(f"\n{'model'.ljust(width)}  {'dtype':<6} {'max Δ':>10} {'mean Δ':>10}  disagreements")
    print("-" * (width + 52))
    for name, dtype, mx, mean, flip, n, metric in rows:
        note = f"{flip} ({metric}, {n} cases)"
        print(f"{name.ljust(width)}  {dtype:<6} {_fmt(mx):>10} {_fmt(mean):>10}  {note}")
    print("\nΔ is max absolute difference in the final probability vs fp32 PyTorch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
