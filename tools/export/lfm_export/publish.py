"""CLI: `python -m lfm_export.publish` — push the exported ONNX repos to the Hub.

Requires an authenticated `hf` session (`hf auth login` with a *write* token).
Nothing here reads or stores a token itself; it relies on the CLI's own
credential store.

    python -m lfm_export.publish --dry-run          # show what would happen
    python -m lfm_export.publish --dtypes q8,q4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .onnx_export import DTYPE_FILES
from .spec import BY_NAME, SPECS, ModelSpec

# Always shipped alongside the graphs: transformers.js reads these from the repo root.
METADATA = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")

CARD = """---
license: other
license_name: lfm1.0
license_link: https://huggingface.co/{source}/blob/main/LICENSE
base_model: {source}
base_model_relation: quantized
library_name: transformers.js
pipeline_tag: {pipeline}
tags:
- onnx
- transformers.js
- lfm2
- quantized
language:
- en
- de
- es
- fr
- it
- nl
- pl
- pt
- ar
- hi
- ja
- ru
- tr
- vi
- zh
---

# {name}

ONNX export of [`{source}`]({source_url}), quantized to run **fully in the browser** through
[transformers.js](https://github.com/huggingface/transformers.js). No inference server: the weights are
fetched once, cached, and every forward pass happens in the tab.

{blurb}

All credit for the model itself goes to [Liquid AI](https://huggingface.co/LiquidAI). This repository
contains only a re-export; the weights are unchanged apart from quantization, and the original
[LFM Open License v1.0]({source_url}/blob/main/LICENSE) applies.

**[Try it in your browser →](https://kucukkanat.github.io/lfm-encoders/)** — no install, no API key.

![{screenshot_alt}]({screenshot})

Tooling, demo and the export pipeline: <https://github.com/kucukkanat/lfm-encoders>

## Files

| dtype | File | Size |
| --- | --- | --: |
{files}

The graph takes `input_ids` + `attention_mask`, is dynamic in batch and sequence, and returns
{outputs}.

## Usage

```js
import {{ AutoTokenizer, PreTrainedModel, Tensor }} from "@huggingface/transformers";

const id = "{repo}";
const tokenizer = await AutoTokenizer.from_pretrained(id);
const model = await PreTrainedModel.from_pretrained(id, {{ dtype: "q8" }});

const {{ input_ids }} = tokenizer("some text");
const out = await model({{
  input_ids,
  attention_mask: new Tensor("int64", new BigInt64Array(input_ids.dims[1]).fill(1n), input_ids.dims),
}});
```

`PreTrainedModel` rather than `AutoModel` is deliberate: this is a plain "feed the named inputs, read the
named outputs" session, not one of transformers.js's built-in architectures.

{extra}

## Accuracy

Measured from JavaScript against the fp32 PyTorch reference. Δ is the largest absolute difference in a
final probability.

{accuracy}

## Notes

- `q8` is smaller on disk but uses **more** browser RAM than fp32 and runs slower: onnxruntime's WASM
  kernels compute in float, so quantized weights are unpacked at session load. Quantization here buys
  download size, not speed or memory.
- Budget roughly 1.5 GB of RAM per resident model, and expect a tab to hold its high-water mark until
  reloaded.
- `fp16` / `q4f16` are deliberately absent: RMSNorm's variance overflows fp16 on this architecture and
  every hidden state collapses to zeros.
"""

BLURBS = {
    "fill-mask": (
        "The base bidirectional encoder with its tied masked-LM head. Outputs vocabulary logits and the\n"
        "final hidden states, so it doubles as a sentence-embedding backbone.",
    ),
    "zero-shot-routing": (
        "A zero-shot prompt router. Categories are ordinary prose supplied at call time — nothing is\n"
        "trained or cached per label set. One bidirectional pass over the category list *and* the text\n"
        "scores every category at once.",
    ),
    "zero-shot-token-matching": (
        "A zero-shot policy linter. Rules are ordinary prose supplied at call time. One pass scores every\n"
        "token against every rule, so adding a rule costs a few tokens rather than another model.",
    ),
    "masked-diffusion": (
        "A diffusion language model. It is not a decoder and emits no token stream: generation starts from\n"
        "a canvas of `<|mask|>` tokens and *denoises* it, committing only the most confident predictions\n"
        "each pass and re-guessing the rest with full sight of what landed on either side of them.",
    ),
}

EXTRA_DIFFUSION = """### Generating

The graph is a plain masked-LM forward; what makes it a chatbot is the loop around it, and the loop's
schedule ships in `config.json` under `diffusion` so consumers do not have to hard-code it:

```
[Question]
<your question>
[/Question]

[Answer]
<max_new_tokens copies of <|mask|>>
```

Each pass predicts every still-masked position, and the scheduler commits a subset:

1. **Blocks.** Unmasking is confined to a `block_size` window sweeping left to right. Without it the
   model scatters confident punctuation across the whole canvas and then has to write prose around it.
2. **Confidence.** Within the block, candidates are ranked by softmax probability. Anything above `tau`
   is committed immediately; otherwise just enough are taken to keep the block inside its step budget.
3. **Adjacency.** Two neighbouring positions are never committed in the same pass — each was predicted
   while the other was still masked, so both are individually likely and jointly often not.

Ids at or above `real_vocab_size` are alignment padding and must be excluded from the argmax *and* the
softmax denominator.

[`@lfm-encoder/tasks`](https://github.com/kucukkanat/lfm-encoders) implements the whole loop, including
frame-by-frame callbacks for animating the denoising.

The reference implementation caches K/V and shortconv state so later passes recompute only the active
block. This export has no cache inputs and re-runs the full canvas each pass instead — exact, simpler,
and a constant factor more compute.
"""

EXTRA_TWO_TOWER = """### Prompt format

Both projection towers expect one string laid out exactly like this — the model was trained on it and the
character arithmetic that locates each label depends on it byte for byte:

```
{heading}:
- label one
- label two

Text:
<the text>
```

`token_proj` is the query tower and `rule_proj` the key tower, both 256-d and emitted **per token**. Pool
the tokens covering each label to get its vector. Pooling after projecting is exact rather than an
approximation: both towers are affine, and an affine map commutes with a mean — which is what keeps one
static graph usable for any number of labels.

Scoring is `{scoring}`.

[`@lfm-encoder/tasks`](https://github.com/kucukkanat/lfm-encoders) implements all of this, including the
character-offset reconstruction transformers.js does not provide.
"""

ACCURACY = {
    "fill-mask": (
        "| dtype | max Δ | mean Δ | top-5 disagreements (3 cases) |\n"
        "| --- | --: | --: | --: |\n"
        "| `fp32` | 9.6e-5 | 4.1e-5 | 0 |\n"
        "| `q8` | 0.1846 | 0.1235 | 4 |\n"
        "| `q4` | 0.2180 | 0.0978 | 3 |"
    ),
    "zero-shot-routing": (
        "| dtype | max Δ | mean Δ | top-1 flips (4 cases) |\n"
        "| --- | --: | --: | --: |\n"
        "| `fp32` | 6.4e-5 | 1.6e-5 | 0 |\n"
        "| `q8` | 0.0910 | 0.0230 | 0 |\n"
        "| `q4` | 0.1221 | 0.0309 | 0 |"
    ),
    "zero-shot-token-matching": (
        "| dtype | max Δ | mean Δ | threshold flips (6 cases) |\n"
        "| --- | --: | --: | --: |\n"
        "| `fp32` | 8.4e-4 | 3.4e-4 | 0 |\n"
        "| `q8` | 0.5241 | 0.2328 | 3 |\n"
        "| `q4` | 0.3698 | 0.1818 | 3 |"
    ),
    "masked-diffusion": (
        "Per-logit error is the wrong metric for a generative loop — error that never moves an argmax is\n"
        "free, and error that does is compounded by every later pass. So this is measured by decoding the\n"
        "same prompts greedily with each dtype and counting *generated tokens* that differ from the fp32\n"
        "PyTorch decode (3 prompts, 32-token canvas, 16 passes).\n\n"
        "| dtype | differing tokens | mean fraction |\n"
        "| --- | --: | --: |\n"
        "| `fp32` | 0 | 0.000 |\n"
        "| `q8` | 16 | 0.167 |\n\n"
        "`fp32` is token-identical to PyTorch. `q8` paraphrases rather than degrades — it still answers\n"
        "the question — but it is not the same token stream.\n\n"
        "`q4` is **not published**: it fails the export's own cosine-similarity gate against fp32 (0.85,\n"
        "threshold 0.90). A one-shot encoder can absorb that; a loop that conditions each pass on the last\n"
        "cannot."
    ),
}

RAW = "https://raw.githubusercontent.com/kucukkanat/lfm-encoders/main/docs/screenshots"

SCREENSHOTS = {
    "fill-mask": ("fill-mask.png", "Fill-mask running in the browser"),
    "zero-shot-routing": ("prompt-routing.png", "Zero-shot prompt routing in the browser"),
    "zero-shot-token-matching": ("policy-linting.png", "Zero-shot policy linting in the browser"),
    "masked-diffusion": ("masked-diffusion.png", "Masked diffusion running in the browser"),
}

PIPELINE = {
    "fill-mask": "fill-mask",
    "zero-shot-routing": "zero-shot-classification",
    "zero-shot-token-matching": "token-classification",
    "masked-diffusion": "text-generation",
}


def build_card(spec: ModelSpec, out_dir: Path, repo: str, dtypes: tuple[str, ...]) -> str:
    config = json.loads((out_dir / "config.json").read_text())
    head = config.get("head")
    rows = []
    for dtype in dtypes:
        path = out_dir / "onnx" / DTYPE_FILES[dtype]
        if path.exists():
            rows.append(f"| `{dtype}` | `onnx/{path.name}` | {path.stat().st_size / 1e6:.0f} MB |")

    outputs = " and ".join(f"`{name}`" for name in config["onnx"]["outputs"])
    extra = EXTRA_DIFFUSION if spec.task == "masked-diffusion" else ""
    if head:
        scoring = (
            "cosine between the L2-normalised pooled towers, scaled by a learned temperature, "
            "then a softmax across labels"
            if head["kind"] == "cosine"
            else "a dot product scaled by 1/sqrt(256) plus a bias, through a sigmoid per (token, rule) pair"
        )
        extra = EXTRA_TWO_TOWER.format(heading=head["prefix_heading"], scoring=scoring)

    shot, alt = SCREENSHOTS[spec.task]
    return CARD.format(
        screenshot=f"{RAW}/{shot}",
        screenshot_alt=alt,
        name=spec.name,
        source=spec.repo,
        source_url=f"https://huggingface.co/{spec.repo}",
        repo=repo,
        pipeline=PIPELINE[spec.task],
        blurb="".join(BLURBS[spec.task]),
        files="\n".join(rows),
        outputs=outputs,
        extra=extra,
        accuracy=ACCURACY[spec.task],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lfm_export.publish", description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--dtypes", default="q8,q4")
    parser.add_argument(
        "--only", action="append", choices=sorted(BY_NAME), help="publish just this checkpoint"
    )
    parser.add_argument("--owner", help="Hub user or org (default: the logged-in user)")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--cards-only",
        action="store_true",
        help="re-upload README.md only, leaving the (unchanged, multi-GB) graphs alone",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        owner = args.owner or api.whoami()["name"]
    except Exception as error:  # not logged in -> say exactly what to do
        print(f"Not authenticated with the Hub ({error}).\nRun:  hf auth login", file=sys.stderr)
        return 1

    dtypes = tuple(d.strip() for d in args.dtypes.split(",") if d.strip())
    files = [DTYPE_FILES[d] for d in dtypes]

    specs = [BY_NAME[n] for n in args.only] if args.only else list(SPECS)
    for spec in specs:
        out_dir = args.out / spec.hub_id
        repo = f"{owner}/{spec.name}"
        if not (out_dir / "config.json").exists():
            print(f"[publish] {spec.name}: not exported, skipping")
            continue

        (out_dir / "README.md").write_text(build_card(spec, out_dir, repo, dtypes))
        present = [f for f in files if (out_dir / "onnx" / f).exists()]
        payload = (
            ["README.md"]
            if args.cards_only
            else [*METADATA, "README.md", *(f"onnx/{f}" for f in present)]
        )
        existing = [p for p in payload if (out_dir / p).exists()]
        total = sum((out_dir / p).stat().st_size for p in existing) / 1e6

        print(f"[publish] {repo}: {len(existing)} files, {total:.0f} MB")
        for p in existing:
            print(f"            {p}")
        if args.dry_run:
            continue

        api.create_repo(repo, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=repo,
            folder_path=str(out_dir),
            allow_patterns=existing,
            commit_message=(
                "Update model card" if args.cards_only else "Add ONNX export for transformers.js"
            ),
        )
        print(f"[publish] {repo}: https://huggingface.co/{repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
