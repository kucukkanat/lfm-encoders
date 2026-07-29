# tools/export

Torch → ONNX → quantized ONNX, laid out as transformers.js model repos. Everything under `models/` is
produced here; nothing in it is committed.

Two CLIs:

| Command | Does |
| --- | --- |
| `python -m lfm_export` | Downloads the checkpoints, traces each to ONNX, quantizes, writes the repos. |
| `python -m lfm_export.check` | Compares every exported graph against fp32 PyTorch and writes `fixtures.json`. |

## Requirements

Python **3.12**. The exporter is not packaged (no `pyproject.toml`, no lockfile) — it is a plain package
on `sys.path`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch "transformers>=5.12" onnx onnxruntime onnx_ir onnxconverter_common
```

`numpy` and `huggingface_hub` arrive transitively. `transformers>=5.12` is a hard floor: the exporter uses
the `dtype=` argument to `from_pretrained` and the LFM2 modeling module's current `create_causal_mask`
seam. The checkpoints need `trust_remote_code=True`.

The version pins above are the shape of the environment, not a reproduction of it — the venv used during
development is not in the repo. If a graph fails to load after an upstream release, `verify()` will say so
before anything ships.

### Running it

`lfm_export` lives in `tools/export`, which is not on the path by default:

```bash
# From the repo root
bun run export           # -> python3 -m lfm_export --out models
bun run export:check     # -> python3 -m lfm_export.check --out models

# Or directly
cd tools/export && python -m lfm_export --out ../../models
```

The root scripts set `PYTHONPATH` themselves; invoking `python -m lfm_export` directly needs
`PYTHONPATH=tools/export`.

## `python -m lfm_export`

```bash
python -m lfm_export --out models
python -m lfm_export --only LFM2.5-Encoder-350M-Prompt-Router --dtypes q4
python -m lfm_export --dtypes q8 --keep-fp32 --force
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--out DIR` | `models` | Where the repos are written. One subdirectory per model. |
| `--only NAME` | all three | Repeatable. Takes the *source* checkpoint name, without the `-ONNX` suffix: `LFM2.5-Encoder-350M`, `LFM2.5-Encoder-350M-Prompt-Router`, `LFM2.5-Encoder-350M-Policy-Linter`. |
| `--dtypes LIST` | `q8,q4` | Comma-separated subset of `fp32,fp16,q8,q4,q4f16`. Unknown names are a hard error. |
| `--keep-fp32` | off | Keep `model.onnx` after quantizing. Required by the strict-parity tests. |
| `--force` | off | Delete and re-export a repo that already has quantized artifacts. |

Expect tens of minutes and ~4.3 GB of downloads for a cold full run. Most of the wall time is tracing;
most of the memory is the fp32 graph.

Each repo comes out as:

```
models/LFM2.5-Encoder-350M-Prompt-Router-ONNX/
	config.json              # model_type, task, head scalars, onnx input/output names
	tokenizer.json           # copied verbatim from the source repo
	tokenizer_config.json
	special_tokens_map.json  # base encoder only; the fine-tuned repos have none
	fixtures.json            # written by lfm_export.check
	onnx/
		model.onnx             # only with --keep-fp32
		model_quantized.onnx   # q8 — note the legacy name, see below
		model_q4.onnx
		model_q4f16.onnx
```

### What each dtype produces

| dtype | File | How |
| --- | --- | --- |
| `fp32` | `model.onnx` | `torch.onnx.export`, opset 17, `dynamo=False`, eager attention, constant folding on. The reference every other variant is derived from. |
| `q8` | `model_quantized.onnx` | `quantize_dynamic`, QInt8, per-channel, subgraphs enabled. Quantizes the embedding table too. |
| `q4` | `model_q4.onnx` | `MatMulNBitsQuantizer`, block 32, symmetric, `accuracy_level=4` — the same preset transformers.js's `q4` users expect. **MatMul only**, so the 65536×1024 embedding `Gather` stays fp32, which is why `q4` lands *larger* than `q8` here. |
| `q4f16` | `model_q4f16.onnx` | `q4` folded to fp16 with `onnxconverter_common.float16`, halving everything the 4-bit pass left alone: embeddings, norms, biases. Smallest file, but **collapses to zeros** on WASM and WebGPU alike — RMSNorm's variance overflows fp16. Not built by default; `verify()` rejects it. |
| `fp16` | `model_fp16.onnx` | The fp32 graph converted to fp16. Not built by default; useful for isolating a fp16 problem from a quantization problem. |

The `q8` filename is not a typo. transformers.js resolves `dtype` to a filename suffix and `q8` maps to the
legacy `_quantized`, so a stock `dtype: "q8"` load only finds the file if it is named that way.

Two derived behaviours worth knowing: asking for `q4f16` builds `q4` first and deletes it afterwards unless
`q4` was also requested, and `fp32` is always produced even when not requested — it is the input to
everything else — then deleted unless `--keep-fp32` or `fp32 ∈ --dtypes`.

Every artifact passes through `verify()` before the exporter reports success: it loads the file in
onnxruntime, runs a fixed 10-token sequence, and rejects any output containing NaN or Inf. An ONNX file
that loads but computes garbage is worse than no file — it ships and breaks in someone's browser.

### Resume

Two independent levels, both automatic:

- **Per repo.** A repo whose `onnx/` already contains any `model_*.onnx` is skipped entirely with
  `exists, skipping`. `--force` deletes the directory and starts over.
- **Per graph.** Inside a repo, an existing `onnx/model.onnx` is reused and the trace is skipped —
  `reusing existing fp32 graph`. Tracing is the slow, memory-hungry step and every dtype is derived from
  its output, so re-quantizing at a different precision is cheap instead of a full reload-and-retrace.

Note the interaction: the per-repo glob is `model_*.onnx`, which does not match `model.onnx`. A repo
holding only the fp32 graph is therefore *not* considered exported, and a re-run picks up at quantization.
That is the fast path for adding a dtype:

```bash
python -m lfm_export --out models --dtypes q8 --keep-fp32          # trace once
python -m lfm_export --out models --dtypes q4 --keep-fp32 --force  # re-quantize, no retrace
```

## Graph fixes

Three non-obvious transforms the exporter applies. Each exists because the naive path produces a file that
either does not load or is unusable in a browser; all three are covered by the numeric parity check rather
than trusted on reasoning.

### 1. Broadcast pad-mask substitution

`_use_broadcast_mask()` monkey-patches `transformers.models.lfm2.modeling_lfm2.create_causal_mask` for the
duration of the trace.

The shipped `_bidirectional_mask` materialises a dense `(batch, 1, seq, seq)` additive mask by
slice-assigning into a zero tensor. Traced, that becomes a `ScatterND` plus a tensor costing 4 MB at
seq=1024 and growing quadratically — and on a WebGPU backend `ScatterND` is an op that can force a
readback. Every row of that mask is identical (it only ever encodes *key* padding), so `(batch, 1, 1, seq)`
broadcasts to the same attention scores. Attention slices the mask to the key length, a no-op here, so the
substitution is exact for the no-KV-cache encoder forward.

The parity harness compares against the *unpatched* PyTorch model, so any drift shows up as a `fp32` error
well above the ~1e-4 floor.

Related, in the same area: the graph is traced with a **2-row padded batch**, not batch-1. The short-conv
path guards its padding fix-up behind `attention_mask.shape[0] > 1`, so a batch-1 trace would bake in a
graph that silently ignores padding and returns wrong answers for every batched call. Feeding a genuine pad
makes the mask multiply part of the graph, which is correct for batch 1 too — its mask is all ones.

### 2. `_isolate_outputs`

Runs before the fp16 conversion. Gives every graph output a tensor of its own behind an `Identity` node.

The masked-LM graph exposes `last_hidden_state` as an output *and* feeds it to the LM head, so one tensor
is simultaneously a graph boundary and an interior value. `keep_io_types=True` then pins it to fp32 for the
boundary's sake, and the now-fp16 LM head receives an fp32 activation — a type clash that only surfaces
when onnxruntime loads the file, not when the converter writes it.

Splitting the alias gives the converter an unambiguous boundary to put its own `Cast` on and leaves the
interior free to become fp16. `Identity` is free at runtime; onnxruntime elides it during graph
optimisation.

### 3. `_retype_interior_casts`

Runs after the fp16 conversion.

`convert_float_to_float16` retypes *tensors*, but it will not touch a `Cast` whose source is an integer or
boolean — the padding mask (`int64 → float`), the `Equal` result in the mask builder, the rotary position
table. Those nodes keep `to=FLOAT` and hand an fp32 tensor to a consumer the converter just moved to fp16,
and the model fails type checking the moment it loads.

With `keep_io_types=True` and no op block list, the converter's contract is "fp32 only at the graph
boundary", so any such `Cast` that does not feed a graph output belongs in fp16. That same test excludes
the `Cast` nodes the converter itself inserted in front of the outputs.

Shape inference must also run before the conversion: without it the converter leaves scalar initializers
(the RMSNorm epsilons) at fp32 while their consumers become fp16, and the graph fails to load with a
type-binding error.

## `python -m lfm_export.check`

```bash
python -m lfm_export.check --out models [--only NAME]
```

Loads the **unmodified** upstream PyTorch model (remote code, fp32, eager attention), reproduces each head
in NumPy, and compares against every ONNX variant it finds in `onnx/`. Prints max Δ, mean Δ and
disagreement counts per dtype, where Δ is the largest absolute difference in a final probability.

This measures the *graph* under Python's onnxruntime. It is not the number a browser sees — for that, run
`bun run parity` from the repo root, which measures the same fixtures through Node's onnxruntime and the
TypeScript pooling. The two disagree, notably on fp16, because the runtimes link different kernels. Both
numbers are real; neither substitutes for the other.

Run it after any change to the graph fixes or the quantization presets.

### `fixtures.json`

The check writes one `fixtures.json` per repo. This is the contract between the two languages: everything
the browser shows is produced by the ONNX graph *and* a pooling/scoring step reimplemented in TypeScript,
and only a frozen fixture pins both.

Two-tower repos (router, linter):

```jsonc
{
	"task": "zero-shot-routing",
	"head": { "kind": "cosine", "scale": 1.371…, "bias": -0.272…, "prefix_heading": "Categories", … },
	"cases": [{
		"text": "My invoice from last month charged me twice, can I get a refund?",
		"labels": ["billing and payments", "technical support", …],
		"prefix": "Categories:\n- billing and payments\n…\n\nText:\n",
		"input_ids": [1, 544, 32830, …],
		"offsets": [[0, 0], [0, 1], [1, 10], …],  // from the HF *fast* tokenizer
		"reference": [0.83812, 0.053959, …]       // fp32 PyTorch probabilities
	}]
}
```

The base encoder's is the same idea with `mask_position` and a `top_k` list of `{id, token, prob}`.

The diffusion repo's is different in kind, because per-logit error is the wrong question for a decode
loop — error that never moves an argmax is free, and error that does is compounded by every later pass.
So the check *decodes* each prompt greedily with fp32 PyTorch and with every ONNX variant, and reports how
many generated **tokens** differ:

```jsonc
{
	"task": "masked-diffusion",
	"diffusion": { "mask_token_id": 16, "block_size": 16, "tau": 0.9, "template": { … } },
	"cases": [{
		"prompt": "What is the capital of France?",
		"rendered": "[Question]\nWhat is the capital of France?\n[/Question]\n\n[Answer]\n",
		"input_ids": [1, 2263, 8598, …],
		"max_new": 32, "steps": 16, "block_size": 16,
		"reference_ids": [504, 5940, 432, …],
		"reference_text": "The capital of France is Paris.\n[/Answer]\n\n"
	}]
}
```

`check.diffuse` is a deliberately literal transcription of `packages/tasks/src/diffusion.ts`, so a
disagreement between them is a disagreement about the *weights* rather than about the schedule. The TS
integration test then asserts the stronger claim in the other direction: at fp32 the JavaScript loop
reproduces `reference_ids` exactly.

`offsets` is the load-bearing field. transformers.js has no `return_offsets_mapping`, so
`@lfm-encoder/core` reconstructs character offsets from token byte lengths; freezing HuggingFace's own
offsets here is the only thing that catches a drift in that reconstruction — no Python-side check could see
it. `input_ids` catches JS tokenizer drift the same way.

Consumed by:

- `packages/tasks/test/integration/*.test.ts` (`bun test:integration`) — asserts ids and offsets match
  exactly, asserts fp32 probabilities to 1e-3, and asserts the quantized graphs at least keep the same
  winner.
- `packages/tasks/scripts/parity.ts` (`bun run parity`) — the JavaScript-side accuracy table.

Both skip themselves when `fixtures.json` is absent, so a checkout without weights still runs green. The
model root comes from `LFM_MODEL_ROOT`, defaulting to `./models`.

The cases live in `lfm_export/parity.py` (`ROUTING_CASES`, `LINTING_CASES`, `MLM_CASES`) and
`lfm_export/check.py` (`DIFFUSION_CASES`), and deliberately
include non-ASCII text — German, French, Spanish — because that is where a byte-offset reconstruction
breaks if it is wrong.

## Adding a model

`lfm_export/spec.py` is the registry. A `ModelSpec` is a source repo, an auto-class, a graph wrapper from
`lfm_export/graphs.py`, a `head_config` callable that pulls the head's scalars out of the loaded PyTorch
model into `config.json`, and an `extra_config` callable for anything a consumer needs that is not
derivable from the graph. Adding another LFM2.5 head means one `ModelSpec` and, if the head is not a
two-tower cosine/dot, one `nn.Module` wrapper whose `forward` returns per-token tensors.

The diffusion checkpoint is the case where `extra_config` earns its keep: it reuses `MaskedLmGraph`
unchanged — architecturally it *is* the base encoder — and everything that makes it a chatbot is the
decode schedule and prompt template written into `config.json` under `diffusion`. Recording them with the
weights is what keeps the loop from being hard-coded on the JavaScript side.

Keep the wrappers returning *per-token* projections rather than pooled scores. That is what keeps one
static ONNX file usable for any zero-shot label set, and it is exact — both towers are affine and
`W @ mean(h) + b == mean(W @ h + b)`.

One constraint on new specs: `config.json` must declare `model_type: "lfm2-bidirectional"`. Upstream
`lfm2` is the *causal* decoder in transformers.js and expects KV-cache inputs these graphs do not have, so
a load under that type fails on missing inputs. The custom type makes transformers.js fall through to its
generic session runner — "feed the named inputs, return the named outputs" — which is all these graphs
need.
