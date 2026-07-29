"""Torch -> ONNX -> quantized ONNX, laid out as a transformers.js model repo."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx
import torch
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

from .spec import JS_MODEL_TYPE, ModelSpec

# The 4-bit quantizer logs one INFO line per graph node — tens of thousands for a
# 350M model, which buries every message worth reading.
logging.getLogger("onnxruntime.quantization").setLevel(logging.WARNING)

OPSET = 17

# Files transformers.js reads straight from the repo root. `special_tokens_map`
# is absent from the fine-tuned repos, hence the "if it exists" copy.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


@dataclass(frozen=True)
class ExportResult:
    spec: ModelSpec
    out_dir: Path
    files: dict[str, int]  # dtype -> bytes


def _load(spec: ModelSpec) -> tuple[Any, Any]:
    """Load the checkpoint in fp32 with eager attention.

    Eager is deliberate: SDPA lowers to a fused op whose ONNX form varies by
    backend, while eager decomposes into MatMul/Softmax that every onnxruntime
    build — including the WASM one in the browser — supports and re-fuses on its
    own. The numerics are identical.
    """
    auto = AutoModelForMaskedLM if spec.auto_class == "AutoModelForMaskedLM" else AutoModel
    model = auto.from_pretrained(
        spec.repo,
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(spec.repo, trust_remote_code=True)
    return model, tokenizer


def _sample_inputs(tokenizer: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """A padded 2-row batch.

    Tracing with batch>1 *and* a real pad matters: the short-conv path guards its
    padding fix-up behind `attention_mask.shape[0] > 1`, so a batch-of-one trace
    would bake in a graph that silently ignores padding and gives wrong answers
    for every batched call. Feeding a genuine pad makes the mask multiply part of
    the graph, which is correct for batch 1 too (its mask is all ones).
    """
    pad = tokenizer.pad_token_id or 0
    long = tokenizer("Categories:\n- billing\n\nText:\nrefund my last invoice")["input_ids"]
    short = tokenizer("Text:\nhello")["input_ids"]
    width = max(len(long), len(short))
    ids, mask = [], []
    for row in (long, short):
        ids.append(row + [pad] * (width - len(row)))
        mask.append([1] * len(row) + [0] * (width - len(row)))
    return torch.tensor(ids, dtype=torch.int64), torch.tensor(mask, dtype=torch.int64)


def _use_broadcast_mask() -> None:
    """Replace the encoder's pad-mask builder with a broadcasting equivalent.

    The shipped `_bidirectional_mask` materialises a dense `(batch, 1, seq, seq)`
    additive mask by slice-assigning into a zero tensor. Traced, that becomes a
    ScatterND plus a tensor that costs 4 MB at seq=1024 and grows quadratically —
    on a browser WebGPU backend it is also an op that can force a readback.

    Every row of that mask is identical (it only ever encodes *key* padding), so
    `(batch, 1, 1, seq)` broadcasts to the same attention scores. Attention
    slices the mask to the key length, which is a no-op here, so this is an exact
    substitution for the no-KV-cache encoder forward. The parity harness compares
    against the unpatched model, so any drift would surface there.
    """
    import transformers.models.lfm2.modeling_lfm2 as lfm2

    def broadcast_mask(config, input_embeds=None, attention_mask=None, **kwargs):
        embeds = input_embeds if input_embeds is not None else kwargs.get("inputs_embeds")
        if config._attn_implementation == "flash_attention_2":
            return attention_mask if attention_mask is not None else None
        if attention_mask is None:
            return None
        pad = (attention_mask == 0).to(embeds.dtype)
        return (pad * -1e9)[:, None, None, :]

    lfm2.create_causal_mask = broadcast_mask


def export_fp32(spec: ModelSpec, model: Any, tokenizer: Any, path: Path) -> None:
    _use_broadcast_mask()
    graph = spec.graph(model).eval()
    ids, mask = _sample_inputs(tokenizer)
    names = list(spec.output_names)

    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            graph,
            (ids, mask),
            str(path),
            input_names=["input_ids", "attention_mask"],
            output_names=names,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                **{n: {0: "batch", 1: "sequence"} for n in names},
            },
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )


def _load_onnx(path: Path) -> onnx.ModelProto:
    return onnx.load(str(path), load_external_data=True)


def _save_onnx(model: onnx.ModelProto, path: Path) -> None:
    """Write, spilling tensors to a sidecar only when the 2 GB protobuf cap demands it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if model.ByteSize() > 2_000_000_000:
        onnx.save(
            model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{path.name}_data",
            convert_attribute=False,
        )
    else:
        onnx.save(model, str(path))


def _isolate_outputs(model: onnx.ModelProto) -> int:
    """Give every graph output a tensor of its own, behind an `Identity`.

    The masked-LM graph exposes `last_hidden_state` as an output *and* feeds it
    to the LM head, so one tensor is simultaneously a graph boundary and an
    interior value. `keep_io_types=True` then pins it to fp32 for the boundary's
    sake and the fp16 LM head receives an fp32 activation — a type clash that
    only shows up when onnxruntime loads the file.

    Splitting the alias first gives the converter an unambiguous boundary to put
    its own Cast on, and leaves the interior free to become fp16. `Identity` is
    free at runtime; onnxruntime elides it during graph optimisation.
    """
    consumed = {name for node in model.graph.node for name in node.input}
    produced = {name: node for node in model.graph.node for name in node.output}
    added = 0
    for output in model.graph.output:
        if output.name not in consumed:
            continue
        interior = f"{output.name}__interior"
        producer = produced.get(output.name)
        if producer is None:
            continue
        for i, name in enumerate(producer.output):
            if name == output.name:
                producer.output[i] = interior
        for node in model.graph.node:
            for i, name in enumerate(node.input):
                if name == output.name:
                    node.input[i] = interior
        model.graph.node.append(
            onnx.helper.make_node("Identity", [interior], [output.name], name=f"Isolate_{output.name}")
        )
        added += 1  # appended last, so it already follows its producer
    return added


def _retype_interior_casts(model: onnx.ModelProto, original: set[str]) -> int:
    """Point float-producing `Cast` nodes at fp16 after a float16 conversion.

    `convert_float_to_float16` retypes *tensors*, but it will not touch a `Cast`
    whose source is an integer or boolean — the padding mask (int64 -> float),
    the `Equal` result in the mask builder, the rotary position table. Those
    nodes keep `to=FLOAT` and hand an fp32 tensor to a consumer the converter
    just moved to fp16, so the model fails type checking the moment it loads.

    Only casts that were in the traced graph are eligible, hence `original`. The
    converter inserts its own casts to bracket blocked nodes and graph outputs,
    and those are *supposed* to produce fp32 — retyping them reintroduces the
    very type clash this function exists to remove.
    """
    outputs = {o.name for o in model.graph.output}
    changed = 0
    for node in model.graph.node:
        if node.op_type != "Cast" or node.name not in original or node.output[0] in outputs:
            continue
        target = next((a for a in node.attribute if a.name == "to"), None)
        if target is not None and target.i == onnx.TensorProto.FLOAT:
            target.i = onnx.TensorProto.FLOAT16
            changed += 1
    return changed


def verify(path: Path, reference: Path | None = None, *, min_similarity: float = 0.9) -> None:
    """Load the graph and, when given a reference, check it still computes it.

    A broken artifact is worse than a missing one: it ships and fails in
    someone's browser. Two failure modes are caught here.

    NaN/Inf is the easy one. The dangerous one is *collapse* — an fp16 pass that
    overflows a reduction leaves every hidden state at zero, and the model then
    loads cleanly, runs at full speed, and returns a uniform distribution for
    every input. An absolute-difference threshold cannot separate that from
    honest quantization noise, since 4-bit weights legitimately move a
    projection by more than 1.0. Cosine similarity can: quantization rotates the
    output vector slightly, collapse destroys it.
    """
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ids = np.asarray([[1, 4109, 261, 1830, 9, 44, 512, 27, 3001, 88]], dtype=np.int64)
    feed = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
    got = session.run(None, feed)
    for meta, value in zip(session.get_outputs(), got):
        if not np.isfinite(value).all():
            raise ValueError(f"{path.name}: output '{meta.name}' contains NaN/Inf")
    if reference is None:
        return

    want = ort.InferenceSession(str(reference), providers=["CPUExecutionProvider"]).run(None, feed)
    for meta, a, b in zip(session.get_outputs(), got, want):
        x, y = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
        norms = np.linalg.norm(x) * np.linalg.norm(y)
        similarity = float(x @ y / norms) if norms > 0 else 0.0
        if similarity < min_similarity:
            raise ValueError(
                f"{path.name}: output '{meta.name}' has cosine similarity "
                f"{similarity:.3f} to {reference.name} (need {min_similarity}). "
                "The quantized graph is not computing the same function."
            )


def quantize_q4(src: Path, dst: Path, *, block_size: int = 32) -> None:
    """4-bit block-wise weight-only quantization of every MatMul.

    Matches the `q4` preset transformers.js ships (block 32, symmetric,
    accuracy_level 4) so a `dtype: 'q4'` load behaves the way its users expect.
    Note this touches MatMul nodes only — the 65536x1024 embedding table is a
    Gather and stays at full width, which is why `q4f16` is the one worth
    shipping to a browser.
    """
    try:  # onnxruntime >= 1.22 renamed the module
        from onnxruntime.quantization.matmul_nbits_quantizer import (
            DefaultWeightOnlyQuantConfig,
            MatMulNBitsQuantizer,
        )

        config = DefaultWeightOnlyQuantConfig(
            block_size=block_size, is_symmetric=True, accuracy_level=4, bits=4
        )
        quantizer = MatMulNBitsQuantizer(_load_onnx(src), algo_config=config)
    except ImportError:
        from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer

        quantizer = MatMul4BitsQuantizer(
            _load_onnx(src), block_size=block_size, is_symmetric=True, accuracy_level=4
        )
    quantizer.process()
    _save_onnx(quantizer.model.model, dst)


# transformers.js resolves `dtype` to a filename suffix; `q8` maps to the legacy
# `_quantized` rather than `_q8`, so the artifact has to be named that way for a
# stock `dtype: "q8"` load to find it.
DTYPE_FILES = {
    "fp32": "model.onnx",
    "fp16": "model_fp16.onnx",
    "q8": "model_quantized.onnx",
    "q4": "model_q4.onnx",
    "q4f16": "model_q4f16.onnx",
}


def quantize_q8(src: Path, dst: Path) -> None:
    """8-bit dynamic weight quantization, matching the transformers.js `q8` preset.

    Worth reaching for over `q4` on this family for two reasons. Accuracy is the
    obvious one. The less obvious one is size: `q4` only touches MatMul, so the
    65536x1024 embedding table — a Gather, and 268 MB of the file — stays fp32,
    whereas `q8` quantizes it too. On these encoders q8 lands *smaller* than q4
    while staying far closer to the fp32 reference.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(src),
        str(dst),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        extra_options={"EnableSubgraph": True},
    )


def to_fp16(src: Path, dst: Path) -> None:
    """Halve everything the quantizer left alone (embeddings, norms, biases)."""
    from onnxconverter_common import float16

    # Shape inference must run: without it the converter leaves scalar
    # initializers (the RMSNorm epsilons) at fp32 while their consumers become
    # fp16, and the graph fails to load with a type-binding error.
    source = _load_onnx(src)
    _isolate_outputs(source)
    traced_casts = {n.name for n in source.graph.node if n.op_type == "Cast"}
    model = float16.convert_float_to_float16(
        source,
        keep_io_types=True,
        op_block_list=[*float16.DEFAULT_OP_BLOCK_LIST, *NORM_OPS],
    )
    _retype_interior_casts(model, traced_casts)
    _save_onnx(model, dst)


# The ops RMSNorm is built from. Kept out of the fp16 conversion — see below.
NORM_OPS = ("Pow", "ReduceMean", "Sqrt", "Div")


def _norm_nodes(model: onnx.ModelProto) -> list[str]:
    """Names of the RMSNorm nodes, which must keep computing in fp32.

    RMSNorm squares its input and sums over 1024 channels. Activations of even
    modest size overflow fp16's 65504 ceiling in that sum, `rsqrt(inf)` is 0, and
    the layer output collapses to zeros — which is why `Lfm2RMSNorm` upcasts to
    float32 in PyTorch before taking the variance.

    Converting the graph wholesale threw that protection away: the exported q4f16
    model loaded fine and returned a uniform distribution for every input,
    because every hidden state had been zeroed. Blocking these nodes makes the
    converter bracket them with casts instead, restoring the original numerics
    for the cost of a handful of conversions per layer.
    """
    return [node.name for node in model.graph.node if "norm/" in node.name.lower()]


def write_repo_metadata(
    spec: ModelSpec, model: Any, tokenizer: Any, out_dir: Path
) -> dict[str, Any]:
    """Emit the config + tokenizer files transformers.js expects at the repo root."""
    src_config = AutoConfig.from_pretrained(spec.repo, trust_remote_code=True).to_dict()
    hidden = src_config["hidden_size"]

    config = {
        "model_type": JS_MODEL_TYPE,
        "architectures": [spec.graph.__name__],
        "task": spec.task,
        "source_model": spec.repo,
        "hidden_size": hidden,
        "vocab_size": src_config["vocab_size"],
        # The checkpoint pads its vocabulary for kernel alignment; only ids the
        # tokenizer knows about were ever trained, and consumers need to know
        # where the padding starts to avoid offering untrained tokens.
        "real_vocab_size": len(tokenizer),
        "num_hidden_layers": src_config["num_hidden_layers"],
        "layer_types": src_config["layer_types"],
        "max_position_embeddings": src_config["max_position_embeddings"],
        "transformers.js_config": {
            "kv_cache_dtype": None,
            "use_external_data_format": False,
        },
        "onnx": {
            "inputs": ["input_ids", "attention_mask"],
            "outputs": list(spec.output_names),
        },
    }
    head = spec.head_config(model)
    if head:
        config["head"] = {**head, "proj_dim": model.tok_proj.out_features}
    config.update(spec.extra_config(model, tokenizer))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    src_dir = Path(tokenizer.name_or_path)
    if not src_dir.is_dir():  # loaded by hub id -> resolve the snapshot on disk
        from huggingface_hub import snapshot_download

        src_dir = Path(snapshot_download(spec.repo, allow_patterns=list(TOKENIZER_FILES)))
    for name in TOKENIZER_FILES:
        candidate = src_dir / name
        if candidate.is_file():
            shutil.copy2(candidate, out_dir / name)
    return config
