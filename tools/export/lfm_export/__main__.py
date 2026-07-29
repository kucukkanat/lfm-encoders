"""CLI: export the LFM2.5 encoders to browser-ready ONNX repos.

    python -m lfm_export --out ../../models
    python -m lfm_export --only LFM2.5-Encoder-350M-Prompt-Router --dtypes q4
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from .onnx_export import (
    DTYPE_FILES,
    export_fp32,
    quantize_q4,
    quantize_q8,
    to_fp16,
    verify,
    write_repo_metadata,
)
from .spec import BY_NAME, SPECS, ModelSpec

# fp32 is the parity reference and the input every other variant is derived
# from, so it is always produced even when it is not requested for shipping.
DTYPES = ("fp32", "fp16", "q8", "q4", "q4f16")
DEFAULT_DTYPES = ("q8", "q4")


def _mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.parent.glob(path.name + "*")) / 1e6


def _log(message: str) -> None:
    print(f"[export] {message}", flush=True)


def run(spec: ModelSpec, out_root: Path, dtypes: tuple[str, ...], keep_fp32: bool) -> None:
    from .onnx_export import _load  # imported lazily: pulls torch + the checkpoint

    out_dir = out_root / spec.hub_id
    onnx_dir = out_dir / "onnx"
    fp32 = onnx_dir / "model.onnx"

    started = time.monotonic()
    # Tracing is the slow, memory-hungry step and every dtype is derived from its
    # output, so an existing fp32 graph is reused. That makes re-quantizing at a
    # different precision cheap instead of a full reload-and-retrace.
    if fp32.exists():
        _log(f"{spec.name}: reusing existing fp32 graph ({_mb(fp32):.0f} MB)")
    else:
        _log(f"{spec.name}: loading {spec.repo}")
        model, tokenizer = _load(spec)

        _log(f"{spec.name}: writing config + tokenizer")
        write_repo_metadata(spec, model, tokenizer, out_dir)

        _log(f"{spec.name}: tracing to ONNX (fp32)")
        export_fp32(spec, model, tokenizer, fp32)
        _log(f"{spec.name}: fp32 = {_mb(fp32):.0f} MB")
        del model

    verify(fp32)
    if "q4" in dtypes or "q4f16" in dtypes:
        _log(f"{spec.name}: quantizing q4")
        quantize_q4(fp32, onnx_dir / "model_q4.onnx")
        verify(onnx_dir / "model_q4.onnx", fp32)
        _log(f"{spec.name}: q4 = {_mb(onnx_dir / 'model_q4.onnx'):.0f} MB")
    if "q4f16" in dtypes:
        _log(f"{spec.name}: folding q4 -> fp16")
        to_fp16(onnx_dir / "model_q4.onnx", onnx_dir / "model_q4f16.onnx")
        verify(onnx_dir / "model_q4f16.onnx", fp32)
        _log(f"{spec.name}: q4f16 = {_mb(onnx_dir / 'model_q4f16.onnx'):.0f} MB")
    if "q8" in dtypes:
        _log(f"{spec.name}: quantizing q8")
        quantize_q8(fp32, onnx_dir / DTYPE_FILES["q8"])
        verify(onnx_dir / DTYPE_FILES["q8"], fp32)
        _log(f"{spec.name}: q8 = {_mb(onnx_dir / DTYPE_FILES['q8']):.0f} MB")
    if "fp16" in dtypes:
        _log(f"{spec.name}: converting fp16")
        to_fp16(fp32, onnx_dir / "model_fp16.onnx")
        verify(onnx_dir / "model_fp16.onnx", fp32)
        _log(f"{spec.name}: fp16 = {_mb(onnx_dir / 'model_fp16.onnx'):.0f} MB")
    if "q4" not in dtypes and (onnx_dir / "model_q4.onnx").exists():
        (onnx_dir / "model_q4.onnx").unlink()
    if not keep_fp32 and "fp32" not in dtypes:
        for leftover in onnx_dir.glob("model.onnx*"):
            leftover.unlink()

    _log(f"{spec.name}: done in {time.monotonic() - started:.0f}s -> {out_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lfm_export", description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("models"), help="directory to write model repos into"
    )
    parser.add_argument(
        "--only", action="append", choices=sorted(BY_NAME), help="export just this checkpoint"
    )
    parser.add_argument(
        "--dtypes",
        default=",".join(DEFAULT_DTYPES),
        help=(
            f"comma-separated subset of {','.join(DTYPES)} "
            f"(default: {','.join(DEFAULT_DTYPES)}; q4f16/fp16 are not viable here — see README)"
        ),
    )
    parser.add_argument(
        "--keep-fp32",
        action="store_true",
        help="keep the fp32 graph after quantizing (needed by the parity harness)",
    )
    parser.add_argument("--force", action="store_true", help="re-export even if the repo exists")
    args = parser.parse_args(argv)

    dtypes = tuple(d.strip() for d in args.dtypes.split(",") if d.strip())
    unknown = set(dtypes) - set(DTYPES)
    if unknown:
        parser.error(f"unknown dtype(s): {', '.join(sorted(unknown))}")

    specs = [BY_NAME[n] for n in args.only] if args.only else list(SPECS)
    for spec in specs:
        out_dir = args.out / spec.hub_id
        if (out_dir / "onnx").is_dir() and any((out_dir / "onnx").glob("model_*.onnx")):
            if not args.force:
                _log(f"{spec.name}: exists, skipping (use --force to re-export)")
                continue
            shutil.rmtree(out_dir)
        run(spec, args.out, dtypes, args.keep_fp32)
    return 0


if __name__ == "__main__":
    sys.exit(main())
