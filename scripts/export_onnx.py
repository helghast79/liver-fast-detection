#!/usr/bin/env python3
"""Export a TinyUNet checkpoint to ONNX (+ optional INT8 dynamic quantization)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

import numpy as np
import torch

from src.model import TinyUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "liver_unet_128.onnx")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument(
        "--in-channels",
        type=int,
        default=None,
        help="Input channels (default: read from checkpoint, else 1)",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--quantize", action="store_true", default=True)
    parser.add_argument("--no-quantize", action="store_false", dest="quantize")
    parser.add_argument("--smoke-test", action="store_true", default=True)
    parser.add_argument("--no-smoke-test", action="store_false", dest="smoke_test")
    return parser.parse_args()


def export_onnx(
    checkpoint: Path,
    output: Path,
    image_size: int,
    base_channels: int,
    opset: int,
    in_channels: int | None = None,
) -> Path:
    device = torch.device("cpu")
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        if state.get("in_channels") is not None:
            ckpt_in = int(state["in_channels"])
        elif state.get("context_slices") is not None:
            ckpt_in = 2 * int(state["context_slices"]) + 1
        else:
            ckpt_in = 1
        weights = state["model_state_dict"]
        base_channels = int(state.get("base_channels", base_channels))
        image_size = int(state.get("image_size", image_size))
    else:
        ckpt_in = 1
        weights = state
    in_ch = int(in_channels if in_channels is not None else ckpt_in)
    model = TinyUNet(in_channels=in_ch, out_channels=1, base=base_channels).to(device)
    model.load_state_dict(weights)
    model.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, in_ch, image_size, image_size, device=device)
    export_kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    try:
        # Prefer legacy exporter for broad onnxruntime-web compatibility.
        torch.onnx.export(
            model,
            dummy,
            str(output),
            dynamo=False,
            **export_kwargs,
        )
    except TypeError:
        torch.onnx.export(model, dummy, str(output), **export_kwargs)
    return output


def quantize_onnx(onnx_path: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quant_path = onnx_path.with_name(onnx_path.stem + ".quant.onnx")
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(quant_path),
        weight_type=QuantType.QUInt8,
    )
    return quant_path


def smoke_test(onnx_path: Path, image_size: int, in_channels: int = 1, runs: int = 20) -> dict:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.randn(1, in_channels, image_size, image_size).astype(np.float32)
    # warmup
    for _ in range(3):
        session.run(None, {"input": x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {"input": x})
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "onnx_path": str(onnx_path),
        "size_bytes": onnx_path.stat().st_size,
        "mean_ms": float(np.mean(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
    }


def main() -> int:
    args = parse_args()
    state = torch.load(args.checkpoint, map_location="cpu")
    if args.in_channels is not None:
        in_ch = int(args.in_channels)
    elif isinstance(state, dict) and state.get("in_channels") is not None:
        in_ch = int(state["in_channels"])
    elif isinstance(state, dict) and state.get("context_slices") is not None:
        in_ch = 2 * int(state["context_slices"]) + 1
    else:
        in_ch = 1
    onnx_path = export_onnx(
        args.checkpoint,
        args.output,
        args.image_size,
        args.base_channels,
        args.opset,
        in_channels=in_ch,
    )
    print(f"Exported ONNX: {onnx_path} ({onnx_path.stat().st_size} bytes, in_channels={in_ch})")

    results = []
    if args.smoke_test:
        results.append(smoke_test(onnx_path, args.image_size, in_channels=in_ch))

    if args.quantize:
        quant_path = quantize_onnx(onnx_path)
        print(f"Quantized ONNX: {quant_path} ({quant_path.stat().st_size} bytes)")
        if args.smoke_test:
            results.append(smoke_test(quant_path, args.image_size, in_channels=in_ch))

    for r in results:
        print(
            f"smoke {Path(r['onnx_path']).name}: "
            f"{r['size_bytes']} bytes, mean={r['mean_ms']:.2f}ms "
            f"p50={r['p50_ms']:.2f}ms p95={r['p95_ms']:.2f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
