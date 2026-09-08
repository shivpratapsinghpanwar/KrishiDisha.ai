"""Export the trained checkpoint to ONNX for the web app and register it.

    python -m ml.vision.export --checkpoint models/plant_disease_model.pt --output models

Writes ``plant_disease_model.onnx`` (opset 17, dynamic batch), ``plant_disease_classes.json`` and
``plant_disease_model.meta.json`` (img_size, mean, std, crop_pct, temperature, thresholds, tiers), checks
ONNX Runtime output against PyTorch on random tensors, then appends a ``models/registry.json`` entry.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from ..registry import register
from .model import load_checkpoint


def export_onnx(ckpt_path: Path, output: Path, check: bool = True) -> Path:
    model, ckpt = load_checkpoint(ckpt_path)
    model.eval()
    size = int(ckpt["img_size"])
    dummy = torch.randn(1, 3, size, size)
    onnx_path = output / "plant_disease_model.onnx"
    torch.onnx.export(model, dummy, str(onnx_path), opset_version=17, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, do_constant_folding=True,
                      dynamo=False)
    meta = {"arch": ckpt["arch"], "img_size": size, "mean": ckpt["mean"], "std": ckpt["std"], "crop_pct": ckpt["crop_pct"],
            "classes": ckpt["classes"], "temperature": ckpt.get("temperature", 1.0),
            "ood_threshold": ckpt.get("ood_threshold"), "uncertain_threshold": ckpt.get("uncertain_threshold", 0.45),
            "crop_tiers": ckpt.get("crop_tiers", {}), "val_accuracy": ckpt.get("val_accuracy"),
            "metrics_summary": ckpt.get("metrics_summary"), "trained_at": ckpt.get("trained_at"),
            "sources": ckpt.get("sources", [])}
    (output / "plant_disease_model.meta.json").write_text(json.dumps(meta, indent=1))
    (output / "plant_disease_classes.json").write_text(json.dumps(ckpt["classes"], indent=1))
    if check:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        x = torch.randn(4, 3, size, size)
        with torch.no_grad():
            ref = model(x).numpy()
        out = sess.run(None, {"input": x.numpy()})[0]
        diff = float(np.abs(ref - out).max())
        print(f"onnx parity: max abs diff {diff:.2e}")
        if diff > 1e-2:
            raise RuntimeError(f"ONNX output differs from PyTorch by {diff}")
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    return onnx_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=Path("models/plant_disease_model.pt"))
    p.add_argument("--output", type=Path, default=Path("models"))
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--no-register", action="store_true")
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    onnx_path = export_onnx(args.checkpoint, args.output, check=not args.no_check)
    if not args.no_register:
        _, ckpt = load_checkpoint(args.checkpoint)
        register("plant_disease", args.output / "plant_disease_model.onnx", arch=ckpt["arch"],
                 metrics={"val_top1": ckpt.get("val_accuracy"), **(ckpt.get("metrics_summary") or {})},
                 extra={"classes": len(ckpt["classes"]), "img_size": ckpt["img_size"],
                        "sources": ckpt.get("sources", [])}, card="reports/plant_disease_card.md",
                 registry_path=args.output / "registry.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
