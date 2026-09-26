"""Merge the LoRA adapter, convert to GGUF and build an Ollama model.

    python -m ml.llm.export_gguf --adapter /kaggle/working/kd-qwen/adapter --base Qwen/Qwen2.5-3B-Instruct \
        --out models/llm --quant q4_k_m q8_0 --llama-cpp /path/to/llama.cpp
    ollama create krishidisha -f models/llm/Modelfile.qwen

Steps: load the base model in fp16 + adapter -> ``merge_and_unload`` -> save HF folder -> llama.cpp
``convert_hf_to_gguf.py`` -> ``llama-quantize`` for each quant -> write ``Modelfile.<family>`` next to the
GGUF (the templates in ``models/llm/Modelfile.*`` are copied with the FROM line filled in).

Run this on the Kaggle/Colab machine right after training (needs ~8 GB RAM for a 3-4B fp16 merge) or
locally on the laptop with ``--cpu`` (slow but works: merging is a one-off).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MODELFILE_DIR = Path(__file__).resolve().parents[2] / "models" / "llm"


def merge(adapter: Path, base: str, out_dir: Path, cpu: bool = False) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_map = {"": "cpu"} if cpu else "auto"
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map=device_map)
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    merged = out_dir / "merged-fp16"
    merged.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(adapter) if (adapter / "tokenizer_config.json").exists() else base).save_pretrained(str(merged))
    print(f"merged model saved to {merged}")
    return merged


def to_gguf(merged: Path, out_dir: Path, name: str, quants: list[str], llama_cpp: Path) -> list[Path]:
    convert = llama_cpp / "convert_hf_to_gguf.py"
    if not convert.exists():
        raise SystemExit(f"{convert} not found; clone https://github.com/ggerganov/llama.cpp and build llama-quantize")
    f16 = out_dir / f"{name}-f16.gguf"
    subprocess.run([sys.executable, str(convert), str(merged), "--outfile", str(f16), "--outtype", "f16"], check=True)
    quantize = next((p for p in [llama_cpp / "build" / "bin" / "llama-quantize", llama_cpp / "build" / "bin" / "Release" / "llama-quantize.exe",
                                 llama_cpp / "llama-quantize", llama_cpp / "llama-quantize.exe"] if p.exists()), None)
    outputs = []
    for q in quants:
        target = out_dir / f"{name}-{q}.gguf"
        if quantize is None:
            print("llama-quantize binary not found; keeping f16 only")
            break
        subprocess.run([str(quantize), str(f16), str(target), q.upper()], check=True)
        outputs.append(target)
        print(f"wrote {target} ({target.stat().st_size / 1e9:.2f} GB)")
    return outputs or [f16]


def write_modelfile(family: str, gguf: Path, out_dir: Path) -> Path:
    template = MODELFILE_DIR / f"Modelfile.{family}"
    text = template.read_text(encoding="utf-8")
    text = text.replace("FROM ./MODEL.gguf", f"FROM ./{gguf.name}")
    target = out_dir / f"Modelfile.{family}"
    target.write_text(text, encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--base", required=True, help="HF id of the *unquantized* base, e.g. Qwen/Qwen2.5-3B-Instruct or google/gemma-3-4b-it")
    p.add_argument("--out", type=Path, default=Path("models/llm"))
    p.add_argument("--name", default="krishidisha")
    p.add_argument("--quant", nargs="*", default=["q4_k_m"])
    p.add_argument("--llama-cpp", type=Path, default=Path("llama.cpp"))
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--skip-merge", action="store_true", help="reuse <out>/merged-fp16")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    merged = args.out / "merged-fp16" if args.skip_merge else merge(args.adapter, args.base, args.out, args.cpu)
    ggufs = to_gguf(merged, args.out, args.name, args.quant, args.llama_cpp)
    family = "gemma" if "gemma" in args.base.lower() else "qwen"
    mf = write_modelfile(family, ggufs[0], args.out)
    print(f"Modelfile: {mf}\nnext: ollama create {args.name} -f {mf}")
    if not args.skip_merge:
        shutil.rmtree(merged, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
