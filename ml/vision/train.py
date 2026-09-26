"""Train the KrishiDisha field disease classifier from ``manifest.csv``.

    python -m ml.vision.train --manifest <DATA_ROOT>/disease_unified/manifest.csv --arch timm:efficientnet_b0 \
        --img-size 224 --batch-size 32 --epochs 12 --ema --output models
    python -m ml.vision.train ... --limit-per-class 40 --epochs 1            # 5-minute smoke test
    python -m ml.vision.train ... --init models/plant_disease_model.pt --lr 1e-4 --epochs 4 --own-weight 6   # stage B

Recipe: AdamW + OneCycle, Mixup/CutMix with label smoothing, RandAugment, EMA weights, class/source
balanced sampler, AMP + channels_last on CUDA, resume from ``<output>/checkpoints/last.pt`` every epoch.
Writes ``plant_disease_model.pt`` (format 2, all metadata the app needs), ``plant_disease_classes.json``,
``disease_training_log.json`` and hands over to :mod:`ml.vision.calibrate` when ``--calibrate`` is set.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import ManifestDataset, MixupCutmix, balanced_sampler, read_manifest, soft_cross_entropy
from .model import ModelEma, build_model, checkpoint_payload, get_transforms, resolve_preprocessing


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(pref: str) -> torch.device:
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool):
    model.eval()
    logits_all, y_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device.type, enabled=use_amp):
            out = model(x)
        logits_all.append(out.float().cpu())
        y_all.append(y)
    logits = torch.cat(logits_all)
    y = torch.cat(y_all)
    top1 = (logits.argmax(1) == y).float().mean().item()
    top3 = (logits.topk(3, dim=1).indices == y[:, None]).any(1).float().mean().item()
    loss = nn.functional.cross_entropy(logits, y).item()
    return {"loss": loss, "top1": top1, "top3": top3}, logits, y


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--arch", default="timm:efficientnet_b0")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.02)
    p.add_argument("--warmup-pct", type=float, default=0.15)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mix-prob", type=float, default=0.5, help="Mixup/CutMix probability (0 disables)")
    p.add_argument("--no-randaug", action="store_true")
    p.add_argument("--ema", action="store_true", help="track an EMA of the weights and save it if it evaluates better")
    p.add_argument("--field-weight", type=float, default=1.0)
    p.add_argument("--own-weight", type=float, default=3.0)
    p.add_argument("--max-class-share", type=float, default=0.03)
    p.add_argument("--max-source-share", type=float, default=0.25)
    p.add_argument("--limit-per-class", type=int, default=0)
    p.add_argument("--exclude-sources", nargs="*", default=[], help="sources to leave out of training")
    p.add_argument("--init", type=Path, help="initialise from a previous checkpoint (stage B)")
    p.add_argument("--resume", action="store_true", help="resume from <output>/checkpoints/last.pt")
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output", type=Path, default=Path("models"))
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calibrate", action="store_true", help="run temperature scaling + thresholds after training")
    args = p.parse_args(argv)

    seed_all(args.seed)
    device = pick_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    workers = args.num_workers if device.type == "cuda" else min(args.num_workers, 2)
    args.output.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ data
    rows = read_manifest(args.manifest)
    root = args.manifest.parent
    rows = [r for r in rows if r["source"] not in set(args.exclude_sources)]
    train_rows = [r for r in rows if r["split"] == "train"]
    valid_rows = [r for r in rows if r["split"] == "valid"]
    classes = sorted({r["label"] for r in train_rows})
    prep = resolve_preprocessing(args.arch, args.img_size)
    prep["img_size"] = args.img_size
    train_ds = ManifestDataset(train_rows, root, classes,
                               get_transforms(True, args.img_size, prep["mean"], prep["std"], randaug=not args.no_randaug),
                               limit_per_class=args.limit_per_class, seed=args.seed)
    valid_ds = ManifestDataset(valid_rows, root, classes,
                               get_transforms(False, args.img_size, prep["mean"], prep["std"], prep["crop_pct"]),
                               limit_per_class=max(args.limit_per_class // 2, 4) if args.limit_per_class else 0,
                               seed=args.seed)
    sampler = balanced_sampler(train_ds, args.field_weight, args.own_weight, args.max_class_share,
                               args.max_source_share, args.seed)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=workers,
                              pin_memory=pin, drop_last=len(train_ds) > args.batch_size, persistent_workers=workers > 0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=workers,
                              pin_memory=pin, persistent_workers=workers > 0)
    print(f"device={device} amp={use_amp} classes={len(classes)} train={len(train_ds)} valid={len(valid_ds)} "
          f"arch={args.arch} img={args.img_size} bs={args.batch_size}x{args.accum}")
    (args.output / "plant_disease_classes.json").write_text(json.dumps(classes, indent=1))

    # ----------------------------------------------------------------- model
    model = build_model(args.arch, len(classes), pretrained=not args.no_pretrained)
    if args.init:
        init = torch.load(args.init, map_location="cpu", weights_only=False)
        state = init["state_dict"]
        if init.get("classes") != classes:  # class list changed: drop the head
            head_keys = [k for k in state if state[k].shape[0] == len(init.get("classes", [])) and k.split(".")[-1] in ("weight", "bias")]
            for k in head_keys:
                state.pop(k)
            print(f"init: class list differs; dropped {len(head_keys)} head tensors")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"init from {args.init}: missing={len(missing)} unexpected={len(unexpected)}")
    if args.grad_checkpoint and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
    model = model.to(device).to(memory_format=torch.channels_last)
    ema = ModelEma(model) if args.ema else None

    decay, no_decay = [], []
    for n, prm in model.named_parameters():
        (no_decay if prm.ndim <= 1 or n.endswith(".bias") else decay).append(prm)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    steps_per_epoch = max(1, len(train_loader) // args.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
                                                pct_start=args.warmup_pct, div_factor=20, final_div_factor=200)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    mixer = MixupCutmix(len(classes), prob=args.mix_prob, label_smoothing=args.label_smoothing)

    start_epoch, best_top1, history = 0, -1.0, []
    last_path = ckpt_dir / "last.pt"
    if args.resume and last_path.exists():
        st = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        if ema and st.get("ema"):
            ema.module.load_state_dict(st["ema"])
        start_epoch, best_top1, history = st["epoch"] + 1, st["best_top1"], st["history"]
        print(f"resumed from epoch {start_epoch} (best top1 {best_top1:.4f})")

    # ------------------------------------------------------------------ loop
    log_path = args.output / "disease_training_log.json"
    best_path = args.output / "plant_disease_model.pt"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        run_loss, n_batches = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            x, target = mixer(x, y)
            with torch.autocast(device.type, enabled=use_amp):
                logits = model(x)
                loss = soft_cross_entropy(logits.float(), target) / args.accum
            scaler.scale(loss).backward()
            if (i + 1) % args.accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if sched.last_epoch < args.epochs * steps_per_epoch - 1:
                    sched.step()
                if ema:
                    ema.update(model)
            run_loss += loss.item() * args.accum
            n_batches += 1
            if n_batches % 100 == 0:
                print(f"  ep{epoch + 1} it{n_batches}/{len(train_loader)} loss {run_loss / n_batches:.4f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} {time.time() - t0:.0f}s", flush=True)
        train_loss = run_loss / max(n_batches, 1)

        val, _, _ = evaluate(model, valid_loader, device, use_amp)
        val_ema = evaluate(ema.module.to(device), valid_loader, device, use_amp)[0] if ema else None
        use_ema = bool(val_ema and val_ema["top1"] >= val["top1"])
        chosen = val_ema if use_ema else val
        rec = {"epoch": epoch + 1, "train_loss": round(train_loss, 4), "val_loss": round(val["loss"], 4),
               "val_top1": round(val["top1"], 4), "val_top3": round(val["top3"], 4),
               "ema_top1": round(val_ema["top1"], 4) if val_ema else None, "seconds": round(time.time() - t0)}
        history.append(rec)
        print(f"epoch {epoch + 1}/{args.epochs}: train {train_loss:.4f} | val top1 {val['top1']:.4f} top3 {val['top3']:.4f}"
              + (f" | ema top1 {val_ema['top1']:.4f}" if val_ema else "") + f" | {rec['seconds']}s", flush=True)
        log_path.write_text(json.dumps({"args": vars(args) | {"manifest": str(args.manifest), "output": str(args.output),
                                                              "init": str(args.init) if args.init else None},
                                        "classes": classes, "history": history}, indent=1, default=str))

        if chosen["top1"] > best_top1:
            best_top1 = chosen["top1"]
            src = ema.module if use_ema else model
            payload = checkpoint_payload(src, args.arch, classes, prep, val_accuracy=round(best_top1, 4),
                                         val_top3=round(chosen["top3"], 4), epoch=epoch + 1, ema=use_ema,
                                         manifest=str(args.manifest), trained_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                         train_images=len(train_ds), sources=sorted({r["source"] for r in train_rows}))
            torch.save(payload, best_path)
            print(f"  saved {best_path} (top1 {best_top1:.4f}{' ema' if use_ema else ''})")
        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "ema": ema.module.state_dict() if ema else None,
                    "best_top1": best_top1, "history": history}, last_path)

    print(f"best val top1 {best_top1:.4f}; checkpoint {best_path}")
    if args.calibrate and best_path.exists():
        from .calibrate import calibrate_checkpoint

        calibrate_checkpoint(best_path, args.manifest, device=device, batch_size=args.batch_size * 2, workers=workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
