"""Plant disease detection from leaf images.

Backends (tried in order when DISEASE_MODEL_BACKEND=auto):

1. ``onnx``   - ``models/plant_disease_model.onnx`` + ``plant_disease_model.meta.json`` exported by
                ``ml/vision/export.py`` (KrishiDisha's own field model; no torch needed in the web process).
2. ``local``  - ``models/plant_disease_model.pt`` checkpoint from ``ml/vision/train.py`` (format 2) or the
                older ``ml/train_disease.py`` (format 1) + ``plant_disease_classes.json``.
3. ``hf``     - a pretrained PlantVillage classifier from the Hugging Face Hub (lab data; demo fallback only).
4. ``legacy`` - the original 39-class ``plant_disease_model_1_latest.pt`` from CNN.py.
5. ``stub``   - no model; returns a clear explanation instead of crashing.

Every prediction carries ``uncertain`` (calibrated confidence below the checkpoint's threshold),
``is_plant`` (top class is not ``Other___not_a_leaf`` and confidence above the OOD threshold) and
``crop_tier`` (A/B/C coverage tier from the evaluation report) so the UI and the assistant can hedge.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

NOT_A_LEAF = "Other___not_a_leaf"
DEFAULT_UNCERTAIN = 0.45
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def pretty_label(raw: str) -> str:
    raw = raw.replace("___", " : ").replace("__", " : ").replace("_", " ")
    return " ".join(raw.split())


_KNOWN_CROPS = ("apple", "tomato", "potato", "corn", "maize", "grape", "peach", "pepper", "cherry", "strawberry",
                "orange", "squash", "soybean", "raspberry", "blueberry", "rice", "wheat", "cotton", "sugarcane",
                "mango", "banana", "citrus", "chilli", "groundnut", "cashew", "cassava", "other")


def split_label(raw: str) -> tuple[str, str]:
    """Split a class name into (crop, condition) for the label styles seen in the wild:
    ``Tomato___Late_blight`` (PlantVillage / KrishiDisha), ``Tomato : Late blight``, ``Tomato with Late Blight``,
    ``Healthy Tomato Plant`` / ``Tomato healthy`` and free text such as ``Cedar Apple Rust``."""
    s = raw.strip()
    low = s.lower()
    if "___" in s:
        crop, cond = s.split("___", 1)
    elif " : " in s:
        crop, cond = s.split(" : ", 1)
    elif " with " in low:
        i = low.index(" with ")
        crop, cond = s[:i], s[i + 6:]
    elif low.startswith("healthy "):
        crop, cond = s[8:], "healthy"
        crop = crop[:-6].strip() if crop.lower().endswith(" plant") else crop
    elif low.endswith(" healthy") or low.endswith("_healthy"):
        crop, cond = s[:-8], "healthy"
    else:
        words = s.replace("(", " ").replace(")", " ").split()
        hit = next((w for w in words if w.lower().strip(",") in _KNOWN_CROPS), None)
        crop, cond = (hit or words[0]), s
    crop = crop.replace("_", " ").replace(",", "").strip()
    cond = cond.replace("_", " ").strip()
    return crop, cond


def _preprocess(img: Image.Image, img_size: int, mean, std, crop_pct: float | None) -> np.ndarray:
    """Resize(short side = img_size / crop_pct) -> centre crop, identical to ``ml.vision.model.get_transforms``."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    if crop_pct:
        resize_to = int(round(img_size / crop_pct))
        w, h = img.size
        scale = resize_to / min(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BILINEAR)
        w, h = img.size
        left, top = (w - img_size) // 2, (h - img_size) // 2
        img = img.crop((left, top, left + img_size, top + img_size))
    else:
        img = img.resize((img_size, img_size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


class DiseaseDetector:
    def __init__(self, models_dir: Path, backend: str = "auto", hf_model: str | None = None):
        self.models_dir = Path(models_dir)
        self.backend_pref = backend
        self.hf_model = hf_model
        self._lock = threading.Lock()
        self._loaded = False
        self.backend = "stub"
        self.classes: list[str] = []
        self._predict_fn = None          # PIL image -> logits (np.ndarray) or probabilities
        self._returns_probs = False
        self.model_name = "none"
        self.temperature = 1.0
        self.ood_threshold: float | None = None
        self.uncertain_threshold = DEFAULT_UNCERTAIN
        self.crop_tiers: dict[str, str] = {}
        self.meta: dict[str, Any] = {}

    # ----------------------------------------------------------------- load
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            order = {
                "auto": ["onnx", "local", "hf", "legacy"],
                "onnx": ["onnx"], "local": ["local"], "hf": ["hf"], "legacy": ["legacy"], "stub": [],
            }.get(self.backend_pref, ["onnx", "local", "hf", "legacy"])
            for name in order:
                try:
                    getattr(self, f"_load_{name}")()
                    self.backend = name
                    log.info("Disease detector backend: %s (%s, %d classes)", name, self.model_name, len(self.classes))
                    break
                except Exception as exc:  # noqa: BLE001
                    log.warning("Disease backend %s unavailable: %s", name, exc)
            self._loaded = True

    def _apply_meta(self, meta: dict[str, Any]) -> None:
        self.meta = meta
        self.temperature = float(meta.get("temperature") or 1.0)
        self.ood_threshold = meta.get("ood_threshold")
        self.uncertain_threshold = float(meta.get("uncertain_threshold") or DEFAULT_UNCERTAIN)
        self.crop_tiers = dict(meta.get("crop_tiers") or {})

    def _load_onnx(self) -> None:
        onnx_path = self.models_dir / "plant_disease_model.onnx"
        meta_path = self.models_dir / "plant_disease_model.meta.json"
        if not onnx_path.exists() or not meta_path.exists():
            raise FileNotFoundError("onnx export not found")
        import onnxruntime as ort

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        self.classes = list(meta["classes"])
        self._apply_meta(meta)
        size, mean, std, crop_pct = int(meta["img_size"]), meta.get("mean", IMAGENET_MEAN), meta.get("std", IMAGENET_STD), meta.get("crop_pct")

        def predict(img: Image.Image):
            x = _preprocess(img, size, mean, std, crop_pct)
            return sess.run(None, {input_name: x})[0][0]

        self._predict_fn = predict
        self._returns_probs = False
        acc = meta.get("val_accuracy")
        self.model_name = f"KrishiDisha {meta.get('arch', 'onnx')} (field model, val {acc})" if acc else f"KrishiDisha {meta.get('arch', 'onnx')}"

    def _load_local(self) -> None:
        import torch

        ckpt_path = self.models_dir / "plant_disease_model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError("local checkpoint not found")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt.get("format", 1) >= 2:
            from ml.vision.model import build_model

            self.classes = list(ckpt["classes"])
            arch = ckpt["arch"]
            size, mean, std, crop_pct = int(ckpt["img_size"]), ckpt.get("mean", IMAGENET_MEAN), ckpt.get("std", IMAGENET_STD), ckpt.get("crop_pct")
        else:  # format 1: ml/train_disease.py checkpoints
            from ml.disease_model import build_model

            classes_path = self.models_dir / "plant_disease_classes.json"
            if not classes_path.exists():
                raise FileNotFoundError("plant_disease_classes.json not found")
            self.classes = json.loads(classes_path.read_text())
            arch = ckpt.get("arch", "mobilenet_v3_large")
            size, mean, std, crop_pct = int(ckpt.get("img_size", 224)), IMAGENET_MEAN, IMAGENET_STD, None
        self._apply_meta(ckpt)
        model = build_model(arch, len(self.classes), pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        def predict(img: Image.Image):
            x = torch.from_numpy(_preprocess(img, size, mean, std, crop_pct))
            with torch.no_grad():
                return model(x)[0].numpy()

        self._predict_fn = predict
        self._returns_probs = False
        self.model_name = f"{arch} (KrishiDisha, val {ckpt.get('val_accuracy', '?')})"

    def _load_hf(self) -> None:
        import torch
        from transformers import AutoModelForImageClassification

        name = self.hf_model
        if not name:
            raise ValueError("no HF model configured")
        model = AutoModelForImageClassification.from_pretrained(name)
        model.eval()
        id2label = model.config.id2label
        self.classes = [id2label[i] for i in range(len(id2label))]
        processor = self._hf_processor(name, model.config)

        def predict(img: Image.Image):
            inputs = processor(images=img.convert("RGB"), return_tensors="pt")
            with torch.no_grad():
                return model(**inputs).logits[0].numpy()

        self._predict_fn = predict
        self._returns_probs = False
        self.model_name = f"huggingface:{name} (lab-trained demo model)"

    @staticmethod
    def _hf_processor(name: str, config):
        """Resolve an image processor even for older Hub repos whose preprocessor_config.json
        lacks ``image_processor_type`` (transformers >= 5 refuses to guess)."""
        import transformers
        from transformers import AutoImageProcessor

        try:
            return AutoImageProcessor.from_pretrained(name)
        except Exception as exc:  # noqa: BLE001
            log.info("AutoImageProcessor failed for %s (%s); trying model-specific processor", name, exc)
        by_type = {
            "mobilenet_v2": "MobileNetV2ImageProcessor", "mobilenet_v1": "MobileNetV1ImageProcessor",
            "vit": "ViTImageProcessor", "deit": "DeiTImageProcessor", "convnext": "ConvNextImageProcessor",
            "resnet": "ConvNextImageProcessor", "efficientnet": "EfficientNetImageProcessor",
            "beit": "BeitImageProcessor", "swin": "ViTImageProcessor", "mobilevit": "MobileViTImageProcessor",
        }
        cls_name = by_type.get(getattr(config, "model_type", ""), "ViTImageProcessor")
        cls = getattr(transformers, cls_name, None)
        if cls is not None:
            try:
                return cls.from_pretrained(name)
            except Exception as exc:  # noqa: BLE001
                log.info("%s failed for %s (%s); using a plain torchvision transform", cls_name, name, exc)
        import torch
        from torchvision import transforms

        size = getattr(config, "image_size", 224) or 224
        tf = transforms.Compose([
            transforms.Resize(int(size * 256 / 224)), transforms.CenterCrop(size), transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        def processor(images, return_tensors="pt"):
            return {"pixel_values": tf(images).unsqueeze(0)}
        return processor

    def _load_legacy(self) -> None:
        import torch
        import torchvision.transforms.functional as TF

        import CNN  # original architecture shipped with the project

        path = self.models_dir / "plant_disease_model_1_latest.pt"
        if not path.exists():
            raise FileNotFoundError("legacy checkpoint not found")
        model = CNN.CNN(39)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        self.classes = [CNN.idx_to_classes[i] for i in range(39)]

        def predict(img: Image.Image):
            x = TF.to_tensor(img.convert("RGB").resize((224, 224))).unsqueeze(0)
            with torch.no_grad():
                return model(x)[0].numpy()

        self._predict_fn = predict
        self._returns_probs = False
        self.model_name = "legacy 4-block CNN (39 classes)"

    # -------------------------------------------------------------- predict
    def predict(self, image: Image.Image | str | Path, top_k: int = 3) -> dict[str, Any]:
        self._ensure_loaded()
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if self._predict_fn is None:
            return {
                "available": False,
                "backend": "stub",
                "message": "No disease detection model is loaded. Train one with `python -m ml.vision.train` and export "
                           "it with `python -m ml.vision.export`, or set DISEASE_MODEL_BACKEND=hf with internet access.",
                "predictions": [],
            }
        out = np.asarray(self._predict_fn(image), dtype=np.float64).reshape(-1)
        probs = out if self._returns_probs else _softmax(out / self.temperature)
        order = np.argsort(-probs)[:top_k]
        preds = []
        for i in order:
            raw = self.classes[int(i)]
            crop, cond = split_label(raw)
            preds.append({
                "label": raw,
                "name": pretty_label(raw),
                "crop": crop,
                "condition": cond or "healthy",
                "is_healthy": "healthy" in raw.lower(),
                "confidence": round(float(probs[i]), 4),
                "crop_tier": self.crop_tiers.get(crop.replace(" ", "_"), self.crop_tiers.get(crop)),
            })
        top = preds[0]
        conf = top["confidence"]
        not_leaf = top["label"] == NOT_A_LEAF
        below_ood = self.ood_threshold is not None and conf < float(self.ood_threshold)
        is_plant = not not_leaf and not below_ood
        uncertain = (not is_plant) or conf < self.uncertain_threshold
        if not_leaf:
            message = "This does not look like a plant leaf. Retake the photo with one leaf filling the frame."
        elif below_ood:
            message = "The photo is unclear for the model. Retake in daylight with a single leaf on a plain background."
        elif uncertain:
            message = "Low confidence. Treat this as a hint and confirm with your KVK or an agronomist."
        elif top.get("crop_tier") == "C":
            message = "This crop has limited training data (experimental). Confirm the diagnosis with an expert."
        else:
            message = None
        return {
            "available": True,
            "backend": self.backend,
            "model": self.model_name,
            "predictions": preds,
            "top": top,
            "uncertain": bool(uncertain),
            "is_plant": bool(is_plant),
            "crop_tier": top.get("crop_tier"),
            "message": message,
        }

    def info(self) -> dict[str, Any]:
        self._ensure_loaded()
        return {"backend": self.backend, "model": self.model_name, "num_classes": len(self.classes),
                "available": self._predict_fn is not None, "temperature": self.temperature,
                "ood_threshold": self.ood_threshold, "uncertain_threshold": self.uncertain_threshold,
                "crop_tiers": self.crop_tiers, "field_model": self.backend in ("onnx", "local")}
