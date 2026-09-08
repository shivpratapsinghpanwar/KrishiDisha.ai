"""Plant disease detection from leaf images.

Backends (tried in order when DISEASE_MODEL_BACKEND=auto):

1. ``local``  - a checkpoint trained with ``ml/train_disease.py``
                (``models/plant_disease_model.pt`` + ``models/plant_disease_classes.json``).
2. ``hf``     - a pretrained PlantVillage classifier pulled from the Hugging Face Hub
                (works out of the box with an internet connection).
3. ``legacy`` - the original 39-class ``plant_disease_model_1_latest.pt`` from CNN.py.
4. ``stub``   - no model; returns a clear explanation instead of crashing.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from PIL import Image

log = logging.getLogger(__name__)

# Canonical PlantVillage label -> (crop, condition) split helper.
def pretty_label(raw: str) -> str:
    raw = raw.replace("___", " : ").replace("__", " : ").replace("_", " ")
    return " ".join(raw.split())


_KNOWN_CROPS = ("apple", "tomato", "potato", "corn", "maize", "grape", "peach", "pepper", "cherry", "strawberry",
                "orange", "squash", "soybean", "raspberry", "blueberry", "rice", "wheat", "cotton", "sugarcane")


def split_label(raw: str) -> tuple[str, str]:
    """Split a class name into (crop, condition) for the label styles seen in the wild:
    ``Tomato___Late_blight`` (PlantVillage), ``Tomato : Late blight``, ``Tomato with Late Blight``,
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


class DiseaseDetector:
    def __init__(self, models_dir: Path, backend: str = "auto", hf_model: str | None = None):
        self.models_dir = Path(models_dir)
        self.backend_pref = backend
        self.hf_model = hf_model
        self._lock = threading.Lock()
        self._loaded = False
        self.backend = "stub"
        self.classes: list[str] = []
        self._predict_fn = None
        self.model_name = "none"

    # ----------------------------------------------------------------- load
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            order = {
                "auto": ["local", "hf", "legacy"],
                "local": ["local"], "hf": ["hf"], "legacy": ["legacy"], "stub": [],
            }.get(self.backend_pref, ["local", "hf", "legacy"])
            for name in order:
                try:
                    getattr(self, f"_load_{name}")()
                    self.backend = name
                    log.info("Disease detector backend: %s (%s)", name, self.model_name)
                    break
                except Exception as exc:
                    log.warning("Disease backend %s unavailable: %s", name, exc)
            self._loaded = True

    def _load_local(self) -> None:
        import torch
        from torchvision import transforms

        ckpt_path = self.models_dir / "plant_disease_model.pt"
        classes_path = self.models_dir / "plant_disease_classes.json"
        if not ckpt_path.exists() or not classes_path.exists():
            raise FileNotFoundError("local checkpoint not found")
        from ml.disease_model import build_model  # noqa: WPS433 - local import to keep torch optional

        self.classes = json.loads(classes_path.read_text())
        ckpt = torch.load(ckpt_path, map_location="cpu")
        arch = ckpt.get("arch", "mobilenet_v3_large")
        model = build_model(arch, len(self.classes), pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        def predict(img: Image.Image):
            with torch.no_grad():
                x = tf(img.convert("RGB")).unsqueeze(0)
                probs = torch.softmax(model(x), dim=1)[0]
            return probs.tolist()

        self._predict_fn = predict
        self.model_name = f"{arch} (KrishiDisha fine-tuned, {ckpt.get('val_accuracy', '?')} val acc)"

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
                logits = model(**inputs).logits
            return torch.softmax(logits, dim=1)[0].tolist()

        self._predict_fn = predict
        self.model_name = f"huggingface:{name}"

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
        # Last resort: standard ImageNet preprocessing that suits nearly every classifier on the Hub.
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
                probs = torch.softmax(model(x), dim=1)[0]
            return probs.tolist()

        self._predict_fn = predict
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
                "message": "No disease detection model is loaded. Train one with `python -m ml.train_disease` "
                           "or set DISEASE_MODEL_BACKEND=hf with internet access.",
                "predictions": [],
            }
        probs = self._predict_fn(image)
        order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_k]
        preds = []
        for i in order:
            raw = self.classes[i]
            crop, cond = split_label(raw)
            preds.append({
                "label": raw,
                "name": pretty_label(raw),
                "crop": crop,
                "condition": cond or "healthy",
                "is_healthy": "healthy" in raw.lower(),
                "confidence": round(float(probs[i]), 4),
            })
        return {
            "available": True,
            "backend": self.backend,
            "model": self.model_name,
            "predictions": preds,
            "top": preds[0],
        }

    def info(self) -> dict[str, Any]:
        self._ensure_loaded()
        return {"backend": self.backend, "model": self.model_name, "num_classes": len(self.classes),
                "available": self._predict_fn is not None}
