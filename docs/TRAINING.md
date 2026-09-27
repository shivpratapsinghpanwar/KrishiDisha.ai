# Training the KrishiDisha models (lab repository only)

This file and the `ml/` training modules, `notebooks/` and the pipeline tests live only in the private lab repository `shivpratapsinghpanwar/KrishiDisha-ml`. The public repository carries the app, the runtime subset of `ml/` (estimators, registry, taxonomy, model factory, tool format), metrics, reports and model cards.

```bash
python -m ml.train_tabular --data-dir data --output models        # compares RF / GB / XGBoost / SVM / kNN / NB ...
python -m ml.train_all                                            # tabular + disease (image stage skipped if dataset absent)
```

Disease model — trained only on field photographs (see [Results](#results) above):

```bash
python -m ml datasets download --all            # Kaggle/GitHub sources listed in ml/datasets/sources.yaml
python -m ml build disease-manifest --resize 320 # unified manifest, dedup, per-source splits
python -m ml train disease --manifest <DATA_ROOT>/disease_unified/manifest.csv --arch timm:efficientnet_b0 --ema --calibrate
python -m ml eval disease --manifest ...        # per-crop report, coverage tiers, model card
python -m ml export disease                     # ONNX for the app + models/registry.json
```

Larger backbones train in `notebooks/kaggle_train_disease/`; the RTX 2050 handles
`efficientnet_b0` locally. Fallbacks without a trained model: `DISEASE_MODEL_BACKEND=hf`
downloads a pretrained PlantVillage MobileNet from the Hugging Face Hub (lab data, demo only);
the original `plant_disease_model_1_latest.pt` from `CNN.py` is also supported (`legacy`).

## Kaggle

Kernels cannot clone a private GitHub repo without a token, so they read the source from the private Kaggle dataset `sspanwar/krishidisha-lab-src`, refreshed with `python scripts/kaggle_sync_src.py` before a run. Notebooks: `notebooks/kaggle_train_disease/`, `notebooks/kaggle_train_llm/`, `notebooks/kaggle_train_tabular/`.

## Publishing the public mirror

`python scripts/publish_public.py` clones this repo, strips the training paths from the whole history with git-filter-repo and force-pushes the result to the public `KrishiDisha.ai` main branch.
