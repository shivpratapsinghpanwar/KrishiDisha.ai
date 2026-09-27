# Model card: KrishiDisha field disease classifier

* Architecture: `timm:convnext_tiny`, input 224px, 54 classes (taxonomy 2026.09)
* Trained: 2026-09-26T13:48:43 on 10344 field images from coco_val_subset, cotton_leaf, mango_leaf, plantdoc, rice_leaf_4, sugarcane_leaf, wheat_leaf, wheat_rust_cgiar
* Validation top-1 99.6%; calibrated with temperature 0.569
* Headline: own field test - top-1 (0 images); public field test 95.6% top-1 (1650 images)
* PlantVillage (lab photography) is NOT part of the training set or any number above.

## Intended use
Decision support for Indian farmers photographing a single leaf in daylight. Tier C crops are experimental; every prediction below the uncertainty threshold is shown as uncertain. Not a substitute for a KVK diagnosis.

## Data sources and licences

| Source | Domain | Country | Licence |
|---|---|---|---|
| coco_val_subset | ood | n/a | CC BY 4.0 (COCO annotations); images under Flickr terms - used only for the not-a-leaf class |
| cotton_leaf | field | India / Pakistan (field photos) | verify |
| mango_leaf | field | Bangladesh (orchards, phone camera) | CC BY 4.0 (Mendeley Data: MangoLeafBD) |
| plantdoc | field | internet field photographs (worldwide) | CC BY 4.0 |
| rice_leaf_4 | field | Bangladesh / India (field photos) | CC BY 4.0 (Mendeley origin) - verify on the Kaggle page |
| sugarcane_leaf | field | India (sugarcane fields) | verify |
| wheat_leaf | field | Ethiopia (field photos) | verify |

## Crop coverage

| Crop | Tier | Top-1 |
|---|---|---|
| Apple | C | 72.4% |
| Bell_pepper | C | 88.2% |
| Blueberry | C | 72.7% |
| Cherry | C | 60.0% |
| Cotton | B | 99.3% |
| Grape | C | 90.0% |
| Maize | C | 61.5% |
| Mango | A | 100.0% |
| Peach | C | 66.7% |
| Potato | C | 50.0% |
| Raspberry | C | 100.0% |
| Rice | A | 100.0% |
| Soybean | C | 75.0% |
| Squash | C | 100.0% |
| Strawberry | C | 100.0% |
| Sugarcane | A | 98.3% |
| Tomato | C | 69.6% |
| Wheat | B | 96.7% |