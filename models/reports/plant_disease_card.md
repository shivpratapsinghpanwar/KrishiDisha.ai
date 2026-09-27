# Model card: KrishiDisha field disease classifier

* Architecture: `timm:convnext_tiny`, input 224px, 60 classes (taxonomy 2026.09)
* Trained: 2026-09-26T19:36:31 on 15840 field images from coco_val_subset, cotton_leaf, mango_leaf, paddy_doctor, plantdoc, rice_leaf_4, sugarcane_leaf, wheat_leaf, wheat_rust_cgiar
* Validation top-1 96.5%; calibrated with temperature 0.719
* Headline: own field test - top-1 (0 images); public field test 94.3% top-1 (2828 images)
* PlantVillage (lab photography) is NOT part of the training set or any number above.

## Availability

The weights are private (not in the repository, not on Kaggle). Code, data recipe, metrics and this card are public. 
To obtain the weights for research, a pilot or a partnership, email Shivpratap Singh Panwar at shivpratapsinghpanwar19@gmail.com.

## Intended use
Decision support for Indian farmers photographing a single leaf in daylight. Tier C crops are experimental; every prediction below the uncertainty threshold is shown as uncertain. Not a substitute for a KVK diagnosis.

## Data sources and licences

| Source | Domain | Country | Licence |
|---|---|---|---|
| coco_val_subset | ood | n/a | CC BY 4.0 (COCO annotations); images under Flickr terms - used only for the not-a-leaf class |
| cotton_leaf | field | India / Pakistan (field photos) | verify |
| mango_leaf | field | Bangladesh (orchards, phone camera) | CC BY 4.0 (Mendeley Data: MangoLeafBD) |
| paddy_doctor | field | India (Tamil Nadu paddy fields, phone camera) | Kaggle competition rules (Paddy Doctor, 2022) - verify commercial use; the authors also published the dataset (Petchiammal et al.) - prefer that release for a commercial build |
| plantdoc | field | internet field photographs (worldwide) | CC BY 4.0 |
| rice_leaf_4 | field | Bangladesh / India (field photos) | CC BY 4.0 (Mendeley origin) - verify on the Kaggle page |
| sugarcane_leaf | field | India (sugarcane fields) | verify |
| wheat_leaf | field | Ethiopia (field photos) | verify |

## Crop coverage

| Crop | Tier | Top-1 |
|---|---|---|
| Apple | C | 79.3% |
| Bell_pepper | C | 58.8% |
| Blueberry | C | 63.6% |
| Cherry | C | 70.0% |
| Cotton | B | 99.3% |
| Grape | C | 95.0% |
| Maize | C | 57.7% |
| Mango | A | 100.0% |
| Peach | C | 66.7% |
| Potato | C | 37.5% |
| Raspberry | C | 100.0% |
| Rice | A | 94.2% |
| Soybean | C | 62.5% |
| Squash | C | 100.0% |
| Strawberry | C | 100.0% |
| Sugarcane | A | 98.8% |
| Tomato | C | 68.1% |
| Wheat | B | 100.0% |