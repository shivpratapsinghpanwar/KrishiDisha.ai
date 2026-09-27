# Disease classifier evaluation (2026-09-26T13:52:13)

Checkpoint `plant_disease_model.pt`, arch `timm:convnext_tiny`, 224px, 54 classes, temperature 0.569, 8.6 ms/image on cuda.

## Headline

| Set | Images | Top-1 | Top-3 |
|---|---|---|---|
| **KrishiDisha own field test** | 0 | - | - |
| Public field test (all sources) | 1650 | 95.6% | 99.3% |
| All test incl. not-a-leaf | 1875 | 96.1% | 99.4% |

Macro-F1 85.7%; ECE 0.0235; not-a-leaf AUROC 49.4%, leaf-pass threshold 0.997 lets 97.8% of non-leaves through

## Per source

| Source | Images | Top-1 | Top-3 |
|---|---|---|---|
| coco_val_subset | 225 | 99.1% | 100.0% |
| cotton_leaf | 149 | 99.3% | 100.0% |
| mango_leaf | 544 | 100.0% | 100.0% |
| plantdoc | 236 | 73.3% | 94.9% |
| rice_leaf_4 | 317 | 100.0% | 100.0% |
| sugarcane_leaf | 343 | 98.3% | 100.0% |
| wheat_leaf | 61 | 96.7% | 100.0% |

## Per crop (field test)

| Crop | Tier | Train field imgs | Test imgs | Top-1 | Top-3 | Classes |
|---|---|---|---|---|---|---|
| Apple | C | 239 | 29 | 72.4% | 93.1% | 3 |
| Bell_pepper | C | 112 | 17 | 88.2% | 100.0% | 2 |
| Blueberry | C | 103 | 11 | 72.7% | 100.0% | 1 |
| Cherry | C | 47 | 10 | 60.0% | 90.0% | 1 |
| Cotton | B | 697 | 149 | 99.3% | 100.0% | 4 |
| Grape | C | 111 | 20 | 90.0% | 100.0% | 2 |
| Maize | C | 339 | 26 | 61.5% | 100.0% | 3 |
| Mango | A | 2542 | 544 | 100.0% | 100.0% | 8 |
| Peach | C | 99 | 9 | 66.7% | 100.0% | 1 |
| Potato | C | 196 | 16 | 50.0% | 93.8% | 2 |
| Raspberry | C | 111 | 7 | 100.0% | 100.0% | 1 |
| Rice | A | 1478 | 317 | 100.0% | 100.0% | 4 |
| Soybean | C | 56 | 8 | 75.0% | 87.5% | 1 |
| Squash | C | 122 | 6 | 100.0% | 100.0% | 1 |
| Strawberry | C | 88 | 8 | 100.0% | 100.0% | 1 |
| Sugarcane | A | 1602 | 343 | 98.3% | 100.0% | 5 |
| Tomato | C | 637 | 69 | 69.6% | 89.9% | 8 |
| Wheat | B | 715 | 61 | 96.7% | 100.0% | 3 |

Tier A: >= 1,000 field training images and >= 90% top-1. B: >= 300 and >= 80%. C: experimental (shown with a warning in the app).

## Per class

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| Apple___Cedar_apple_rust | 10 | 0.875 | 0.700 | 0.778 |
| Apple___Scab | 10 | 0.636 | 0.700 | 0.667 |
| Apple___healthy | 9 | 0.700 | 0.778 | 0.737 |
| Bell_pepper___Bacterial_spot | 9 | 0.889 | 0.889 | 0.889 |
| Bell_pepper___healthy | 8 | 0.778 | 0.875 | 0.824 |
| Blueberry___healthy | 11 | 0.800 | 0.727 | 0.762 |
| Cherry___healthy | 10 | 0.667 | 0.600 | 0.632 |
| Cotton___Bacterial_blight | 32 | 1.000 | 1.000 | 1.000 |
| Cotton___Leaf_curl_virus | 33 | 1.000 | 1.000 | 1.000 |
| Cotton___Wilt | 40 | 1.000 | 0.975 | 0.987 |
| Cotton___healthy | 44 | 0.978 | 1.000 | 0.989 |
| Grape___Black_rot | 8 | 0.875 | 0.875 | 0.875 |
| Grape___healthy | 12 | 0.917 | 0.917 | 0.917 |
| Maize___Common_rust | 10 | 1.000 | 0.800 | 0.889 |
| Maize___Gray_leaf_spot | 4 | 0.143 | 0.250 | 0.182 |
| Maize___Northern_leaf_blight | 12 | 0.700 | 0.583 | 0.636 |
| Mango___Anthracnose | 71 | 1.000 | 1.000 | 1.000 |
| Mango___Bacterial_canker | 73 | 1.000 | 1.000 | 1.000 |
| Mango___Cutting_weevil | 36 | 1.000 | 1.000 | 1.000 |
| Mango___Die_back | 71 | 0.986 | 1.000 | 0.993 |
| Mango___Gall_midge | 73 | 1.000 | 1.000 | 1.000 |
| Mango___Powdery_mildew | 74 | 1.000 | 1.000 | 1.000 |
| Mango___Sooty_mould | 72 | 1.000 | 1.000 | 1.000 |
| Mango___healthy | 74 | 0.987 | 1.000 | 0.993 |
| Other___not_a_leaf | 225 | 0.991 | 0.991 | 0.991 |
| Peach___healthy | 9 | 0.857 | 0.667 | 0.750 |
| Potato___Early_blight | 8 | 0.444 | 0.500 | 0.471 |
| Potato___Late_blight | 8 | 0.444 | 0.500 | 0.471 |
| Raspberry___healthy | 7 | 1.000 | 1.000 | 1.000 |
| Rice___Bacterial_leaf_blight | 77 | 1.000 | 1.000 | 1.000 |
| Rice___Blast | 71 | 1.000 | 1.000 | 1.000 |
| Rice___Brown_spot | 91 | 0.989 | 1.000 | 0.995 |
| Rice___Tungro | 78 | 1.000 | 1.000 | 1.000 |
| Soybean___healthy | 8 | 0.750 | 0.750 | 0.750 |
| Squash___Powdery_mildew | 6 | 1.000 | 1.000 | 1.000 |
| Strawberry___healthy | 8 | 0.889 | 1.000 | 0.941 |
| Sugarcane___Mosaic | 56 | 1.000 | 0.982 | 0.991 |
| Sugarcane___Red_rot | 76 | 0.938 | 1.000 | 0.968 |
| Sugarcane___Rust | 66 | 1.000 | 0.970 | 0.985 |
| Sugarcane___Yellow_leaf | 74 | 1.000 | 0.960 | 0.979 |
| Sugarcane___healthy | 71 | 0.986 | 1.000 | 0.993 |
| Tomato___Bacterial_spot | 9 | 1.000 | 0.444 | 0.615 |
| Tomato___Early_blight | 9 | 0.875 | 0.778 | 0.824 |
| Tomato___Late_blight | 10 | 0.727 | 0.800 | 0.762 |
| Tomato___Leaf_mold | 6 | 0.417 | 0.833 | 0.556 |
| Tomato___Mosaic_virus | 10 | 1.000 | 0.300 | 0.462 |
| Tomato___Septoria_leaf_spot | 11 | 0.733 | 1.000 | 0.846 |
| Tomato___Yellow_leaf_curl_virus | 6 | 1.000 | 0.833 | 0.909 |
| Tomato___healthy | 8 | 0.714 | 0.625 | 0.667 |
| Wheat___Septoria | 15 | 1.000 | 1.000 | 1.000 |
| Wheat___Stripe_rust | 31 | 0.968 | 0.968 | 0.968 |
| Wheat___healthy | 15 | 0.875 | 0.933 | 0.903 |