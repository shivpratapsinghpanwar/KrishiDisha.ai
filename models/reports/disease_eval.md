# Disease classifier evaluation (2026-09-26T19:37:05)

Checkpoint `plant_disease_model.pt`, arch `timm:convnext_tiny`, 224px, 60 classes, temperature 0.719, 5.9 ms/image on cuda.

## Headline

| Set | Images | Top-1 | Top-3 |
|---|---|---|---|
| **KrishiDisha own field test** | 0 | - | - |
| Public field test (all sources) | 2828 | 94.3% | 98.8% |
| All test incl. not-a-leaf | 3053 | 94.7% | 98.9% |

Macro-F1 84.8%; ECE 0.0212; non-leaf photos rejected 99.6%, real leaves accepted 93.2% (max-softmax AUROC 43.3%, secondary)

## Per source

| Source | Images | Top-1 | Top-3 |
|---|---|---|---|
| coco_val_subset | 225 | 99.6% | 100.0% |
| cotton_leaf | 149 | 99.3% | 100.0% |
| mango_leaf | 544 | 100.0% | 100.0% |
| paddy_doctor | 1178 | 92.7% | 98.4% |
| plantdoc | 236 | 70.3% | 93.6% |
| rice_leaf_4 | 317 | 100.0% | 100.0% |
| sugarcane_leaf | 343 | 98.8% | 100.0% |
| wheat_leaf | 61 | 100.0% | 100.0% |

## Per crop (field test)

| Crop | Tier | Train field imgs | Test imgs | Top-1 | Top-3 | Classes |
|---|---|---|---|---|---|---|
| Apple | C | 239 | 29 | 79.3% | 96.6% | 3 |
| Bell_pepper | C | 112 | 17 | 58.8% | 88.2% | 2 |
| Blueberry | C | 103 | 11 | 63.6% | 100.0% | 1 |
| Cherry | C | 47 | 10 | 70.0% | 90.0% | 1 |
| Cotton | B | 697 | 149 | 99.3% | 100.0% | 4 |
| Grape | C | 112 | 20 | 95.0% | 100.0% | 2 |
| Maize | C | 339 | 26 | 57.7% | 96.2% | 3 |
| Mango | A | 2542 | 544 | 100.0% | 100.0% | 8 |
| Peach | C | 100 | 9 | 66.7% | 88.9% | 1 |
| Potato | C | 196 | 16 | 37.5% | 93.8% | 2 |
| Raspberry | C | 111 | 7 | 100.0% | 100.0% | 1 |
| Rice | A | 6972 | 1495 | 94.2% | 98.7% | 10 |
| Soybean | C | 56 | 8 | 62.5% | 100.0% | 1 |
| Squash | C | 122 | 6 | 100.0% | 100.0% | 1 |
| Strawberry | C | 88 | 8 | 100.0% | 100.0% | 1 |
| Sugarcane | A | 1602 | 343 | 98.8% | 100.0% | 5 |
| Tomato | C | 637 | 69 | 68.1% | 88.4% | 8 |
| Wheat | B | 715 | 61 | 100.0% | 100.0% | 3 |

Tier A: >= 1,000 field training images and >= 90% top-1. B: >= 300 and >= 80%. C: experimental (shown with a warning in the app).

## Per class

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| Apple___Cedar_apple_rust | 10 | 1.000 | 0.800 | 0.889 |
| Apple___Scab | 10 | 0.889 | 0.800 | 0.842 |
| Apple___healthy | 9 | 0.538 | 0.778 | 0.636 |
| Bell_pepper___Bacterial_spot | 9 | 0.625 | 0.556 | 0.588 |
| Bell_pepper___healthy | 8 | 0.714 | 0.625 | 0.667 |
| Blueberry___healthy | 11 | 0.700 | 0.636 | 0.667 |
| Cherry___healthy | 10 | 0.700 | 0.700 | 0.700 |
| Cotton___Bacterial_blight | 32 | 1.000 | 1.000 | 1.000 |
| Cotton___Leaf_curl_virus | 33 | 1.000 | 1.000 | 1.000 |
| Cotton___Wilt | 40 | 0.976 | 1.000 | 0.988 |
| Cotton___healthy | 44 | 1.000 | 0.977 | 0.989 |
| Grape___Black_rot | 8 | 1.000 | 0.875 | 0.933 |
| Grape___healthy | 12 | 0.923 | 1.000 | 0.960 |
| Maize___Common_rust | 10 | 1.000 | 0.800 | 0.889 |
| Maize___Gray_leaf_spot | 4 | 0.143 | 0.250 | 0.182 |
| Maize___Northern_leaf_blight | 12 | 0.600 | 0.500 | 0.545 |
| Mango___Anthracnose | 71 | 0.986 | 1.000 | 0.993 |
| Mango___Bacterial_canker | 73 | 1.000 | 1.000 | 1.000 |
| Mango___Cutting_weevil | 36 | 1.000 | 1.000 | 1.000 |
| Mango___Die_back | 71 | 1.000 | 1.000 | 1.000 |
| Mango___Gall_midge | 73 | 1.000 | 1.000 | 1.000 |
| Mango___Powdery_mildew | 74 | 1.000 | 1.000 | 1.000 |
| Mango___Sooty_mould | 72 | 1.000 | 1.000 | 1.000 |
| Mango___healthy | 74 | 1.000 | 1.000 | 1.000 |
| Other___not_a_leaf | 225 | 0.996 | 0.996 | 0.996 |
| Peach___healthy | 9 | 0.857 | 0.667 | 0.750 |
| Potato___Early_blight | 8 | 0.444 | 0.500 | 0.471 |
| Potato___Late_blight | 8 | 0.333 | 0.250 | 0.286 |
| Raspberry___healthy | 7 | 0.875 | 1.000 | 0.933 |
| Rice___Bacterial_leaf_blight | 134 | 0.934 | 0.948 | 0.941 |
| Rice___Bacterial_leaf_streak | 35 | 0.970 | 0.914 | 0.941 |
| Rice___Bacterial_panicle_blight | 43 | 0.915 | 1.000 | 0.956 |
| Rice___Blast | 267 | 0.952 | 0.888 | 0.919 |
| Rice___Brown_spot | 196 | 0.945 | 0.969 | 0.957 |
| Rice___Dead_heart | 170 | 1.000 | 0.994 | 0.997 |
| Rice___Downy_mildew | 62 | 0.853 | 0.839 | 0.846 |
| Rice___Hispa | 169 | 0.929 | 0.923 | 0.926 |
| Rice___Tungro | 203 | 0.910 | 0.946 | 0.927 |
| Rice___healthy | 216 | 0.959 | 0.977 | 0.968 |
| Soybean___healthy | 8 | 0.500 | 0.625 | 0.556 |
| Squash___Powdery_mildew | 6 | 1.000 | 1.000 | 1.000 |
| Strawberry___healthy | 8 | 0.889 | 1.000 | 0.941 |
| Sugarcane___Mosaic | 56 | 1.000 | 1.000 | 1.000 |
| Sugarcane___Red_rot | 76 | 1.000 | 0.974 | 0.987 |
| Sugarcane___Rust | 66 | 1.000 | 0.985 | 0.992 |
| Sugarcane___Yellow_leaf | 74 | 0.949 | 1.000 | 0.974 |
| Sugarcane___healthy | 71 | 1.000 | 0.986 | 0.993 |
| Tomato___Bacterial_spot | 9 | 0.571 | 0.444 | 0.500 |
| Tomato___Early_blight | 9 | 0.545 | 0.667 | 0.600 |
| Tomato___Late_blight | 10 | 0.667 | 0.800 | 0.727 |
| Tomato___Leaf_mold | 6 | 0.500 | 0.667 | 0.571 |
| Tomato___Mosaic_virus | 10 | 1.000 | 0.600 | 0.750 |
| Tomato___Septoria_leaf_spot | 11 | 0.692 | 0.818 | 0.750 |
| Tomato___Yellow_leaf_curl_virus | 6 | 1.000 | 0.833 | 0.909 |
| Tomato___healthy | 8 | 0.714 | 0.625 | 0.667 |
| Wheat___Septoria | 15 | 1.000 | 1.000 | 1.000 |
| Wheat___Stripe_rust | 31 | 1.000 | 1.000 | 1.000 |
| Wheat___healthy | 15 | 1.000 | 1.000 | 1.000 |