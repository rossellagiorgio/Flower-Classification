# Flower Classification with Transfer Learning (VGG16)

Classifies images into **104 flower classes** in TensorFlow/Keras on a single Kaggle GPU (Tesla T4).

## Results

| Stage | Macro F1 | Precision | Recall | Val accuracy |
|---|---|---|---|---|
| Head only (VGG16 frozen) | 0.822 | 0.856 | 0.806 | see notebook |
| + Fine-tuning (`block5`) | **0.886** | 0.915 | 0.873 | 0.894 |

Fine-tuning the last convolutional block improved F1 by about 6 points.

Instead of training a network from scratch, we reuse VGG16, which was already trained on ImageNet (1.2M images, 1000 classes), and adapt it to our 104 flower classes. This needs less data and less time, and usually gives better accuracy.

## How it works

1. **Data:** TFRecords decoded to RGB, resized to 331x331.
2. **Model:** VGG16 (no top layer) + augmentation (flip, rotation) + `GlobalAveragePooling2D` + `Dropout(0.3)` + `Dense(104, softmax)`.
3. **Stage 1 (head only):** VGG16 frozen, only the new head is trained (Adam, lr = 1e-3).
4. **Stage 2 (fine-tuning):** the last block of VGG16 (`block5`) is unfrozen and trained together with the head (Adam, lr = 1e-5). The model must be recompiled after changing `trainable`.

## Files

```
.
├── notebook.ipynb    # full pipeline: data, training, evaluation, submission
└── README.md
```

## How to run

1. Open the notebook on Kaggle and attach the dataset.
2. Set **Accelerator -> GPU**.
3. Run all cells in order. The last cell writes `submission.csv` (test predictions (id, label)).


## Possible improvements

- Unfreeze `block4` or try DenseNet201.
- Stronger augmentation and test-time augmentation.

