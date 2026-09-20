
# Speech Emotion Recognition

## Task 7: Standard Supervised Contrastive Learning

### Objective

To implement Standard Supervised Contrastive Learning (SupCon) for Speech Emotion Recognition using WavLM embeddings.

### Implementation

The following components were implemented:

1. Added a projection head after the WavLM feature extraction layer.
2. Implemented Standard Supervised Contrastive Loss.
3. Created positive pairs using samples belonging to the same emotion class.
4. Added a standard batch sampler.
5. Integrated supervised contrastive loss with the classification loss.
6. Added support for freezing the WavLM backbone.
7. Implemented evaluation using Accuracy, UAR, Macro-F1, and a confusion matrix.
8. Kept the WavLM + Cross-Entropy baseline configuration separate.
9. Added a combined training objective using SupCon Loss and Cross-Entropy Loss.

### Model Configuration

| Parameter | Value |
|---|---|
| Backbone | microsoft/wavlm-base |
| Learning approach | Standard SupCon |
| Projection head | Enabled |
| Classifier | Enabled |
| Projection dimension | 128 |
| Pooling | Attention |
| Backbone freezing | Enabled in configuration |
| Batch sampler | Standard |
| Batch size | 16 |
| Temperature | 0.07 |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Training epochs | 10 |
| SupCon loss weight | 1.0 |
| Cross-Entropy loss weight | 1.0 |
| Target sampling rate | 16,000 Hz |
| Maximum audio duration | 6 seconds |

### Loss Function

The model uses a combined training objective:

\[
L = \lambda_{\text{SupCon}}L_{\text{SupCon}}
+ \lambda_{\text{CE}}L_{\text{CE}}
\]

The current configuration uses:

- SupCon loss weight: 1.0
- Cross-Entropy loss weight: 1.0
- Temperature: 0.07

Therefore, both supervised contrastive learning and classification loss contribute to the training objective.

---

## Experimental Results

### Initial Training Run

The initial training run showed highly concentrated predictions, with the model predicting a single emotion class for the validation samples.

| Metric | Result |
|---|---:|
| Best UAR | 16.67% |
| Final UAR | 16.67% |
| Final Weighted Accuracy | 18.18% |
| Macro-F1 | 5.13% |

The initial results indicated that the model was not effectively distinguishing between the emotion classes.

### Latest Training Run

A subsequent training run was performed using the Standard SupCon configuration.

| Metric | Result |
|---|---:|
| Validation Accuracy | 54.55% |
| UAR | 50.00% |
| WAR | 54.55% |
| Macro-F1 | 45.50% |
| Weighted F1-Score | 49.64% |
| Training Epochs | 10 |

The validation UAR improved from 16.67% in the initial run to 50.00% in the latest run.

The latest results indicate improved class-wise recognition compared with the initial run. However, further analysis is required to understand the performance of individual emotion classes.

### Training Progress

| Epoch | Validation Accuracy | UAR |
|---|---:|---:|
| 1 | 25.00% | 22.92% |
| 2 | 32.73% | 30.00% |
| 3 | 39.55% | 36.25% |
| 4 | 42.27% | 38.75% |
| 5 | 44.09% | 40.42% |
| 6 | 47.73% | 43.75% |
| 7 | 49.09% | 45.00% |
| 8 | 49.09% | 45.00% |
| 9 | 53.64% | 49.17% |
| 10 | 54.55% | 50.00% |

The validation metrics showed an overall improvement during training.

---

## Latest Confusion Matrix

The latest validation confusion matrix is:

| Actual Class | Predicted Angry | Predicted Disgust | Predicted Fear | Predicted Happy | Predicted Neutral | Predicted Sad |
|---|---:|---:|---:|---:|---:|---:|
| Angry | 27 | 7 | 2 | 0 | 0 | 4 |
| Disgust | 1 | 38 | 0 | 0 | 0 | 1 |
| Fear | 1 | 6 | 20 | 2 | 0 | 11 |
| Happy | 15 | 6 | 7 | 8 | 0 | 4 |
| Neutral | 2 | 0 | 0 | 0 | 0 | 18 |
| Sad | 3 | 8 | 1 | 1 | 0 | 27 |

### Confusion Matrix Observations

- Disgust achieved high recognition in the latest validation run.
- Neutral samples were frequently classified as sad.
- Happy had lower recognition compared with some of the other emotion classes.
- The model still shows confusion between multiple emotion categories.
- Class-wise recall should be analyzed alongside the overall UAR.

**Note:** The emotion labels must be displayed in the same order as the numerical label mapping used during training.

The label order is:

```python
EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad"
]
```

---

## Dataset Distribution

The training and validation distributions used in the latest run were:

### Training Set

| Emotion | Samples |
|---|---:|
| Angry | 112 |
| Disgust | 112 |
| Fear | 112 |
| Happy | 112 |
| Neutral | 56 |
| Sad | 112 |
| **Total** | **616** |

### Validation Set

| Emotion | Samples |
|---|---:|
| Angry | 40 |
| Disgust | 40 |
| Fear | 40 |
| Happy | 40 |
| Neutral | 20 |
| Sad | 40 |
| **Total** | **220** |

The neutral class contains fewer samples than the other emotion classes. This distribution should be considered when analyzing class-wise performance.

---

## Limitations and Future Work

- Analyze the reasons for confusion between emotion classes.
- Investigate the low recognition rate of the neutral class.
- Verify that the WavLM backbone freezing configuration is applied correctly.
- Analyze the Standard SupCon loss implementation and positive-pair generation.
- Evaluate class-wise precision, recall, and F1-score.
- Investigate the effect of batch size on supervised contrastive learning.
- Compare the Standard SupCon model with the WavLM + Cross-Entropy baseline.
- Experiment with class-balanced sampling or class-weighted classification loss.
- Perform evaluation on the test dataset.
- Compare multiple runs using consistent random seeds.
- Investigate the effect of different temperature values and loss weights.

---

## Status

The Standard SupCon implementation and evaluation pipeline were completed.

The initial training run achieved a UAR of 16.67%. A subsequent training run achieved a validation UAR of 50.00%, showing improved performance.

The latest results are validation results and require further investigation through class-wise analysis, test-set evaluation, and comparison with the baseline model.

The implementation and experimental results will be further refined as part of the ongoing Speech Emotion Recognition project.
