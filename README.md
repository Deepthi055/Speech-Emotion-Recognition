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
| Temperature | 0.07 |
| Learning rate | 1e-4 |
| Training epochs | 10 |

### Initial Training Results

An initial training run was performed using the Standard SupCon configuration.

| Metric | Result |
|---|---:|
| Best UAR | 16.67% |
| Final UAR | 16.67% |
| Final Weighted Accuracy | 18.18% |
| Macro-F1 | 5.13% |

### Initial Confusion Matrix

The initial validation results showed that the model predicted the same emotion class for all validation samples.

| Actual Emotion | Predicted Emotion |
|---|---|
| Angry | Disgust |
| Disgust | Disgust |
| Fear | Disgust |
| Happy | Disgust |
| Neutral | Disgust |
| Sad | Disgust |

These results indicate that the initial training run did not achieve effective emotion classification.

The results are documented as initial training results and should not be interpreted as evidence of successful model learning.

### Limitations and Future Work

- Investigate why predictions were concentrated in a single emotion class.
- Verify training behavior after applying the backbone-freezing configuration fix.
- Perform another training run if final performance evaluation of the corrected implementation is required.
- Analyze class-wise performance and the confusion matrix.

### Status

The Standard SupCon implementation and initial evaluation were completed. The initial model performance requires further investigation.
