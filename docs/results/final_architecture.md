# Final Speech Emotion Recognition Architecture

This document outlines the actual end-to-end architecture and pipeline used in the final evaluation of the project.

## 1. Pipeline Overview

The final architecture follows a two-stage paradigm where feature extraction is decoupled from the classification training, leveraging a frozen, highly capable self-supervised model.

**Dataset &rarr; Preprocessing &rarr; Offline Feature Extraction &rarr; Joint Training (SupCon + CE) &rarr; Evaluation**

---

## 2. Component Details

### 2.1 Datasets
The project utilizes three standard emotion recognition corpora:
* **RAVDESS**
* **IEMOCAP**
* **CREMA-D**
* **Classes (6):** Angry, Disgust, Fear, Happy, Neutral, Sad.

### 2.2 Audio Preprocessing
Before feeding data to the model, the raw audio is standardized:
* **Format:** Mono-channel.
* **Sampling Rate:** Resampled to a consistent `16 kHz` (using `scipy.signal.resample_poly`).
* **Duration Handling:** Variable lengths are handled by mean-pooling the output representations over the time dimension.

### 2.3 Feature Extraction (Backbone)
* **Model:** `microsoft/wavlm-base` (WavLM Base).
* **Process:** The backbone is completely **frozen** (offline extraction) to save computational resources.
* **Representation:** The waveform is passed through WavLM. The `last_hidden_state` (dimensions: `[1, frames, 768]`) is extracted. The time dimension is mean-pooled to generate a single, fixed-size **768-dimensional embedding** per audio file.

### 2.4 Sampling and Batching
* **Batching:** Standard random batch sampling is used to construct batches.
* **Speaker & Corpus-Aware Weighting:** Instead of manipulating the data loader, the "sampling" logic is embedded in the Supervised Contrastive (SupCon) Loss function via custom contrastive weights:
  * **1.0** weight for positive pairs with the same emotion, different speaker, and different corpus (highest priority).
  * **0.5** weight for positive pairs with the same emotion, different speaker, but same corpus.
  * **0.25** weight for positive pairs with the same emotion and same speaker (lowest priority).

### 2.5 Training Architecture (`WavLMEmbeddingSupConModel`)
The model trained on top of the frozen embeddings contains two parallel heads:
1. **Projection Head:** Projects the 768-dim embeddings down to a `128-dimensional` space. This head is exclusively used to compute the SupCon loss, learning disentangled emotion representations.
2. **Classification Head:** A standard linear layer mapping the 768-dim embeddings directly to the `6` emotion logits. This is used for Cross-Entropy (CE) classification.

### 2.6 Training Procedure
The model is trained **jointly** on both tasks. In every training step:
* The SupCon Loss is calculated using the projections and the custom speaker/corpus-aware weights.
* The Cross-Entropy Loss is calculated using the classification logits and the ground-truth labels.
* The final loss is a weighted sum: `Loss = (supcon_weight * SupCon Loss) + (ce_weight * CE Loss)`.

### 2.7 Evaluation Pipeline
* **Within-Corpus Evaluation:** Evaluated on the held-out test split of the target dataset (e.g., RAVDESS test split).
* **Cross-Corpus Evaluation:** The model is trained on the full sets of two corpora and evaluated directly on the entirety of an unseen third corpus (Zero-Shot Domain Transfer).
* **Metrics:** Evaluated using Test Accuracy, Test Macro-F1, and primarily **Test Unweighted Average Recall (UAR)** to account for class imbalances and model collapse.
