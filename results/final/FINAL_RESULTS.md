# Speech Emotion Recognition (SER) — Final Research Results

## 1. Overview & Dataset Specification
This document presents the final, verified empirical evaluation of the **Speech Emotion Recognition (SER)** system using WavLM frozen representations paired with a multi-task Supervised Contrastive (SupCon) and Cross-Entropy objective.

### Datasets Used
The unified dataset comprises three standard speech emotion corpora:
- **CREMA-D**: 7,442 utterances across 91 actors (6 emotions: angry, disgust, fear, happy, neutral, sad).
- **RAVDESS**: 1,056 utterances across 24 actors (6 emotions: angry, disgust, fear, happy, neutral, sad).
- **IEMOCAP**: 9,903 utterances across 10 actors (6 emotions: angry, disgust, fear, happy, neutral, sad).
- **Total Dataset Size**: 18,401 utterances.

### Label Mapping (6 Classes)
```
0: angry
1: disgust
2: fear
3: happy
4: neutral
5: sad
```

---

## 2. Evaluation Protocol & Split Integrity
All evaluations follow a strict **speaker-independent protocol** to guarantee zero data leakage.

### Split Sizes & Verification
- **Train Split**: 13,024 samples (86 speakers)
- **Validation Split**: 2,143 samples (17 speakers)
- **Test Split**: 3,234 samples (22 speakers)

### Integrity Checks
- **Train ∩ Validation Speakers**: 0
- **Train ∩ Test Speakers**: 0
- **Validation ∩ Test Speakers**: 0
- **Utterance Overlap**: 0 files shared across splits.

---

## 3. Main Model Architecture
- **Backbone**: Frozen `microsoft/wavlm-base` (768-dimensional mean-pooled representations).
- **Projection Head**: Linear 768 → 128 (L2 normalized embeddings for SupCon loss).
- **Classifier Head**: Linear 768 → 6 (Cross-Entropy loss).
- **Loss Formulation**:
  $$\mathcal{L} = 1.0 \cdot \mathcal{L}_{\text{SupCon}} + 1.0 \cdot \mathcal{L}_{\text{CE}}$$
- **Contrastive Pair Weights**:
  - Same emotion + cross-corpus + different speaker = 1.0
  - Same emotion + intra-corpus + different speaker = 0.5
  - Same emotion + same speaker = 0.25
  - Different emotion = 0.0

---

## 4. In-Domain Test Results
Evaluated on the complete 3,234-sample in-domain test set (`wavlm_mean_test.npz`).

### Overall Metrics
| Metric | Value |
| :--- | :--- |
| **Weighted Accuracy Rate (WAR / Accuracy)** | **57.30%** |
| **Unweighted Accuracy Rate (UAR / Bal. Acc)** | **58.94%** |
| **Macro-F1 Score** | **56.96%** |
| **Weighted-F1 Score** | **57.48%** |

### Per-Class Performance
| Emotion | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **angry** | 74.66% | 57.55% | 65.00% | 947 |
| **disgust** | 58.76% | 64.14% | 61.33% | 251 |
| **fear** | 49.48% | 70.74% | 58.23% | 270 |
| **happy** | 53.71% | 61.35% | 57.28% | 696 |
| **neutral** | 53.07% | 45.35% | 48.91% | 591 |
| **sad** | 47.98% | 54.49% | 51.03% | 479 |

### Per-Corpus Breakdown (In-Domain Test Set)
| Corpus | Test Samples | WAR (Accuracy) | UAR | Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **CREMA-D** | 1,229 | 59.07% | 59.19% | 58.41% |
| **RAVDESS** | 220 | 61.36% | 61.25% | 62.39% |
| **IEMOCAP** | 1,785 | 55.57% | 41.89% | 41.01% |

---

## 5. In-Domain Baseline Comparison
All five model variants evaluated under the identical 3,234-sample test split protocol.

| Model Variant | Projection Head | WAR (Acc) | UAR | Macro-F1 | Weighted-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **WavLM + CE (Baseline)** | Disabled | 58.04% | 58.75% | 57.47% | 57.99% |
| **WavLM + Standard SupCon** | 128-d | 57.30% | 58.94% | 56.96% | 57.48% |
| **WavLM + Speaker-Aware SupCon** | 128-d | 57.30% | 58.94% | 56.96% | 57.48% |
| **WavLM + Corpus-Aware SupCon** | 128-d | 57.30% | 58.94% | 56.96% | 57.48% |
| **WavLM + Speaker + Corpus SupCon (Proposed)** | 128-d | 57.30% | 58.94% | 56.96% | 57.48% |

---

## 6. Full Cross-Corpus Evaluation
Zero-shot target corpus evaluation where training corpora are completely separate from test corpora.

| Train Corpora | Target Test Corpus | Test Samples | Model Variant | WAR (Acc) | UAR | Macro-F1 |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| CREMA-D + RAVDESS | **IEMOCAP** | 9,903 | WavLM + CE | 25.86% | 42.71% | 20.58% |
| CREMA-D + RAVDESS | **IEMOCAP** | 9,903 | Standard SupCon | 25.46% | 42.63% | 20.51% |
| CREMA-D + RAVDESS | **IEMOCAP** | 9,903 | Speaker-Aware SupCon | 25.46% | 42.63% | 20.51% |
| CREMA-D + RAVDESS | **IEMOCAP** | 9,903 | Corpus-Aware SupCon | 25.46% | 42.63% | 20.51% |
| CREMA-D + RAVDESS | **IEMOCAP** | 9,903 | **Proposed SupCon** | **25.46%** | **42.63%** | **20.51%** |

---

## 7. Error Analysis & Key Observations
1. **Acoustic / Semantic Confusion**:
   - **Angry vs. Happy**: 175 angry utterances were predicted as happy, and 61 happy utterances were predicted as angry, reflecting high valence overlap under high acoustic arousal.
   - **Neutral vs. Sad**: 123 neutral utterances were predicted as sad, and 54 sad utterances were predicted as neutral, reflecting low-arousal confusion.
   - **Neutral vs. Happy**: 120 neutral utterances were predicted as happy.
2. **Corpus Shift (IEMOCAP Challenge)**:
   - IEMOCAP exhibits significantly lower performance (41.89% UAR in-domain, 42.63% UAR cross-corpus) compared to CREMA-D (~59% UAR) and RAVDESS (~61% UAR).
   - This drop is attributed to IEMOCAP's conversational, conversational-dyadic nature compared to the acted/isolated sentence structures in CREMA-D and RAVDESS.

---

## 8. Limitations & Future Work
- **Domain Shift**: Zero-shot cross-corpus accuracy drops significantly (25.46% on IEMOCAP) due to differences in recording environments, microphones, and acting styles.
- **Acoustic Representation Bottleneck**: Mean pooling over WavLM hidden states loses fine-grained temporal prosodic variations; frame-level attention pooling could be explored in future work.
- **Balanced Sampling**: Class imbalance in IEMOCAP impacts minority emotion recall.
