# Experimental Hyperparameters

This document details all relevant hyperparameters configured in the final evaluation experiments (both within-corpus and cross-corpus).

## 1. Model Configuration
* **Feature Backbone:** `microsoft/wavlm-base` (Frozen)
* **Input Feature Dimension:** `768` (Mean-pooled over time)
* **Projection Head Output Dimension (SupCon):** `128`
* **Classification Head Output Dimension:** `6` (Emotion Classes)
* **Dropout:** None explicitly added to the linear projection/classification heads.

## 2. Training Hyperparameters
* **Optimizer:** `AdamW`
* **Learning Rate:** `0.001`
* **Weight Decay:** `0.0001`
* **Batch Size:** `64`
* **Max Epochs:** `30`
* **Scheduler/Warmup:** None explicitly configured.
* **Random Seed:** `42` (Used for dataset splitting and network initialization)

## 3. Loss & Sampling Parameters
* **Loss Function:** Joint optimization of Cross-Entropy (CE) and Supervised Contrastive Loss (SupCon).
* **Loss Weighting:** CE Weight = `1.0`, SupCon Weight = `1.0`.
* **Class Weighting (CE):** None (Standard, unweighted CE).
* **SupCon Temperature:** `0.07`
* **Contrastive Weights (Speaker+Corpus-Aware Sampling):**
  * Different Speaker, Different Corpus = `1.0`
  * Different Speaker, Same Corpus = `0.5`
  * Same Speaker, Same Corpus = `0.25`

## 4. Evaluation & Checkpoint Selection
* **Early Stopping Patience:** `7` epochs
* **Early Stopping Metric:** Validation Unweighted Average Recall (UAR)
* **Checkpoint Selection:** The model checkpoint achieving the highest Validation UAR is selected for final Test evaluation.
