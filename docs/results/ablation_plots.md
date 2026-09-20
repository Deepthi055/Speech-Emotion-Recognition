# Ablation Study Plots

The following plots visualize the within-corpus performance (on the RAVDESS dataset) across all tested model configurations.

## 1. Test Accuracy

![Ablation Accuracy](/d:/DLproject/Speech-Emotion-Recognition/docs/results/images/ablation_accuracy.png)

## 2. Test Unweighted Average Recall (UAR)

![Ablation UAR](/d:/DLproject/Speech-Emotion-Recognition/docs/results/images/ablation_uar.png)

## 3. Test Macro-F1

![Ablation Macro-F1](/d:/DLproject/Speech-Emotion-Recognition/docs/results/images/ablation_macro_f1.png)

*Observation: Adding Supervised Contrastive Learning (SupCon) improves performance over the WavLM + CE Baseline across all metrics. However, in this specific within-corpus scenario, the variants of SupCon (Standard, Speaker-Aware, Corpus-Aware, and Proposed) achieve identical test results.*
