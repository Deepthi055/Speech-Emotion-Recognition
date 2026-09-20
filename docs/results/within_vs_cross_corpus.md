# Within-Corpus vs. Cross-Corpus Performance Analysis

This document analyzes the generalization capabilities of the Proposed SupCon (Speaker + Corpus-Aware) model by comparing its performance when evaluated within a single corpus (RAVDESS) versus its performance in cross-corpus settings.

## 1. Performance Overview

| Setup | Test Accuracy | Test UAR | Test Macro-F1 |
|:---|---:|---:|---:|
| **Within-Corpus (RAVDESS)** | 49.55% | 49.17% | 48.23% |
| **Cross-Corpus (Target: IEMOCAP)** | 62.50% | 12.82% | 13.89% |
| **Cross-Corpus (Target: RAVDESS)** | 21.88% | 19.64% | 12.19% |
| **Cross-Corpus (Target: CREMA-D)** | 21.88% | 16.67% | 6.14% |

## 2. Generalization Gap and Domain Shift

The experimental results demonstrate a severe **generalization gap** caused by domain shift across corpora. 

While the model achieves a balanced and reasonable performance in the within-corpus setting (UAR of 49.17%), its true performance collapses in cross-corpus evaluation. Unweighted Average Recall (UAR), which accounts for class imbalances and model collapse, drops precipitously to between 12.82% and 19.64% in cross-corpus scenarios. Since the classification task contains 6 emotions, random guessing would yield a UAR of approximately 16.67%. Thus, the cross-corpus performance is essentially at or below random chance.

## 3. The Metric Discrepancy (Accuracy vs. UAR)

A critical observation is the stark discrepancy between Accuracy and UAR, particularly in the `CREMA-D + RAVDESS -> IEMOCAP` experiment. The model achieves a deceptively high Test Accuracy of 62.50%, but a UAR of only 12.82%.

This discrepancy is a direct result of model collapse under domain shift, as seen in the confusion matrices. When faced with unseen acoustic environments, the model defaults to predicting a single majority class (overwhelmingly `Angry`). If the target test set contains a high proportion of that specific emotion, the standard accuracy appears artificially high, while UAR correctly penalizes the model for failing to recognize the other five emotions. 

## 4. Robustness Across Target Corpora

The degradation is consistent across all three cross-corpus conditions, demonstrating that the lack of robustness is systemic rather than corpus-specific:
* When **RAVDESS** is the target (having been excluded from training), the UAR drops to 19.64% (compared to the 49.17% achieved when RAVDESS was included in training).
* When **CREMA-D** is the target, performance reaches its absolute lowest (UAR of 16.67%, Macro-F1 of 6.14%), indicating that its acoustic or recording characteristics differ the most significantly from the other two datasets.

## 5. Conclusion

The numerical evidence strongly indicates that while Supervised Contrastive Learning improves within-domain feature separation, it is currently insufficient to overcome the massive acoustic and domain shifts present between different speech emotion corpora. The representations learned by the model are highly corpus-specific, leading to catastrophic prediction collapse when generalized to unseen datasets.
