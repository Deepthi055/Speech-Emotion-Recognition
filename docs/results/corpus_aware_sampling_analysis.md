# Corpus-Aware Sampling Analysis

This document analyzes the design, implementation, and empirical effects of the Corpus-Aware Sampling strategy introduced in this project.

## 1. Mechanism and Implementation

Similar to Speaker-Aware Sampling, **Corpus-Aware Sampling** is implemented by dynamically adjusting the contrastive weights in the Supervised Contrastive Learning (SupCon) loss function (via `compute_corpus_aware_weights` in `src/losses/contrastive_weights.py`).

The weighting scheme penalizes the model for relying on corpus-specific acoustic traits:
* **Same Emotion + Different Corpus:** Weight = `1.0` (Preferred positive pair, forces learning cross-corpus emotion features).
* **Same Emotion + Same Corpus:** Weight = `0.25` (Weaker positive pair).

The final **Proposed (Speaker + Corpus-Aware) SupCon** combines these by giving highest priority (1.0) to pairs that differ in *both* speaker and corpus, medium priority (0.5) to pairs that differ only in speaker, and the lowest priority (0.25) to same-speaker pairs.

## 2. Motivation

The strategy was designed to tackle the severe domain shift problem observed in cross-corpus emotion recognition. Datasets like IEMOCAP, RAVDESS, and CREMA-D have vastly different recording conditions, background noise profiles, and acting styles. By explicitly prioritizing cross-corpus positive pairs during training, the model is theoretically forced to learn domain-invariant emotion representations, reducing "corpus bias."

## 3. Experimental Results

Despite the logical and theoretical soundness of the approach, the actual experimental results reveal no performance changes.

* **Within-Corpus Performance (RAVDESS):** The Test Accuracy and UAR were completely identical to the Standard SupCon baseline and the Speaker-Aware variant.
* **Cross-Corpus Performance:** The performance metrics across all target corpora combinations (IEMOCAP, RAVDESS, CREMA-D) were also exactly identical between Standard SupCon, Corpus-Aware SupCon, and the combined Proposed model. 

## 4. Implications and Trade-offs

1. **Failure to Overcome Domain Shift:** The fact that Corpus-Aware Sampling yielded identical results to Standard SupCon implies that re-weighting positive pairs is an insufficient technique for overcoming the massive acoustic disparities between corpora. The domain shift is too severe for loss-reweighting alone to enforce domain invariance.
2. **Backbone Robustness:** As observed with the speaker-aware sampling, the powerful pre-trained `WavLM` backbone might be generating highly stable representations that resist being pulled into different clustering arrangements merely by altering the contrastive weight from 1.0 to 0.25.
3. **No Trade-offs Incurred:** While the strategy did not improve cross-corpus generalization, it is important to note that it also did *not degrade* within-corpus performance. There was no observed trade-off, simply an identical result, meaning it is safe to use but currently ineffective.

**Conclusion:** Corpus-Aware Sampling aims to resolve the critical generalization gap in cross-corpus emotion recognition. However, empirical results show it fails to shift the model's representations beyond what Standard SupCon already achieves, highlighting that domain shift in speech emotion recognition requires more aggressive domain adaptation or alignment techniques than simple loss re-weighting.
