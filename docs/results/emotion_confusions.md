# Emotion Confusion Patterns

This document analyzes the common emotion confusions based on the predictions from the Proposed SupCon (Speaker + Corpus-Aware) model, comparing within-corpus behavior against cross-corpus generalization challenges.

## 1. Within-Corpus Confusion Patterns

In the within-corpus scenario (RAVDESS), the model shows a reasonable distribution of predictions, but distinct confusion pairs emerge:

* **High Energy Confusions:** The most significant confusions occur between high-energy or intense emotions.
  * `Happy` is frequently misclassified as `Fear` (37.50%).
  * `Angry` is frequently misclassified as `Fear` (35.00%).
* **Low Energy Confusions:** 
  * `Neutral` is often misclassified as `Sad` (30.00%) or `Happy` (25.00%).
  * `Disgust` is frequently confused with `Sad` (25.00%).
* **Symmetry and Difficult Emotions:** The confusion is often asymmetric. While `Happy` is often predicted as `Fear`, `Fear` is rarely predicted as `Happy` (instead achieving the highest class accuracy of 77.50%). Conversely, `Happy` is consistently the most difficult emotion to correctly identify (32.50% accuracy).

## 2. Cross-Corpus Confusion Patterns

The cross-corpus evaluation reveals severe domain shift issues, characterized by the model's predictions collapsing into a single or very few classes.

* **Target: IEMOCAP:** 
  * The model strongly favors the `Angry` class, achieving 76.92% accuracy on true `Angry` samples, but severely over-predicting it. 
  * 100% of `Happy` samples are misclassified as `Angry`.
  * `Neutral` samples are dispersed across `Disgust` (40%), `Angry` (20%), `Happy` (20%), and `Sad` (20%).
* **Target: RAVDESS:**
  * Similar to IEMOCAP, the model collapses towards `Angry`. 100% of `Fear`, `Happy`, and `Neutral` samples are misclassified as `Angry`.
  * Interestingly, `Sad` samples are predominantly misclassified as `Fear` (62.50%).
* **Target: CREMA-D:**
  * Complete model collapse into the `Angry` class. 
  * 100% of `Disgust`, `Fear`, `Happy`, and `Neutral` samples, along with 80% of `Sad` samples, are predicted as `Angry`. 
  * Class accuracy for `Angry` is technically 100%, but all other classes are 0%.

## 3. Key Takeaways

1. **Within-Corpus:** Emotion confusion roughly aligns with acoustic similarities (e.g., intense emotions confusing with other intense emotions like `Happy`/`Angry` -> `Fear`; subdued emotions confusing with `Sad`).
2. **Cross-Corpus:** The model suffers from catastrophic class collapse under domain shift. When faced with acoustic characteristics from an unseen corpus, the learned representations map overwhelmingly to a single class (most often `Angry`), indicating that the model has overfit to the source corpus characteristics rather than learning generalized emotion representations.
