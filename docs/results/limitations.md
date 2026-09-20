# Study Limitations

This section records the limitations of the completed Speech Emotion
Recognition study. The statements below are separated into limitations
demonstrated by the saved experiments, methodological constraints, and
reasonable directions for future work.

## 1. Limitations Demonstrated by the Experiments

### 1.1 Weak cross-corpus generalization

The saved cross-corpus results show a substantial transfer failure. For the
proposed Speaker + Corpus-Aware SupCon model, test UAR is 0.1282 when IEMOCAP
is the target, 0.1964 when RAVDESS is the target, and 0.1667 when CREMA-D is
the target. With six emotion classes, uniform random performance is
approximately 0.1667 UAR. These results therefore indicate performance at or
below chance-level class-balanced recall under the reported domain-transfer
conditions.

The confusion analyses show that this is associated with prediction collapse:
the model frequently over-predicts `angry`, while several other emotions have
zero or near-zero recall. Accuracy is consequently misleading in some
cross-corpus cases. For example, the proposed model obtains 0.6250 accuracy
but only 0.1282 UAR on the IEMOCAP-target configuration.

### 1.2 Limited observed benefit from the proposed weighting schemes

The reported within-corpus ablation gives identical test results for Standard
SupCon, Speaker-Aware SupCon, Corpus-Aware SupCon, and the proposed combined
variant: Accuracy 0.4955, UAR 0.4917, and Macro-F1 0.4823. The same numerical
ties are reported for all three cross-corpus target configurations. Thus, the
current experiments do not demonstrate an empirical improvement from the
speaker- or corpus-aware weighting rules over Standard SupCon.

### 1.3 Emotion-specific errors

Within the RAVDESS evaluation, `happy` is reported as the most difficult
emotion, while `fear` is recognized more reliably. The documented confusion
patterns include asymmetric confusions between `happy` and `fear`, and
between `neutral` and `sad`. Under cross-corpus transfer, the errors become
more severe, with multiple target emotions being mapped predominantly to
`angry`. The model therefore does not provide uniform performance across the
six classes.

### 1.4 Degenerate CE baseline artifact for one transfer direction

The saved CE result for CREMA-D + RAVDESS -> IEMOCAP reports zero Accuracy,
UAR, and Macro-F1, and its classification report has zero support for several
classes. This is a degenerate or failed evaluation artifact rather than
evidence that CE universally achieves zero performance. It should be retained
for traceability but interpreted separately from valid comparative results.

## 2. Methodological Constraints

### 2.1 Restricted corpus and speaker coverage

The study uses only three corpora: IEMOCAP, CREMA-D, and RAVDESS. Their
recording conditions, acted or elicited speech protocols, speaker populations,
and annotation procedures do not represent the full diversity of natural
speech. The available speaker counts also differ substantially between
corpora, so conclusions should not be generalized to unseen languages,
accents, demographic groups, or spontaneous conversational settings.

### 2.2 Label harmonization and annotation mismatch

The experiments map all data to six classes: `angry`, `disgust`, `fear`,
`happy`, `neutral`, and `sad`. IEMOCAP `frustrated` is mapped to `angry` and
`excited` is mapped to `happy`; RAVDESS emotions outside this common label
space are excluded. These transformations improve label compatibility but
also remove distinctions and may merge states that are not equivalent across
corpora. Differences in annotation practice and acted intensity remain after
harmonization.

### 2.3 Restricted evaluation support in the saved transfer reports

The saved cross-corpus JSON reports contain 32 target examples per
configuration, and some target emotion classes have zero support. This makes
Macro-F1 and UAR estimates unstable and limits the reliability of class-wise
comparisons. The reported setup describes full target-corpus evaluation, so
the difference between that protocol description and the saved supports must
be resolved before treating the cross-corpus numbers as definitive final
estimates.

### 2.4 Frozen, mean-pooled representation

The final pipeline uses a frozen `microsoft/wavlm-base` backbone and mean-pools
its frame-level hidden states into a 768-dimensional embedding. This reduces
computational cost and makes the experiments reproducible, but it prevents
end-to-end adaptation of the acoustic representation to the emotion task and
may discard temporal information relevant to prosody and expression.

### 2.5 Single-seed and limited ablation evidence

The documented ablation uses seed 42. The absence of repeated seeds, variance
estimates, and statistical significance testing limits confidence in small
differences such as the improvement of Standard SupCon over the CE baseline.
The corpus-aware ablation is also not independently interpretable in a
single-corpus training condition because corpus identity does not vary within
the training data.

### 2.6 Computational constraints

The experiments use offline feature extraction and a frozen backbone to make
the work feasible under CPU and limited-resource conditions. This constraint
limits exploration of end-to-end fine-tuning, larger backbones, broader
hyperparameter searches, multiple random seeds, and more comprehensive
domain-adaptation methods.

## 3. Future Improvement Areas

The following are reasonable improvements motivated by the observed results,
not claims about experiments already completed:

- Re-run cross-corpus evaluation on the complete target test sets and verify
  per-class support before reporting final transfer metrics.
- Repeat each configuration across multiple seeds and report confidence
  intervals or standard deviations.
- Investigate domain adaptation or domain-invariant representation learning
  beyond positive-pair reweighting.
- Compare temporal pooling and end-to-end fine-tuning against the frozen,
  mean-pooled baseline.
- Add corpora with broader language, accent, speaker, and spontaneous-speech
  diversity.
- Preserve finer-grained emotion labels where the source annotation supports
  them, or report the consequences of each label-mapping decision.

## 4. Scope of Interpretation

The study supports the conclusion that the frozen WavLM plus SupCon pipeline
can achieve moderate within-corpus performance on the reported RAVDESS split,
but the current saved transfer experiments do not establish robust
cross-corpus emotion recognition. The weighting variants are methodologically
motivated, yet their benefit is not demonstrated by the available results.
Claims beyond these settings should therefore be treated as hypotheses for
future investigation rather than established findings.
