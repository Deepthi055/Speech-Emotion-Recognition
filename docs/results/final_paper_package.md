# Final Paper and Presentation Results Package

This index is the handoff package for the Speech Emotion Recognition paper,
project report, and presentation. It points to the committed evidence and
keeps the interpretation bounded by the saved experiment outputs.

## Executive Findings

- The within-corpus RAVDESS ablation reports 0.4955 test Accuracy, 0.4917
  test UAR, and 0.4823 test Macro-F1 for Standard SupCon and all three
  speaker/corpus-aware variants. The CE baseline reports 0.4727 Accuracy,
  0.4750 UAR, and 0.4598 Macro-F1.
- Cross-corpus generalization is weak. The proposed model reports UAR values
  of 0.1282, 0.1964, and 0.1667 for IEMOCAP, RAVDESS, and CREMA-D targets,
  respectively.
- The saved cross-corpus reports show identical metrics for Standard SupCon,
  Speaker-Aware SupCon, Corpus-Aware SupCon, and the proposed variant. The
  weighting schemes therefore have no demonstrated improvement in this
  experiment set.
- The saved transfer reports contain 32 target examples per configuration,
  including zero-support emotion classes in some runs. These results require
  that limitation to be stated alongside any paper claim.

## Evidence Map

| Purpose | Document |
| --- | --- |
| Final comparison tables | [final_comparison_tables.md](final_comparison_tables.md) |
| Confusion matrices | [confusion_matrices.md](confusion_matrices.md) |
| Cross-corpus plots | [cross_corpus_plots.md](cross_corpus_plots.md) |
| Ablation plots | [ablation_plots.md](ablation_plots.md) |
| Emotion confusion analysis | [emotion_confusions.md](emotion_confusions.md) |
| Within vs. cross-corpus analysis | [within_vs_cross_corpus.md](within_vs_cross_corpus.md) |
| Speaker-aware analysis | [speaker_aware_sampling_analysis.md](speaker_aware_sampling_analysis.md) |
| Corpus-aware analysis | [corpus_aware_sampling_analysis.md](corpus_aware_sampling_analysis.md) |
| Architecture | [final_architecture.md](final_architecture.md) |
| Preprocessing and setup | [experimental_setup.md](experimental_setup.md) |
| Hyperparameters | [hyperparameters.md](hyperparameters.md) |
| Limitations | [limitations.md](limitations.md) |

## Result Sources

The numerical source files are preserved under:

- `results/cross_corpus/cross_corpus_summary.csv`
- `results/cross_corpus/cross_corpus_summary.json`
- `results/cross_corpus/<setup>/<variant>/metrics.json`
- `results/task12/ablation_summary.csv`
- `results/task12/ablation_results.json`

The figures are under `docs/results/images/` and use the emotion order
`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`.

## Recommended Paper Interpretation

The defensible conclusion is that the frozen WavLM representation with a
joint CE and SupCon objective provides moderate performance in the reported
within-corpus RAVDESS setting, while the current saved zero-shot transfer
results do not establish robust cross-corpus emotion recognition. The
speaker-aware and corpus-aware weighting rules are theoretically motivated,
but their empirical effect is not demonstrated by the available runs.

The complete limitation statement, including the restricted target support
and the degenerate CE/IEMOCAP artifact, is documented in
[limitations.md](limitations.md).
