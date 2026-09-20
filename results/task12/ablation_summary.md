# Task 12: Ablation Study

Results from the five full runs using seed 42 and the RAVDESS-only metadata available in this project.

| Method | Best validation UAR | Test Accuracy | Test UAR | Test Macro-F1 | Test Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| WavLM + CE Baseline | 0.5583 | 0.4727 | 0.4750 | 0.4598 | 0.4572 |
| Standard SupCon | 0.5667 | 0.4955 | 0.4917 | 0.4823 | 0.4853 |
| Speaker-Aware SupCon | 0.5667 | 0.4955 | 0.4917 | 0.4823 | 0.4853 |
| Corpus-Aware SupCon | 0.5667 | 0.4955 | 0.4917 | 0.4823 | 0.4853 |
| Speaker + Corpus-Aware SupCon | 0.5667 | 0.4955 | 0.4917 | 0.4823 | 0.4853 |

## Component Differences

| Comparison | Accuracy | UAR | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| Standard SupCon vs WavLM + CE Baseline | +0.0227 | +0.0167 | +0.0225 | +0.0280 |
| Speaker-Aware SupCon vs WavLM + CE Baseline | +0.0227 | +0.0167 | +0.0225 | +0.0280 |
| Corpus-Aware SupCon vs WavLM + CE Baseline | +0.0227 | +0.0167 | +0.0225 | +0.0280 |
| Speaker + Corpus-Aware SupCon vs WavLM + CE Baseline | +0.0227 | +0.0167 | +0.0225 | +0.0280 |
| Speaker-Aware SupCon vs Standard SupCon | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| Corpus-Aware SupCon vs Standard SupCon | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| Speaker + Corpus-Aware SupCon vs Speaker-Aware SupCon | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

Confusion matrices are preserved in `ablation_results.json` for each run.
