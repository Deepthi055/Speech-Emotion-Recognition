# Confusion Matrices

The following confusion matrices are generated from the predictions of the final proposed model (Speaker + Corpus-Aware SupCon). They are normalized by true labels to show the proportion of predictions for each class.

## 1. Within-Corpus Evaluation

Performance of the model on the RAVDESS dataset. 

![Within-Corpus (RAVDESS): Proposed SupCon](/d:/DLproject/Speech-Emotion-Recognition/docs/results/images/cm_within_corpus_proposed.png)

## 2. Cross-Corpus Evaluations

Performance of the model when trained on two source corpora and evaluated on a third unseen corpus.

### CREMA-D + RAVDESS &rarr; IEMOCAP

![Cross-Corpus (CREMA-D + RAVDESS -> IEMOCAP)](/d:/DLproject/Speech-Emotion-Recognition/docs/results/images/cm_cross_corpus_crema_ravdess_to_iemocap.png)

### CREMA-D + IEMOCAP &rarr; RAVDESS

![Cross-Corpus (CREMA-D + IEMOCAP -> RAVDESS)](/d:/DLproject/Speech-Emotion-Recognition/docs/results/images/cm_cross_corpus_crema_iemocap_to_ravdess.png)

### RAVDESS + IEMOCAP &rarr; CREMA-D

![Cross-Corpus (RAVDESS + IEMOCAP -> CREMA-D)](/d:/DLproject/Speech-Emotion-Recognition/docs/results/images/cm_cross_corpus_ravdess_iemocap_to_cremad.png)
