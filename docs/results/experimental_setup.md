# Preprocessing & Experimental Setup

This document details the actual data preparation, preprocessing, and experimental setups used across all evaluations.

## 1. Datasets and Splits

Three standard emotion recognition corpora were utilized:
* **IEMOCAP** (Interactive Emotional Dyadic Motion Capture database)
* **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
* **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset)

**Emotion Classes (6):** 
The datasets were mapped to a unified 6-class emotion space: `Angry`, `Disgust`, `Fear`, `Happy`, `Neutral`, and `Sad`. (Emotions outside this set, such as 'Calm' or 'Surprised' in RAVDESS, or 'Excited'/'Frustrated' in IEMOCAP which are mapped to 'Happy'/'Angry' respectively, were either mapped or excluded to ensure consistency).

**Speaker-Independent Split Strategy:**
To prevent data leakage, datasets were split into Train (70%), Validation (15%), and Test (15%) strictly at the **speaker level**. No speaker present in the training set appears in the validation or test sets.

## 2. Audio Preprocessing

Before feature extraction, raw audio files were standardized:
* **Sampling Rate:** All audio files were loaded as mono and explicitly resampled to a target rate of `16,000 Hz` (`16 kHz`) using `scipy.signal.resample_poly`.
* **Duration Handling / Padding:** Instead of rigidly trimming or zero-padding raw waveforms, variable lengths were natively handled by the Transformer backbone. The resulting sequence of frames was then mean-pooled to generate a single, fixed-size vector per file.

## 3. Feature Extraction and Representation

* **Backbone:** The project utilizes the `microsoft/wavlm-base` pre-trained model as the feature extractor.
* **Extraction Process:** The extraction is performed *offline* to save compute. The raw resampled waveform is passed through the frozen WavLM model.
* **Representation Dimension:** The `last_hidden_state` of WavLM is extracted and mean-pooled across the time (frame) dimension, yielding a single **768-dimensional** continuous embedding vector for each audio sample. These embeddings are saved as compressed NumPy arrays (`.npz`) and loaded directly into the training loop.

## 4. Evaluation Protocols

The project evaluates generalization under two strict protocols:

### 4.1 Within-Corpus Protocol (Ablation Study)
* **Setup:** The model is trained, validated, and tested entirely within a single corpus (RAVDESS).
* **Train:** RAVDESS Train Split (70%)
* **Val:** RAVDESS Validation Split (15%) - Used for early stopping and model selection.
* **Test:** RAVDESS Test Split (15%) - Held-out, unseen speakers.

### 4.2 Cross-Corpus Protocol (Domain Transfer)
* **Setup:** The model is trained on a combination of two source corpora and tested on a third, completely unseen corpus. This acts as a Zero-Shot domain transfer test.
* **Source Datasets:** All available data (Train, Val, Test splits combined) from the two source corpora are used for training/validation.
* **Target Dataset:** The entire target corpus acts as the test set. 
* **Tested Configurations:**
  1. Train: CREMA-D + RAVDESS &rarr; Target: IEMOCAP
  2. Train: CREMA-D + IEMOCAP &rarr; Target: RAVDESS
  3. Train: RAVDESS + IEMOCAP &rarr; Target: CREMA-D
