# Speech Command Recognition with TDNN and Data Augmentation

A keyword-spotting model that classifies short spoken commands using a **Time-Delay Neural Network (TDNN)** built from Conv1D layers over MFCC features, then improved through audio data augmentation and fine-tuning.

## Results

| Stage | Train acc. | Test acc. |
|---|---|---|
| Baseline TDNN | 90% | 82% |
| + augmented data (19,000 samples) and fine-tuning | — | **85%** |

## Pipeline

1. **Data** — Google Speech Commands v0.02 ([download](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)) plus self-recorded commands, split into one-second clips (`Converting_audio_into_smaller_audios.ipynb`).
2. **Features** — MFCCs extracted per clip.
3. **Model** — TDNN (stacked Conv1D layers over the time axis) trained in TensorFlow/Keras (`Preprocessing_data_and_Training_model.ipynb`).
4. **Augmentation** — noise addition and pitch shifting to expand the dataset to 19,000 samples.
5. **Fine-tuning** — the baseline model fine-tuned on the augmented set (`fine_tuning.ipynb`), improving generalisation on held-out data.

## Repository

```
Converting_audio_into_smaller_audios.ipynb   clip segmentation
Preprocessing_data_and_Training_model.ipynb  MFCC extraction + TDNN training
fine_tuning.ipynb                            fine-tuning on augmented data
model/, fine_tuned_model/                    saved Keras models
unprocessed_recorded_audio/                  sample self-recorded clips
*.csv                                        extracted features / metadata
```

The augmented and processed audio folders are not stored in the repo (≈18k WAV files); regenerate them by running the notebooks in order.

## Tech stack

Python · TensorFlow/Keras · librosa · NumPy · pandas
