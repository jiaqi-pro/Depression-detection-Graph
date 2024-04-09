
<p align="left">
  <img src="converted_logo.webp" width="200" height="200" alt="logo" title="logo">
</p>

# Graph-Based Video Depression Recognition

## Overview

This project explores automatic recognition of depressive states in videos using graph representation. It utilizes the AVEC 2013, AVEC 2014, and AVEC 2019 datasets, focusing on prediction analysis with multi-temporal attention, network slicing, and spectral graph models.

[中文版 README](Readme_zh.md)

## Dataset Introduction

The AVEC series datasets, focusing on video and audio features for emotion analysis, are pivotal in our study. Each dataset provides unique insights into emotion recognition through video and audio cues.

## Data Preprocessing

Key preprocessing steps include data filtering to lengths that are multiples of 30 and format conversion to `.mat`, organizing samples in batches of 30.

## Model Training and Testing

### Training

The process involves training multiple models:
- **MTB Training**: Defined in `Model/MTB.py` and trained using `MTB/MTB_Fusion_Train.py`.
- **MTA Training**: Based on `Model/MTA.py`, executed via `Model/MTA_Train.py`.
- Additional steps for NS Training, Spectral Encoder, and Graph Model training are outlined, focusing on feature extraction and network training.

### Testing

Models are tested against a preprocessed dataset to ensure accuracy and effectiveness in real-world scenarios.

## Environment Setup

Dependencies include `tqdm`, `dgl`, and `torch`. Ensure these libraries are installed for seamless project execution.

## Additional Resources

For researchers interested in further exploration, model weights and preprocessed features are available [here](link).

## Notes

Adherence to dataset terms and detailed operation instructions are provided for reproducibility and ethical research practices.

## Citations and Acknowledgments

Our methodology builds on existing research, with specific citations and acknowledgments given to foundational works and contributors.

