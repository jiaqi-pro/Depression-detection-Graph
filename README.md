<p align="left">
  <img src="fig/converted_logo.webp" width="200" height="200" alt="logo" title="logo">
</p>

# Video-based Depression Detection using Graph Representation
[中文](README_ZH.md)   

[English](README.md)

## Introduction
This is the official code repository for the _Two-stage Temporal Modelling Framework for Video-based Depression Recognition using Graph Representation_. For a detailed explanation, refer to our paper: [arXiv:2111.15266](https://arxiv.org/abs/2111.15266).

The project introduces a two-stage model for video-based depression detection:
- **Short-term Depressive Behavior Modeling** using the Multi-scale Temporal Behavioural Feature Extraction-Depression Feature Enhancement (MTB-DFE) model.
- **Video-level Depressive Behavior Modeling** using Spectral Encoding Graph (SEG) and Spectral Propagation Graph (SPG) models.
The models have been tested on the AVEC 2013, AVEC 2014, and AVEC 2019 datasets.

<p align="center">
  <img src="fig/pipeline.png"  alt="pipeline" title="pipeline">
</p>

## Getting Started

### Installation

#### Clone the repository
```bash
git clone https://github.com/jiaqi-pro/Depression-detection-Graph.git
pip install -r requirements.txt
```

### Dataset
#### Download the dataset

The project utilizes the AVEC 2013, AVEC 2014, and AVEC 2019 datasets for training and testing.

You need to contact the authors to gain access to the datasets.

#### Data Preprocessing

_No face extraction step is needed for the AVEC 2019 dataset as it provides feature files._

1. Use [OpenFace 2.0](https://github.com/TadasBaltrusaitis/OpenFace) with the CE-CLM extractor to obtain face images resized to 224x224.
   Directory structure:
```
 ${DATASET_ROOT_FOLDER}
└───path_to_dataset
    └───train
        └───subject_id
            └───frame_det_00_000001.bmp
            └───frame_det_00_000002.bmp
```

2. For all datasets, adjust the length of selected videos to multiples of 30 to ensure a uniform distribution of samples. Each group of 30 frames corresponds to a depression level, saved as a text file, formatted as follows:

```
./Training_face/203_1_cut_combined_aligned/frame_det_00_000001.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000002.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000003.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000004.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000005.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000006.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000007.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000008.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000009.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000010.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000011.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000012.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000013.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000014.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000015.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000016.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000017.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000018.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000019.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000020.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000021.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000022.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000023.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000024.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000025.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000026.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000027.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000028.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000029.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000030.bmp,3
```

### Training Process Overview

#### Stage One: Training the MTB-DFE Model

This stage focuses on training the Multi-scale Temporal Behavioral Feature Extraction - Depression Feature Enhancement (MTB-DFE) model, which captures and enhances short-term depression behavioral features from video sequences.

##### 1. Input and Feature Extraction
- **Video Frame Sequence to MTB**: The video frame sequence is first input to the Multi-scale Temporal Behavioral Feature Extraction (MTB) component, obtaining multi-scale spatio-temporal behavioral features $f^{MTB}$.

##### 2. Feature Enhancement and Preliminary Prediction
- **MTB Output to MTA**: The output features from MTB are fed into the Mutual Temporal Attention (MTA) module. This module enhances features highly related to depressive states, yielding the weighted feature vector $f^{MTA}$, which is then used for prediction, resulting in the MTA prediction outcomes $D^{MTA}$.

  - Calculate the MTA Loss Function $L_{MTA}$:

    $$L_{\text{MTA}} = \frac{1}{N} \sum_{n=1}^{N} \left(D_n^{\text{MTA}}-D_n\right)^{2}$$


    Where, $D_n$ represents the corresponding depression level, used to train and assess model accuracy.

##### 3. In-depth Feature Separation and Loss Calculation

- **MTA Output to NS**: The feature vectors outputted from MTA are then passed into the **Noise Separation (NS)** component. This component separates features related to depression $F_{n}^\text{Dep}$ from unrelated noise $F_{n}^\text{Non}$, and reconstructs features $F_{n}^\text{Dec}$, as well as predicting depression levels $D_{NS}$.

  **Calculate NS-related Loss**:

  - Prediction Loss Function for NS $L_{NS}$:

    $$L_{\text{NS}} = \frac{1}{N} \sum_{n=1}^{N} \left(D_n^{\text{NS}}-D_n\right)^{2}$$

    Where,

    $D_n^{\text{NS}}$ represents the predicted depression level for the $n^{th}$ sample.

    $D_n$ represents the actual depression level for the $n^{th}$ sample.

  - Calculate Similarity Function $L_{sim}$:
 


    $$L_{\text{sim}} = \frac{1}{N^2}\sum_{n=1}^{N-1} \sum_{i=n+1}^n (F_{n}^\text{Dep}-F_{i}^\text{Dep})^2$$

    Where,

    $F_{n}^\text{Dep}$ and $F_{i}^\text{Dep}$ are depression-related features extracted from the shared depression encoder.

  - Calculate Dissimilarity Function $L_{D-sim}$:

    $$L_{\text{D-sim}} = \frac{1}{N^2} \sum_{n=1}^{N} \left\|(F_{n}^{\text{Dep}})^{\top} F_{n}^{\text{Non}} \right\|_{\text{Frob}}^{2}$$

    Where,

    $F_{n}^{\text{Dep}}$ is the depression-related feature for the $n^{th}$ input.

    $F_{n}^{\text{Non}}$ is the non-depression-related feature for the $n^{th}$ input.

    $\left\| \cdot \right \| ^{2}_{\text{Frob}}$ represents the squared Frobenius norm.

  - Calculate Reconstruction Function $L_{Rec}$:

    $$L_{\text{Rec}} = \frac{1}{N \times J} \sum_{n=1}^{N} \sum_{j=1}^{J} \left(F_n^{\text{Dec}}(j) - F_n(j)\right)^{2}$$

    Where,

    $F_n(j)$: Represents the $j_{\text{th}}$ element of the $n_{\text{th}}$ input feature vector. This is the original data that was input into the model.

    $F_n^{\text{Dec}}(j)$: The $j_{\text{th}}$ element of the reconstructed feature vector for the $n_{\text{th}}$ sample, which is generated by the decoder.

    $N$: Total number of samples in the dataset.

    $J$: Number of features (or dimensions) in each feature vector.

- Integrating Losses: Combine $L_{MTA}$, $L_{NS}$, $L_{sim}$, $L_{D-sim}$, and $L_{Rec}$ to form $L_{short}$ for optimizing MTB-DFE, as follows:

    $$L_{\text{short}} =  L_{\text{NS}} + W_1 \times L_{\text{MTA}} + W_2 \times L_{\text{sim}} + W_3 \times L_{\text{D-sim}} + W_4 \times L_{\text{Rec}}$$

    Where $W_1$$, $W_2$$, $W_3$ and $W_4$ are weights indicating the importance of each loss component.

- Backpropagation: The loss $L_{\text{short}}$ is then backpropagated to optimize the parameters within the MTB-DFE model.


#### Stage Two: Training the SEG/SPG Models

<p align="center">
  <img src="fig/Graph.png" alt="Graph" title="Graph" width="600">
</p>

##### SEG (Sequential Graph Representations)

**SEG.py**: Defines the SEG model, which integrates short-term depression-related features \(D-feat_i\) from the MTB_DFE output, irrespective of length, using a graph structure for prediction.

**Input**:
- Sequence of short-term depression-related features from MTB_DFE \([D-feat_1, D-feat_2, ..., D-feat_n]\).

**Intermediate Variables**:
- SEG's Graph structure: Constructs a graph to integrate the short-term depression-related features \(D-feat_i\).

**Output**:
- SEG prediction results: SEG predicts depression levels through graph attention network (GAT) message passing and aggregation.

**Training Process**:
- **Loss Calculation**: \(Loss_{SEG}\) is the mean squared error (MSE) between SEG's predictions and the depression labels, used to assess the accuracy of SEG predictions.

```
from SEG import build_graph, GATNet
loss_SEG = SEG_loss()
 for epoch in range(num_epochs):
	for input_tensor, label in dataloader:
		G = build_graph(input_data)
		e = torch.ones(G.num_edges(),1).long()
		result = model(G,input_data,e)
		# Calulate the loss
		loss = loss_SEG(result,label)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```
##### SPG (Spectral Propagation Graph)

**SPG.py**: Defines the SPG model, which applies Discrete Fourier Transform (DFT) to extract spectral signals \(B_n\) from time-series data for constructing a graph representation of depression states.

**Preprocessing**:
- Use `SpectralRepresentation.mlx` to process the sequence of short-term depression-related features \([D-feat_1, D-feat_2, ..., D-feat_n]\) from MTB-DFE, obtaining spectral signals \(B_n\).

**Input**:
- **Spectral signals \(B_n\)**: Spectral signals derived from depression-related features through Discrete Fourier Transform (DFT), used to construct the graph model.

**Intermediate Variables**:
- **SPG's Graph structure**: Converts the spectral signals \(B_n\) into a graph structure, where each node represents a spectral feature and the edges between nodes represent the relationships between features.

**Output**:
- **SPG prediction results**: SPG predicts depression levels through graph attention network (GAT) message passing and feature aggregation.

**Training Process**:
- **Loss Calculation**: **\(Loss_{SPG}\)**: Calculates the mean squared error (MSE) between SPG's predicted depression levels and the actual depression labels, used to assess the accuracy of SPG predictions.

```
from SPG import build_graph, GATNet
loss_SPG = SPG_loss()
 for epoch in range(num_epochs):
	for input_tensor, label in dataloader:
		G = build_graph(input_data)
		e = torch.ones(G.num_edges(),1).long()
		result = model(G,input_data,e)
		# Calulate the loss
		loss = loss_SPG(result,label)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```
## Weight Downloads

Model weights and preprocessed features can be accessed via the following links:
- Baidu Cloud: [Link: https://pan.baidu.com/s/1woEGqgiaCVRepMkWOUIk9Q?pwd=5h2n Code: 5h2n]
- Google Drive: [https://drive.google.com/drive/folders/1JOvTZcVl7EXJnCkhrdAS1dRiN52HD1kj?usp=sharing]

## Considerations

Please adhere to the terms of use for the datasets and refer to the detailed guidelines to ensure the replicability and ethical conduct of the research.

## Future Work

- [ ] Convert **Spectral Representation** to a Python version.
- [ ] Provide **Inference.py**, which takes a video file and predicts the depression level directly.
- [ ] Design a GUI interface or an executable program for easier use.

## Citations and Acknowledgments

This project builds on the following research and acknowledges their contributions:
1. Valstar M, Schuller B, Smith K, et al. Avec 2013: the continuous audio/visual emotion and depression recognition challenge[C]. 2013.
2. Valstar M, Schuller B, Smith K, et al. Avec 2014: 3d dimensional affect and depression recognition challenge[C]. 2014.
3. Ringeval F, Schuller B, Valstar M, et al. AVEC 2019 workshop and challenge: state-of-mind, detecting depression with AI, and cross-cultural affect recognition[C]. 2019.
4. Yang C, Xu Y, Shi J, et al. Temporal pyramid network for action recognition[C]. 2020.
5. Song S, Jaiswal S, Shen L, et al. Spectral representation of behaviour primitives for depression analysis[J]. IEEE Transactions on Affective Computing, 2020.
