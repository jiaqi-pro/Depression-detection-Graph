<p align="left">
  <img src="fig/converted_logo.webp" width="200" height="200" alt="logo" title="logo">
</p>

# <div align="left">基于图表示的视频抑郁识别</div>

## <div align="left">介绍</div>
This is the official code repo of _Two-stage Temporal Modelling Framework for Video-based Depression Recognition using Graph Representation_
(https://arxiv.org/abs/2111.15266)

在这个项目里，提出了二阶段模型来进行视频别的抑郁检测，分别是短期抑郁行为建模 & 视频级别抑郁行为建模。
- 短期抑郁行为建模是基于Multi-scale Temporal Behavioural Feature Extraction-Depression Feature Enhancement (MTB-DFE)模型
- 视频级别抑郁行为建模是根据Spectral Encoding Graph (SEG) | Spectral Propagation Graph (SPG)模型
详细的说明请参阅我们的论文(https://arxiv.org/abs/2111.15266)

所有的模型性能都是在AVEC 2013, AVEC 2014, AVEC 2019数据集上进行的测试。


<p align="center">
  <img src="fig/pipeline.png"  alt="pipeline" title="pipeline">
</p>

## <div align="left">Get Start</div>

### <details open> <summary>1) Installation</summary>

#### Clone the repository
```bash
git clone https://github.com/jiaqi-pro/Depression-detection-Graph.git  # clone
pip install -r requirements.txt  # install
```

### <details open> <summary>2) Dataset</summary>
#### Download the dataset

我们使用AVEC 2013、AVEC 2014以及AVEC 2019数据集进行训练和测试。

您需要联系作者以获取数据集的访问权限。


#### 数据预处理

_由于AVEC 2019提供的是特征文件，无需额外的脸部提取步骤。_

1. 使用[Openface 2.0](https://github.com/TadasBaltrusaitis/OpenFace)进行人脸提取，采用CE-CLM提取器，获得对应的人脸图片，并将图片resize成224*224的大小。
The folders are with format as:

```
 ${DATASET_ROOT_FOLDER}
└───path_to_dataset
    └───train
        └───subject_id
            └───frame_det_00_000001.bmp
            └───frame_det_00_000002.bmp
```

2. 对于所有数据集，将选定视频的长度调整为30的倍数，以确保获取均匀分布的样本组。每组30帧对应一个抑郁等级，结果将保存为文本文件，格式如下所示：
```
./Training_face/203_1_cut_combined_aligned/frame_det_00_000001.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000002.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000003.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000004.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000005.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000006.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000007.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000008.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000009.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000010.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000011.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000012.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000013.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000014.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000015.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000016.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000017.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000018.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000019.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000020.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000021.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000022.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000023.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000024.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000025.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000026.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000027.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000028.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000029.bmp,./Training_face/203_1_cut_combined_aligned/frame_det_00_000030.bmp,3
```



### 训练流程概述

#### 阶段一：训练 MTB-DFE 模型

该阶段侧重于训练多尺度时空行为特征提取-抑郁特征增强（MTB-DFE）模型，该模型从视频序列中捕获并增强短期抑郁行为特征。

##### 1. 输入与特征提取
视频帧序列输入到MTB: 视频帧序列首先被输入到Multi-scale Temporal Behavioral Feature Extraction (MTB) 组件,得到多尺度时空行为特征 $f^{MTB}$ .
##### 2. 特征增强与初步预测
MTB的输出特征被输入到MTA: 通过相互时间注意力（MTA）模块，增强与抑郁状态高度相关的特征，得到加权特征向量 $f^{MTA}$ ,并使用 $f^{MTA}$ 去进
行预测，得到MTA预测结果 $D^{MTA}$ .

- 计算MTA的损失函数 $L_{MTA}$ 

$$
L_{\text{MTA}} = \frac{1}{N} \sum_{n=1}^{N} \left(D_n^{\text{MTA}}-D_n\right)^{2}
$$


其中, $D_n$ 表示对应的抑郁等级，用于训练和评估模型的准确性。

##### 3. 深入特征分离与损失计算

- MTA的输出特征向量接着传入 **Noise Separation (NS)** 组件。该组件分离出与抑郁症相关的特征 $F_{n}^\text{Dep}$ 和与抑郁无关的噪声 $F_{n}^{\text{Non}}$，并重构特征 $R_{feat\_i}$ , 以及预测抑郁等级 $D_{NS}$

**计算NS相关损失**:

- 计算NS的预测损失函数 $L_{NS}$

$$
L_{\text{NS}} = \frac{1}{N} \sum_{n=1}^{N} \left(D_n^{\text{NS}}-D_n\right)^{2}
$$

其中，

$D_n^{\text{NS}}$表示第 $n$ 个样本经过NS 预测的抑郁等级。

$D_n$ 表示第 $n$ 个样本真实的抑郁等级


- 计算相似度函数 $L_{sim}$


$$
L_{\text{sim}} = \frac{1}{N^2}\sum_{n=1}^{N-1} \sum_{i=n+1}^n (F_{n}^\text{Dep}-F_{i}^\text{Dep})^2
$$

其中，

$F_{n}^\text{Dep}$ 与 $F_{i}^\text{Dep}$ 是从共享抑郁编码器中提取的抑郁相关特征。

索引 $n$ 和 $i$ 为相同类别但是不同个体的输入特征索引。



- 计算非相似度函数 $L_{D-sim}$

$$
L_{\text{D-sim}} = \frac{1}{N^2} \sum_{n=1}^{N} \left\|(F_{n}^{\text{Dep}})^{\top} F_{n}^{\text{Non}} \right\|_{\text{Frob}}^{2}
$$

其中，

$F_{n}^{\text{Dep}}$ 为第$n$ 个输入特征的抑郁相关特征

$F_{n}^{\text{Non}}$ 为第$n$ 个输入特征的与抑郁无关特征

$\left\| \cdot \right \| ^{2}_{F}$ 表示平方Frobenius范数，通常用于计算矩阵的元素平方和。

- 计算重构函数 $L_{Rec}$

$$
L_{\text{Rec}} = \frac{1}{N \times J} \sum_{n=1}^{N} \sum_{j=1}^{J} \left(F_n^{\text{Dec}}(j) - F_n(j)\right)^{2}
$$

其中，

$F_n(j) $: Represents the $j_{\text{th}}$ element of the $n_{\text{th}}$ input feature vector. This is the original data that was input into the model.$f^{MTA}$

$F_n^{\text{Dec}}(j)$: The $j_{\text{th}}$ element of the reconstructed feature vector for the $n_{\text{th}}$ sample, which is generated by the decoder.

$N $: Total number of samples in the dataset.

$J$: Number of features (or dimensions) in each feature vector.



将$L_{MTA}$， $L_{NS}$ ， $L_{sim}$， $L_{D-sim}$， $L_{Rec}$进行整合，得到 $L_{short}$ 来优化MTB-DFE。整合方式如下所示


$$
L_{\text{short}} =  L_{\text{NS}} + W_1 \times L_{\text{MTA}} + W_2 \times L_{\text{sim}} + W_3 \times L_{\text{D-sim}} + W_4 \times L_{\text{Rec}}$$



其中， $W_1$, $W_2$, $W_3$ and $W_4$  分别代表每项损失的重要性。

之后将 $L_{\text{short}} $进行反向传播，来优化MTB-DFE里的参数。

#### 阶段二：训练SEG / SPG 模型

  ##### SEG (SEquential Graph representation)


- **NS**的输出 $F_{n}^\text{Dep}$ 传入到 *SEG* 组件, 预测抑郁等级 $D_{SEG}$


- 计算**SEG**的预测损失函数 $L_{SEG}$

$$
L_{\text{SEG}} = \frac{1}{N} \sum_{n=1}^{N} \left(D_n^{\text{SEG}}-D_n\right)^{2}
$$

其中，

$D_n^{\text{SEG}}$表示第 $n$ 个样本经过**SEG**预测的抑郁等级。

$D_n$ 表示第 $n$ 个样本真实的抑郁等级


##### SPG (SPectral Graph representation)


- **NS**的输出 $F_{n}^\text{Dep}$ 通过 `SpectralRepresentation.mlx`, 得到频谱信号 $B_n$
- 将频谱信号 $B_n$ 输入到 *SEG* 组件, 得出预测抑郁等级 $D_{SPG}$
   
- 计算**SPG**的预测损失函数 $L_{SPG}$

$$
L_{\text{SPG}} = \frac{1}{N} \sum_{n=1}^{N} \left(D_n^{\text{SPG}}-D_n\right)^{2}
$$

其中，

$D_n^{\text{SPG}}$表示第 $n$ 个样本经过**SPG**预测的抑郁等级。

$D_n$ 表示第 $n$ 个样本真实的抑郁等级



## <div align="left">权重下载</div>
模型权重和预处理特征可以通过以下链接获取：
- 百度云链接：[链接: https://pan.baidu.com/s/1woEGqgiaCVRepMkWOUIk9Q?pwd=5h2n 提取码: 5h2n]
- Google Drive: [https://drive.google.com/drive/folders/1JOvTZcVl7EXJnCkhrdAS1dRiN52HD1kj?usp=sharing]

## <div align="left">注意事项</div>
请确保遵守数据集的使用条款，并参考详细的操作指南以保证研究的可复制性和道德性。

## <div align="left">后续工作</div>
- [ ] 1. 将**Spectral Representation**变为python版本。
- [ ] 2. 提供**Inference.py**, 输入一个video文件，可以直接预测抑郁等级。
- [ ] 3. 设计成GUI界面 或者 exe程序，方便后使用。

## <div align="left">引用和致谢</div>
本项目建立在以下研究成果之上，特此表示感谢并引用：
1. Valstar M, Schuller B, Smith K, et al. Avec 2013: the continuous audio/visual emotion and depression recognition challenge[C]. 2013.
2. Valstar M, Schuller B, Smith K, et al. Avec 2014: 3d dimensional affect and depression recognition challenge[C]. 2014.
3. Ringeval F, Schuller B, Valstar M, et al. AVEC 2019 workshop and challenge: state-of-mind, detecting depression with AI, and cross-cultural affect recognition[C]. 2019.
4. Yang C, Xu Y, Shi J, et al. Temporal pyramid network for action recognition[C]. 2020.
5. Song S, Jaiswal S, Shen L, et al. Spectral representation of behaviour primitives for depression analysis[J]. IEEE Transactions on Affective Computing, 2020.
