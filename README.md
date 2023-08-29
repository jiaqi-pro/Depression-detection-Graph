# Depression-detection-Graph

https://arxiv.org/pdf/2111.15266.pdf

# Two-stage Temporal Modelling Framework for Video-based Depression Recognition using Graph Representation

这个资源库包含一个基于研究的项目，通过AVEC 2019数据集检测一个人的抑郁程度。在这个项目中，通过MTA、NS和 SPG进行预测分析，三个module的详细说明请见paper。

## 代码使用
操作流程以 Resnet 特征为例，AU操作基本近似。
接下来进入到“Resnet”文件夹进行讲解。
### 1. 首先对数据进行预处理。
   - 将数据进行筛选。由于每个数据的长度不一，因此每个被试特征选择“30”的最大整数倍。
> 例如：被试特征长度为91，我们选择前 91//30 * 30=90 作为我们的实验数据。

   将实验数据。保存成 'mat'形式，每 30 条样本保存一个mat，方便后面存储和调用。可参照“DATA_Feature_Select_30_CNN_ResNet” 文件夹中的样例。
   - 按照数据集合，提供的训练&验证、测试的参与id，生成一个txt，里面的内容是对应 每一个mat的路径，方便后续 torch.Dataset 的制作。
   
### 2. MTB 的训练。

1. 模型的定义：Model/ MTB.py
2. 模型的训练：MTB/MTB_Fusion_Train.py. 最终生成一个 MTB_results 的文件夹，用于保存模型模型权重。

### 3. MTA 的训练

- 模型的定义：Model/MTA.py
  ```
   python MTA.py
  ```
- 模型的训练：Model/MTA_Train.py. 最终生成一个 MTA_results 的文件夹，用于保存模型模型权重。

### 4. NS 的训练。
- 预处理部分：将每个被试对应的全部数据{90,origin_dim}-> 通过MTA 来提取特征-> {90,MTA_dim}.预处理流程见 NS/MTA_feature_extraction.ipynb
-  模型的定义：Model/ DSN.py
-  模型的训练：Resnet/NS/DSN_Train.py  最终生成一个 DSN_results_v2 的文件夹，用于保存模型模型权重。
-  NS 的特征提取：Resnet/NS/NS_feature_extraction.ipynb

### 5. Spectral Encoder:
https://nottingham-repository.worktribe.com/index.php/preview/4744779/Spectral_Representation_camrea_ready.pdf
提取后的特征为spectral的特征为：Resnet_spectral_mat.zip

### 6. Graph部分：
- 模型定义：graph-resnet/nets/molecules_graph_regression/gat_net.py
- 模型训练：graph-resnet/GAT_spetral_32.py

## Requirements
- tqdm
- dgl
- torch

## Note
各个模块的权重、NS-feature和 Spectral-mat

**百度云**
链接: https://pan.baidu.com/s/1rvWPwyeWi7wJqJsbh9jRQA 提取码: frft
