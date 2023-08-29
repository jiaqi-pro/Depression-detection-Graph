# 基于图表示的视频抑郁识别

[原文链接](https://arxiv.org/pdf/2111.15266.pdf)

## 概览
该代码库包含一个基于AVEC 2019数据集的抑郁程度检测研究项目。通过多时态注意力（MTA）、网络切片（NS）和谱图模型（SPG）进行预测分析。有关模块细节，请参考原文。

## 开始使用

### 数据预处理
- **目录**: `Resnet`
- **示例**: 使用ResNet特征
- **操作流程**:
  1. 筛选数据，长度为30的最大整数倍。
      > 例如：数据长度为91，实验数据选择前 91//30 * 30=90。
  2. 以`.mat`格式保存数据，每30个样本保存一个mat文件。

### 模型训练

#### 1. MTB训练
- **定义**: `Model/MTB.py`
- **训练**: `MTB/MTB_Fusion_Train.py`
- **输出**: 生成`MTB_results`文件夹，用于保存模型权重

#### 2. MTA训练
- **定义**: `Model/MTA.py`
- **训练**: `Model/MTA_Train.py`
- **输出**: 生成`MTA_results`文件夹，用于保存模型权重
  ```
  cd Resnet/MTA/
  python MTA.py
  ```

#### 3. NS训练
- **预处理**: 通过MTA转换所有受试者数据为特征向量。见 `NS/MTA_feature_extraction.ipynb`。
- **定义**: `Model/DSN.py`
- **训练**: `Resnet/NS/DSN_Train.py`
- **输出**: 生成`DSN_results_v2`文件夹

#### 4. 谱编码器
- **参考**: [谱表示](https://nottingham-repository.worktribe.com/index.php/preview/4744779/Spectral_Representation_camrea_ready.pdf)
- **特征输出**: `Resnet_spectral_mat.zip`

#### 5. 图模型
- **定义**: `graph-resnet/nets/molecules_graph_regression/gat_net.py`
- **训练**: `graph-resnet/GAT_spetral_32.py`

## 环境需求
- tqdm
- dgl
- torch

## 额外资源
- **模型权重和特征**: [百度云](https://pan.baidu.com/s/1rvWPwyeWi7wJqJsbh9jRQA)（提取码: frft）

## 注意事项
具体操作和配置信息，请参考各模块的目录。
