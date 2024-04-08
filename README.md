# 基于图表示的视频抑郁识别

## AVEC 2013

### 概览
（这里可以添加AVEC 2013特定的项目概览和介绍）

### 开始使用

#### 数据预处理
（这里可以添加AVEC 2013数据预处理的详细步骤）

#### 模型训练
（这里可以添加AVEC 2013模型训练的详细步骤）

### 环境需求
（这里可以添加AVEC 2013项目的环境需求）

### 额外资源
（这里可以添加AVEC 2013项目相关的额外资源链接）

### 注意事项
（这里可以添加AVEC 2013项目的特别注意事项）

---

## AVEC 2014

### 概览
（这里可以添加AVEC 2014特定的项目概览和介绍）

### 开始使用

#### 数据预处理
（这里可以添加AVEC 2014数据预处理的详细步骤）

#### 模型训练
（这里可以添加AVEC 2014模型训练的详细步骤）

### 环境需求
（这里可以添加AVEC 2014项目的环境需求）

### 额外资源
（这里可以添加AVEC 2014项目相关的额外资源链接）

### 注意事项
（这里可以添加AVEC 2014项目的特别注意事项）

---

## AVEC 2019

### 概览
该代码库包含一个基于AVEC 2019数据集的抑郁程度检测研究项目。通过多时态注意力（MTA）、网络切片（NS）和谱图模型（SPG）进行预测分析。有关模块细节，请参考原文。

### 开始使用

#### 数据预处理
- **目录**: `Resnet`
- **示例**: 使用ResNet特征
- **操作流程**:
  - 筛选数据，长度为30的最大整数倍。
      > 例如：数据长度为91，实验数据选择前 91//30 * 30=90。
  - 以`.mat`格式保存数据，每30个样本保存一个mat文件。

#### 模型训练

##### 1. MTB训练
- **定义**: `Model/MTB.py`
- **训练**: `MTB/MTB_Fusion_Train.py`
- **输出**: 生成`MTB_results`文件夹，用于保存模型权重

##### 2. MTA训练
- **定义**: `Model/MTA.py`
- **训练**: `Model/MTA_Train.py`
- **输出**: 生成`MTA_results`文件夹，用于保存模型权重

##### 3. NS训练
- **预处理**: 通过MTA转换所有受试者数据为特征向量。
- **定义**: `Model/DSN.py`
- **训练**: `Resnet/NS/DSN_Train.py`
- **输出**: 生成`DSN_results_v2`文件夹

##### 4. 谱编码器
- **参考**: [谱表示](https://nottingham-repository.worktribe.com/index.php/preview/4744779/Spectral_Representation_camrea_ready.pdf)
- **特征输出**: `Resnet_spectral_mat.zip`

##### 5. 图模型
- **定义**: `graph-resnet/nets/molecules_graph_regression/gat_net.py`
- **训练**: `graph-resnet/GAT_spetral_32.py`

### 环境需求
- tqdm
- dgl
- torch

### 额外资源
- **模型权重和特征**: [百度云](https://pan.baidu.com/s/1rvWPw

yeWi7wJqJsbh9jRQA)（提取码: frft）

### 注意事项
具体操作和配置信息，请参考各模块的目录。

