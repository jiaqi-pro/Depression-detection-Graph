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
### Inference & Training
#### Short-term depressive behaviour modelling
##### **MTB-DFE**

使用**MTB-DFE.py**,得到MTB-DFE模型的预测结果. 运行指令: `python MTB-DFE.py`
   - **输入**：人脸图片
   - **输出**：短期抑郁行为特征。
   - **使用**


       
      ```python
      from MTB-DFE import MTB_DFE
      # label can be a tensor of random
      input_tensor = torch.rand([2,30,3,224,224])
      label = torch.rand([2]).long()
      model = MTB_DFE()
      no_depression_feature, depression_feature, predict_result, encode_result,loss_aux,mta_output= model(input_tensor,label)
      ```
      
   - **训练**
     
        训练MTB-DFE使用如下的损失函数。 

        $Loss_{short}$ = $Loss_{NS}$ + w_{1} * $Loss_{MTA}$ + w_{2} *$Loss_{Sim}$ + w_{3} * $Loss_{DiffSim}$ + w_{4} * $Loss_{Reconstruction}$
        
        - 其中w_{1}, w_{2}, w_{3}, w_{4}是超参数，可以根据实际情况进行调整。本项目中这里设定都为1。 $Loss_{MTA}$, $Loss_{NS}$, $Loss_{Sim}$, $Loss_{DiffSim}$, $Loss_{Reconstruction}$组成。每个函数都在`Loss.py`中定义。
    

        
        ```python
        from loss import SIMSE, Reconstruction, DiffLoss, NS_Regression_loss
        
        simse = SIMSE()
        reconstruction = Reconstruction()
        diff_simse = DiffLoss()
        NS_Regression_loss = NS_Regression_loss()
        w_1, w_2, w_3,w_4 = 1,1,1,1
        
        
        indices = torch.randperm(depression_feature.size(0))
        
        # 使用索引来 shuffle 第一个维度
        shuffled_feature = depression_feature[indices]
        loss_mta = loss_aux['loss_aux']
        loss0 = NS_Regression_loss(predict_result,label)
        loss1 = simse(depression_feature,shuffled_feature)
        loss2 = diff_simse(no_depression_feature, depression_feature)
        loss3 = reconstruction(encode_result, mta_output)
        
        loss = loss0 + w_1 * loss_mta + w_2 *loss1 + w_3 * loss2 + w_4 * loss3
        print(loss)
        
        ```
        
        
   - **说明**
     - 1. $Loss_{DiffSim}$ (Difference Similarity Loss)
		
			  - 描述：$Loss_{DiffSim}$ 旨在减少抑郁特征（Depression feature）和非抑郁特征（Non-Depression feature）之间的相关性。它通过计算这两组特征的归一化版本的内积，生成一个相关矩阵，然后求这个矩阵中所有元素的平方的均值。
		
			  - 实现：使用 DiffLoss 类，该类中首先对输入的特征进行去均值化和归一化处理，然后计算得到的特征之间的内积矩阵，最后对该矩阵的平方求均值，得到最终的损失值。
		
		   2. $Loss_{Sim}$ (Similarity MSE Loss)
		
			   - 描述：$Loss_{Sim}$ 是 Depression feature 与其他 Depression feature 之间的均方误差（MSE）。这种损失函数用于评估相同特征间的相似度。
		
		   3. $Loss_{Reconstruction}$ (Reconstruction Loss)
		
			   - 描述：$Loss_{Reconstruction}$ 是抑郁特征（Depression feature）与重建特征（encode_result）之间的均方误差（MSE）。此损失用于量化重建质量，评估重建特征与原始特征的接近程度。
		
		   4. $Loss_{MTA}$ (MTA Regression Loss)
			   - 描述：$Loss_{MTA}$ 计算 MTA 的预测值与抑郁标签的均方误差（MSE）。这用于评估 MTA 预测的准确性。
		
		   5. $Loss_{NS}$ (NS Regression Loss)
		
			   - 描述：$Loss_{NS}$ 计算 NS 的预测值与抑郁标签的均方误差（MSE）。这用于评估 NS 预测的准确性。

#### Video-level depressive behaviour modelling

##### SEG

1. 将1个被试视频的短期抑郁行为特征`related_feature`进行保存，保存为mat文件形式。
2. 使用`SEG.py`,得到SPG模型的预测结果。`python SEG.py`
  - **输入**:每个视频全部的短期抑郁行为特征`depression_feature`
  - **输出**:抑郁等级。
  - **使用**


         ```
         from gat_net import GATNet
         model = GATNet(net_params)
         print(model)
         input_data = torch.rand([256,28]) # [256,28] keep the input_data same as the net_params['num_atom_type']
         G = build_graph(input_data)
         e = torch.ones(G.num_edges(),1).long()
         result = model(G,input_data,e)
         loss_SEG = SEG_loss()
         ```

    
  - **训练**


      ```python
      from loss import SEG_Regression_loss
      loss_seg = SEG_Regression_loss()
      loss = loss_seg(result,label)
      ```

      
 - **说明**
   - 1. $Loss_{SEG}$ (SEG Regression Loss)
     - 描述：$Loss_{SEG}$ 计算 SEG 的预测值与抑郁标签的均方误差（MSE）。这用于评估 SEG 预测的准确性。


##### SPG

1. 将每个被试视频的短期抑郁行为特征`related_feature`进行保存，保存为mat文件形式。
2. 运行`SpectralRepresentation.mlx`,得到Spectral Represenation Feature,保存为mat文件形式。


  - **输入**：每个视频全部的短期抑郁行为特征`related_feature`
  
        
  - **输出**：每个视频对应的 `Spectral Represenation`
      
3. 使用`SPG.py`,得到SPG模型的预测结果。`python SPG.py`


  - **输入**：每个视频全部的`Spectral Represenation`
  - **输出**：抑郁等级。
  - **使用**


   
          ```python
          from gat_net import GATNet
          model = GATNet(net_params)
          print(model)
          input_data = torch.rand([256,28]) # [256,28] keep the input_data same as the net_params['num_atom_type']
          G = build_graph(input_data)
          e = torch.ones(G.num_edges(),1).long()
          result = model(G,input_data,e)
          loss_SEG = SEG_loss()
   
          ```


  - **训练**
        

          ```python
         
            from loss import SEG_Regression_loss
            loss_seg = SEG_Regression_loss()
            loss = loss_seg(result,label)
           
          ```
 - **说明**
   - 1. $Loss_{SPG}$ (SPG Regression Loss)
     - 描述：$Loss_{SPG}$ 计算 SPG 的预测值与抑郁标签的均方误差（MSE）。这用于评估 SPG 预测的准确性。


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
