# 分析处理 au 数据
'''


当前代码功能 获得 au 特征所对应的 路径
'''
## 导入相应的包
import pandas as  pd
import os
import torch
## 选择一个实例样本进行分析

au_dir = '../../DATA/'
au_list = [au_dir + i for i in os.listdir(au_dir)]




au_path = [] ## 存储 au对印的csv 文件

for i in range(len(au_list)):

    example_dir = au_list[i]

    example_au_csv = example_dir + '/features/'

    example_id = example_dir.split('/')[-1].rstrip('_P') ## 300

    au_unique_name = '_OpenFace2.1.0_Pose_gaze_AUs.csv'

    au_example_name = example_id + au_unique_name ## 300_OpenFace2.1.0_Pose_gaze_AUs.csv


    au_file = example_au_csv + au_example_name  ## ../../DATA/300_P/features/300_OpenFace2.1.0_Pose_gaze_AUs.csv

    au_path.append(au_file)

print(au_path)
## 需要检查对应的csv中的