## 这个py文件是来统计训练特征 验证特征 测试特征对应的特征数量 方便后续计算整体的mse
import numpy as np
import pandas as pd
from os.path import join as pjoin
import os
csv_file ='../../full_data.csv'
label_csv = pd.read_csv(csv_file,index_col=0)
participant_id = label_csv['Participant_ID'].values.tolist()
data_file = '../../DATA_Feature_Select_30_CNN_ResNet'
data_index = [i for i in os.listdir(data_file)]
data_list = [data_file + '/' +  i for i in os.listdir(data_file)]
result_temp = {}
for file in data_list:
    file_index = file.split('/')[-1]
    num_mat = len(os.listdir(file))
    result_temp[file_index] = num_mat

df = pd.DataFrame.from_dict(list(result_temp.items()))
df.columns = ['file','num']
df.to_csv('../../file_count.csv')
