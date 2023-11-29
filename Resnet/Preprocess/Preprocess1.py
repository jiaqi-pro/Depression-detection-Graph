'''
理解所提供的数据集分布。
1. 训练数据、验证数据、测试数据的数量以及对应的标签 (已完成)
2. 读取数据的特征 （）
3. 制作相应的dataloader 以及 dataset 方便后续进行使用。--之前常用的方式是txt

'''
## import 读取对应的包
import torch
import pandas as pd
import os
import scipy.io as scio
import numpy as np

#### 1. 分析训练数据、验证数据和测试数据的csv
train_datafile = '../../train_split.csv'
train_df = pd.read_csv(train_datafile)
print(f'训练数据的数量是:{train_df.shape[0]}')
# print(train_df) ## 读取trai_split 数据集 163 * 6 ,,

'''
读取trai_split 数据集 163 * 6
列名：['Participant_ID', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)',
        'PTSD Severity']
选择PHQ socre作为我们的输出label
'''

dev_datafile = '../../dev_split.csv'
dev_df = pd.read_csv(dev_datafile)
print(f'验证数据的数量是:{dev_df.shape[0]}')

test_datafile = '../../test_split.csv'
test_df = pd.read_csv(test_datafile)
print(f'测试数据的数量是:{test_df.shape[0]}')

data_file_path = '../../DATA/'
data_file = [i for i in os.listdir(data_file_path)]
print(f'总共现在已有的数据集个数是{len(data_file)}')

'''
163 + 56 + 56 = 275 表明数据集全部都有，同时还说明各个数据集之间是没有重复出现的
'''


## 开始观察数据特征
### 选择一个样本数据集进行分析

temp_data_file  = data_file[2]
data_example_path = data_file_path + temp_data_file + '/' + 'features/'
select_feature = 'CNN_ResNet' ##  后续可以用choice 进行选择['CNN_ResNet', 'CNN_VGG' ,]

### 最终选择后的特征路径
data_index = temp_data_file.split('_')[0]  ## str:300
data_example_feature_path = data_example_path + data_index + '_' + select_feature + '.mat' ##'../../DATA/300_P/features/300_CNN_ResNet.mat'
print(f'使用案例路径是:{data_example_feature_path}')

### 开始读取数据
data = scio.loadmat(data_example_feature_path) ## return 一个字典。选择‘feature’
feature = data['feature']##array 形式
feature_th = torch.from_numpy(feature)



'''
开始进行分析。feature 为了后续创建一个csv ，方便调用dataset 和 dataloader方便 
同时更加了解特征。
了解了 目前各个数据集的对应特征长度不一致。

按照之前的做法，将每个数据集对应分开,同时特征[19458,2048] 分开成【30,2048】 。。。。[对应label]

'''
print(f'特征维度：{feature_th.shape}') ##(19458, 2048)  ##(24721, 2048)  (22766, 2048)

split_len = 30## 用作分开大小

feature_split = torch.split(feature_th, split_len, dim=0) ## 返回是一个元组 所以最后一个长度不足的话 就不保存了

## 开始进行dataset的保存了 label 加进去

mode = 'train'

if mode ==  'train':
    df = train_df
elif mode == 'valid':
    df = dev_df
else:
    df = test_df

##　读取label

label = df[df['Participant_ID'] == int(data_index)]['PHQ_Score'].values[0] ## 读取label数据
save_dir = 'D/DATA_Feature_Select_30/'
# for index,temp_feature in enumerate(feature_split):
#     if temp_feature.shape[0] < split_len:
#         break
#     else:
#         temp_dict = {}
#         temp_dict['feature'] = temp_feature
#         temp_dict['label'] = label
#         save_path = save_dir + data_index + '/' + str(index + 1) + '.npz'
#         os.mkdir(save_dir + data_index + '/')
#         np.save(save_path,temp_dict)

## 开始进行保存
##*********************************************************************************************************************
### full——data是将三个df进行合并后的
full_df = pd.read_csv('../../full_data.csv')
data_file = full_df.Participant_ID.values.tolist()


select_feature = 'CNN_ResNet'   ##  后续可以用choice 进行选择['CNN_ResNet', 'CNN_VGG' ,]

split_len = 30## 用作分开大小

# save_dir = 'D:/DATA_Feature_Select_30_CNN_ResNet/'
save_dir = '../..//DATA_Feature_Select_30_CNN_ResNet/'
folder =  os.path.exists(save_dir)
if not folder:
    os.mkdir(save_dir)



for idx,temp_file in enumerate(data_file):
    temp_data_file  = str(data_file[idx]) + '_P'
    data_example_path = data_file_path + temp_data_file + '/' + 'features/'


    ### 最终选择后的特征路径
    data_index = temp_data_file.split('_')[0]  ## str:300
    data_example_feature_path = data_example_path + data_index + '_' + select_feature + '.mat' ##'../../DATA/300_P/features/300_CNN_ResNet.mat'


    data = scio.loadmat(data_example_feature_path) ## return 一个字典。选择‘feature’
    feature = data['feature']##array 形式
    feature_th = torch.from_numpy(feature)

    feature_split = torch.split(feature_th, split_len, dim=0) ## 返回是一个元组 所以最后一个长度不足的话 就不保存了
    # ##　读取label
    # #
    label = full_df[full_df['Participant_ID'] == int(data_index)]['PHQ_Score'].values[0] ## 读取label数据
    #
    for index,temp_feature in enumerate(feature_split):
        if temp_feature.shape[0] < split_len:
            break
        else:




        #     # save_path = save_dir + data_index + '/' + str(index + 1) + '.npz'
            save_path = save_dir + data_index + '/' + str(index + 1) + '.mat'
            folder =  os.path.exists(save_dir + data_index)
            if not folder:
                os.mkdir(save_dir + data_index + '/')
            # np.save(save_path,temp_dict)
            scio.savemat(save_path, {'feature':temp_feature.numpy(),'label':label})
        print(f'{data_index} is ok!')

## 现在数据存储ok
##＃开始进行txt的制作




#### ******************************************************************************************************************************************
## 选择对应的df,选择对应的mode
#
# mode = 'test'
#
# if mode ==  'train':
#     df = train_df
# elif mode == 'valid':
#     df = dev_df
# else:
#     df = test_df
#
# df_index = df.Participant_ID.values.tolist()
# txt_data_dir = '../../DATA_Feature_Select_30_CNN_ResNet/'
# txt_data_dir_file = [txt_data_dir + str(i) + '/' for i in df_index]
#
#
# data_txt_file = '../../' + mode + '_data.txt'
# data_txt = open(data_txt_file, 'a+')
#
#
# # txt_context = ','.join(result)
# # all_txt.write(txt_context)
# # all_txt.write('\n')
# # #
# # # test_txt.write(txt_context)
# # # test_txt.write('\n')
# # start += time_len
# #
# # all_txt.close()
#
# for mat_dir in txt_data_dir_file:
#     mat_file = [mat_dir + i for i in os.listdir(mat_dir)]
#     mat_file.sort(key= lambda x: int(x.split('/')[-1].split('.')[0])) ## 排一下序
#     for file in mat_file:
#         data_txt.write(file)
#         data_txt.write('\n')
# data_txt.close()
#
#
#
#
#
#
#
