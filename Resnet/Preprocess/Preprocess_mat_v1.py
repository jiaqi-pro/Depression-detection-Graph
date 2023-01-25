
## import 读取对应的包
import torch
import pandas as pd
import os
import scipy.io as scio
import numpy as np

#### 1. 分析训练数据、验证数据和测试数据的csv
train_datafile = '../../train_split.csv'
train_df = pd.read_csv(train_datafile)


dev_datafile = '../../dev_split.csv'
dev_df = pd.read_csv(dev_datafile)


test_datafile = '../../test_split.csv'
test_df = pd.read_csv(test_datafile)


## 开始进行dataset的保存了 label 加进去

mode = str(input('输入数据种类，train,valid,test'))

if mode ==  'train':
    df = train_df
elif mode == 'valid':
    df = dev_df
else:
    df = test_df

full_df = pd.read_csv('../../full_data.csv')
data_file = full_df.Participant_ID.values.tolist()


select_feature = 'CNN_VGG'   ##  后续可以用choice 进行选择['CNN_ResNet', 'CNN_VGG' ,]

split_len = 30## 用作分开大小

# save_dir = 'D:/DATA_Feature_Select_30_CNN_ResNet/'
save_dir = 'D:/DATA_Feature_Select_30_' + select_feature + '/'
folder =  os.path.exists(save_dir)
if not folder:
    os.mkdir(save_dir)


data_file_path = '../../DATA/' ## 数据存储路径

for idx,temp_file in enumerate(data_file):
    temp_data_file  = str(data_file[idx]) + '_P' ## 具体的文件夹

    data_example_path = data_file_path + temp_data_file + '/' + 'features/' ## 对引导里面的特征


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
        print(f'{mode}:{data_index} is ok!')

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
