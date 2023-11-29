import os
import pandas as pd
from os.path import join as pjoin

## 使用train &valid 作为训练数据   test 作为测数据集

train_df = pd.read_csv('../../train_split.csv')

valid_df = pd.read_csv('../../dev_split.csv')

test_df = pd.read_csv('../../test_split.csv')

## 读取对应的样本id
train_sample_id = train_df['Participant_ID'].values.tolist()

valid_sample_id = valid_df['Participant_ID'].values.tolist()

test_sample_id = test_df['Participant_ID'].values.tolist()

print(f'The number of train dataset is :{len(train_sample_id)}')

print(f'The number of valid dataset is :{len(valid_sample_id)}')

print(f'The number of test dataset is :{len(test_sample_id)}')

## 由于vgg 的 657实验样本没有vgg特征  -- 在验证数据里
print(657 in valid_sample_id)
valid_sample_id.remove(657)
print(f'del 657 的样本id:{len(valid_sample_id)}')

## 制作txt

### train & valid list 拼接
train_valid_id = train_sample_id + valid_sample_id

mode = input('制作对应数据集的txt 文件：： train_valid or test')



if mode == 'train_valid':
    select_id = train_valid_id
else:
    select_id = test_sample_id

data_txt_file = '../../' + mode + '_vgg.txt'
data_txt = open(data_txt_file, 'a+')

mat_dir = 'D:/DATA_Feature_Select_30_CNN_VGG'
txt_data_dir_file = [mat_dir + '/' + str(i)for i in select_id ]
# mat_dir = txt_data_dir_file[0]
#
# mat_file = [mat_dir + '/' + i for i in os.listdir(mat_dir)]
#
# split_str = (mat_file[0].split("/"))
# split_str[0] = '../..'
#
# final_str = '/'.join(split_str)
# print(final_str)
for mat_dir in txt_data_dir_file:
    mat_file = [mat_dir +'/'+ i for i in os.listdir(mat_dir)]
    mat_file.sort(key= lambda x: int(x.split('/')[-1].split('.')[0])) ## 排一下序
    txt_list = []

    for file in mat_file:
        split_str = file.split("/")
        split_str[0] = '../..'
        final_str = "/".join(split_str)
        txt_list.append(final_str)


    for file in txt_list:
        data_txt.write(file)
        data_txt.write('\n')
data_txt.close()
