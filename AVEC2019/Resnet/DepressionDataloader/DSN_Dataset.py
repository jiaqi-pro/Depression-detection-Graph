import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import os
from os.path import join as pjoin
import scipy.io as scio


class DSN_Dataset(nn.Module):
    def __init__(self, file_path, file_index):
        super(DSN_Dataset, self).__init__()
        self.file_index = np.load(file_index).tolist()
        self.file_path = file_path
        self.mat_name = [str(i) + '.mat' for i in self.file_index]  ## 读取对应id，来区分训练&验证，测试数据集
        self.mat_path = [pjoin(self.file_path, i) for i in self.mat_name]

    def __getitem__(self, index):
        mat_file = self.mat_path[index]

        mat_data = scio.loadmat(mat_file)

        feature_np = mat_data['feature']

        label_np = mat_data['check_label']

        feature_th = torch.from_numpy(feature_np)

        label_th = torch.from_numpy(label_np)

        return feature_th, label_th

    def __len__(self):
        return len(self.mat_path)


# Train_Dsn_dataset = DSN_Dataset(file_path='/hy-tmp/Feature_fusion_mta_save/',
#                                 file_index='/hy-tmp/train_valid_index.npy')

# Dsn_dataloader = DataLoader(Train_Dsn_dataset, shuffle=True, batch_size=1)

# for data in Dsn_dataloader:
#     feature, label = data  ## feature: [1,rand,2048]  label:[1,rand,1]

#     feature = feature.squeeze(dim=0)  ## [rand,2048]

#     feature = feature.unsqueeze(dim=1)  ## [rand, 1, 2048]

#     feature = feature.permute(0, 2, 1)  ## [rand, 2048 ,1 ]

#     label = label.squeeze(dim=0)  ## [rand,1]

#     print(f'feature:{feature.shape}')  ## check. :: [rand,2048,1]

#     print(f'label:{label.shape}')  ## check :: [rand,1]

#     break
