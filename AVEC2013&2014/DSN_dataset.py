import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
import os
import scipy.io as scio
class MyDataset(Dataset):
    def __init__(self, txt_path):
        self.txt_path = txt_path
        self.mat_path = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.mat_path.append(line.split('\n')[0])
    def __getitem__(self, index):
        mat_name = self.mat_path[index]
        # mat_full_path = os.path.join(self.data_path, mat_name)
        data  = scio.loadmat(mat_name)
        feature = torch.from_numpy(data['feature'])
        label = torch.from_numpy(data['label'])
        return feature, label

    def __len__(self):
        return len(self.mat_path)
# dataset = MyDataset('./Training_mat/')

# train_dataset = MyDataset('./no_depression.txt')
# train_loader = DataLoader(dataset=train_dataset,#数据集
#                             batch_size=5,
#                             shuffle=False)#两个线程进行数据读取
