import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import scipy.io as scio

class DepressDataset(Dataset):
    def __init__(self,txt_path):
        self.txt_path = txt_path

        with open(self.txt_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
            features = [i.rstrip('\n') for i in data]
        self.features = features
    def __getitem__(self, index):
        data_file  = self.features[index]

        data = scio.loadmat(data_file)  ## return 一个字典。选择‘feature’
        feature = data['feature']  ##array 形式
        feature_th = torch.from_numpy(feature)


        label = data['label']
        label_th = torch.from_numpy(label).reshape(-1).float()

        return feature_th,label_th

    def __len__(self):
        return len(self.features)

##测试部分
#
# train_dataset = DepressDataset(txt_path= '../../train_data.txt')
# train_dataloader = DataLoader(dataset=train_dataset,batch_size=10,shuffle=True)
# for data in train_dataloader:
#     feature, label = data
#     print(feature.shape)
