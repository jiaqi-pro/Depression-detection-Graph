import scipy.io as scio
import torch
import torch.nn as nn
import os
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import pandas as pd
from dgl.nn.pytorch import GraphConv, GATConv
import numpy as np
import dgl
from nets.molecules_graph_regression.gat_net import GATNet
import datetime
from os.path import join as pjoin
import sys

sys.path.append('..')

from Preprocess.util_ccc import concordance_correlation_coefficient
from Preprocess.sending_email import sendEmail


date = datetime.datetime.today()
'''
model 构建

'''


# def build_graph(x):
#     m = x.shape[0]
#     u = []
#     v = []
#     for i in range(m):
#         for j in range(m):
#             u.append(i)
#             v.append(j)
#     g = dgl.graph((u,v))
#     return g

def build_graph(x):
    m = x.shape[0]
    u = []
    v = []
    for i in range(m):
        count = 0
        for j in range(m):
            u.append(i)
            v.append(j)
            count += 1

    g = dgl.graph((u, v))
    return g


## parameters

net_params = {
    'num_atom_type': 48,
    'num_bond_type': 2,
    'hidden_dim': 8,  ## 10-》exp2
    'out_dim': 4,
    'L': 1,
    'readout': 'mean',
    'residual': False,
    'edge_feat': True,
    'device': 'cpu',
    'pos_enc': False,
    # 'pos_enc_dim':72,
    'batch_norm': True,
    'layer_type': 'edgereprfeat',
    'in_feat_dropout': 0.0,
    'dropout': 0.0,
    'n_heads': 4
}
'''
1.将数据集读取
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch.utils.data import Dataset
import os


class SPG_Dataset(nn.Module):
    def __init__(self, file_path, file_index):
        super(SPG_Dataset, self).__init__()
        self.file_index = np.load(file_index).tolist()
        self.file_path = file_path
        self.mat_name = [str(i) + '.mat' for i in self.file_index]  ## 读取对应id，来区分训练&验证，测试数据集
        self.mat_path = [pjoin(self.file_path, i) for i in self.mat_name]

    def __getitem__(self, index):
        mat_file = self.mat_path[index]

        mat_data = scio.loadmat(mat_file)

        feature_np = mat_data['feature_2d']
        
        

        label_np = mat_data['label'][0,0].reshape(-1)
        

        feature_th = torch.from_numpy(feature_np)

        label_th = torch.from_numpy(label_np)

        return feature_th, label_th

    def __len__(self):
        return len(self.mat_path)


def ccc(x, y):
    # Concordance Correlation Coefficient
    sxy = np.sum((x - x.mean()) * (y - y.mean())) / x.shape[0]
    rhoc = 2 * sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean()) ** 2)
    return rhoc


# train_path = 'C:/matlab_file_weights/training_scripts/Northwind_DSN_Training_256/'

## change
# train_path = 'C:/matlab_file_weights/training_scripts/Training_DSN_32/'

## 文件夹保存
exp_time = str(date.strftime("%Y-%m-%d-%H:%M:%S"))

save_dir = '../AU_SPG'
exp_dir = pjoin(save_dir, exp_time)
folder_all = os.path.exists(save_dir)
folder_exp = os.path.exists(exp_dir)
if not folder_all:
    os.mkdir(save_dir)

if not folder_exp:
    os.mkdir(exp_dir)

mat_path = 'AU_spectral_mat/AU_spectral_mat/'

train_file_index = 'train_valid_index.npy'
test_file_index = 'test_index.npy'

train_dataset = SPG_Dataset(mat_path, file_index=train_file_index)
train_loader = DataLoader(dataset=train_dataset,  # 数据集
                          batch_size=1,
                          shuffle=False)  # 两个线程进行数据读取

test_dataset = SPG_Dataset(mat_path, file_index=test_file_index)
test_loader = DataLoader(dataset=test_dataset,  # 数据集
                         batch_size=1,
                         shuffle=False)  # 两个线程进行数据读取

#
model = GATNet(net_params)

# model.load_state_dict(torch.load('/hy-tmp/Code/AU_SPG/2022-09-13-11:55:50/weights/Epcoh:26_Rmse:6.334392070770264_PCC:0.25820012501176426_CCC:0.20233996897592121.pth'))


print(model)
#### exp1 net_params = {
#     'num_atom_type': 48,
#     'num_bond_type': 2,
#     'hidden_dim': 4,  ## 10-》exp2
#     'out_dim': 8,
#     'L': 1,
#     'readout': 'mean',
#     'residual': False,
#     'edge_feat': True,
#     'device': 'cpu',
#     'pos_enc': False,
#     # 'pos_enc_dim':72,
#     'batch_norm': True,
#     'layer_type': 'edgereprfeat',
#     'in_feat_dropout': 0.0,
#     'dropout': 0.2,
#     'n_heads': 2
# }





### exp2:

optim_1 = torch.optim.Adam(params=model.parameters(), lr=1e-5, betas=(0.9, 0.999))

l1_loss = nn.MSELoss()
maeloss = nn.L1Loss()

# e = torch.ones([17*17,1])
'''
训练
'''

def train(epoch,model,train_loader,test_loader):
    train_loader = train_loader
    test_loader = test_loader
    for i in train_loader:
        train_feature_tensor, train_label_tensor = i
        input_data = train_feature_tensor.long()
        input_data = input_data.squeeze(dim=0)
        input_data = input_data.float()
        label = train_label_tensor.squeeze(dim=0)
        
        label = label.float()
        # input_data = torch.relu(input_data)
        # e = torch.ones([17 * 17, 1])
        # e = self_euclidean_dist(input_data)
        # e = torch.relu(pearsonr_tensor(input_data).long())
        G = build_graph(input_data)
        # G = dgl.add_self_loop(G)
        # print(f'G:{G}')
        e = torch.ones(G.num_edges(), 1).long()
        result_0 = model(G, input_data, e)
        
        # label = label[1].reshape([1])
        result = torch.mean(result_0, dim=0)
        optim_1.zero_grad()
        
        # print(result.shape)
        loss_1 = l1_loss(result, label)
        loss_1.backward()
        optim_1.step()
 #     #
    model.eval()
    total_valid_labels = torch.Tensor()

    total_valid_predicts = torch.Tensor()
    for test_x, test_y in test_loader:
        test_x = torch.squeeze(test_x, dim=0).float()
        
        test_label = test_y.squeeze(dim=0)
        
        test_label = test_label.float()
        # test_x = torch.relu(test_x)
        # e = self_euclidean_dist(test_x)
        # e = torch.relu(pearsonr_tensor(test_x).long())
        # e = e.reshape(-1,1).long()
        G = build_graph(test_x)
        e = torch.ones(G.num_edges(), 1).long()
        predict_result = model.forward(G, test_x, e)
        ##  predict  result
        
        
        predict = predict_result.reshape(-1).to('cpu').data
        
        print(predict.shape)
       
        
        test_label = test_label.to('cpu').data
        total_valid_predicts = torch.cat((total_valid_predicts, predict_result), 0)
        ## ground truthg
        total_valid_labels = torch.cat((total_valid_labels, test_label), 0)

        ###  计算评估指标
        
 
    
    total_valid_predicts = total_valid_predicts.squeeze(dim=1)

    Metric_rmse = l1_loss(total_valid_predicts, total_valid_labels).pow(0.5).item()
    Metric_pcc = pearsonr(total_valid_predicts.detach().numpy().reshape(-1), total_valid_labels.detach().numpy().reshape(-1))[0]
    Metric_ccc = concordance_correlation_coefficient(total_valid_predicts.detach().numpy().reshape(-1), total_valid_labels.detach().numpy().reshape(-1))
    
    print(f"Epcoh:{epoch}:Rmse{Metric_rmse},Pcc:{Metric_pcc},CCC:{Metric_ccc}")
    weights_dir = exp_dir + '/weights'
    folder_weights = os.path.exists(weights_dir)

    if not folder_weights:
        os.mkdir(weights_dir)

    return Metric_rmse, Metric_pcc, Metric_ccc

best_ccc = 0
epochs = 10000


# model.load_state_dict(torch.load('/hy-tmp/Code/Resnet_SPG/2022-09-04-23:20:52/weights/Epcoh:79_Rmse:7.288352966308594_PCC:0.28816209112941604_CCC:0.2636206190325081.pth'))


'''
net_params = {
    'num_atom_type': 48,
    'num_bond_type': 2,
    'hidden_dim': 8,  ## 10-》exp2
    'out_dim': 8,
    'L': 1,
    'readout': 'mean',
    'residual': False,
    'edge_feat': True,
    'device': 'cpu',
    'pos_enc': False,
    # 'pos_enc_dim':72,
    'batch_norm': True,
    'layer_type': 'edgereprfeat',
    'in_feat_dropout': 0.0,
    'dropout': 0.0,
    'n_heads': 4
}

'''
# model.load_state_dict(torch.load('/hy-tmp/Code/Resnet_SPG/2022-09-06-01:05:39/weights/Epcoh:108_Rmse:6.981221675872803_PCC:0.24210830318209015_CCC:0.1791189916116286.pth'))

for epoch in range(epochs):

    test_rmse, test_pcc, test_ccc = train(epoch=epoch, model=model, train_loader=train_loader,test_loader=test_loader)



    weights_dir = exp_dir + '/weights'
    folder_weights = os.path.exists(weights_dir)

    if not folder_weights:
        os.mkdir(weights_dir)

    if best_ccc < test_ccc:
        torch.save(model.state_dict(), os.path.join(weights_dir,
                                                    f'Epcoh:{str(epoch)}_Rmse:{str(test_rmse)}_PCC:{str(test_pcc)}_CCC:{str(test_ccc)}.pth'))
        best_ccc = test_ccc

    if test_rmse < 6.08 and test_ccc > 0.267:
        torch.save(model.state_dict(), os.path.join(weights_dir,
                                                    f'Epcoh:{str(epoch)}_Rmse:{str(test_rmse)}_PCC:{str(test_pcc)}_CCC:{str(test_ccc)}.pth'))
        # sendEmail(content="test_rmse:" + str(test_rmse) + '  test_ccc:' + str(test_ccc))

# sendEmail(content='实验结束')
torch.save(model.state_dict(), os.path.join(weights_dir,
                                            f'Fin_Epcoh:{str(epoch)}_Rmse:{str(test_rmse)}_PCC:{str(test_pcc)}_CCC:{str(test_ccc)}.pth'))