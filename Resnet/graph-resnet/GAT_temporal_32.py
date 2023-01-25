import scipy.io as scio
import torch
import torch.nn as nn
import os
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import pandas as pd
from dgl.nn.pytorch import GraphConv,GATConv
import numpy as np
import dgl
from nets.molecules_graph_regression.gated_gcn_net import  GatedGCNNet
from nets.molecules_graph_regression.gat_net import GATNet


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
        for j in range(i + 1, m, 1):
            u.append(i)
            v.append(j)
            count += 1
            if count == 5:
                if i + 10 <m:
                    u.append(i)
                    v.append(i+10)
                break
    g = dgl.graph((u,v))
    return g




net_params = {
    'num_atom_type': 32,
    'num_bond_type':2,
    'hidden_dim': 16,## 10-》exp2
    'out_dim':26,
    'L':1,
    'readout':'mean',
    'residual': False,
    'edge_feat': True,
    'device': 'cpu',
    'pos_enc':False,
    # 'pos_enc_dim':72,
    'batch_norm':False,
    'layer_type':'edgereprfeat',
    'in_feat_dropout':0.0,
    'dropout':0.0,
    'n_heads' : 6
}
'''
1.将数据集读取
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch.utils.data import Dataset
import os


class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.mat_path = os.listdir(self.data_path)

    def __getitem__(self, index):
        mat_name = self.mat_path[index]
        mat_full_path = os.path.join(self.data_path, mat_name)
        data  = scio.loadmat(mat_full_path)
        feature = torch.from_numpy(data['feature'])
        label = torch.from_numpy(data['label'])
        return feature, label

    def __len__(self):
        return len(self.mat_path)


def ccc(x,y):
    #Concordance Correlation Coefficient
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc


train_path = 'D:/paper_reverge_plus/after-DSN_Training_mat_0815_32/'
train_dataset = MyDataset(train_path)
train_loader = DataLoader(dataset=train_dataset,#数据集
                            batch_size=1,
                            shuffle=False)#两个线程进行数据读取

test_path = 'D:/paper_reverge_plus/after-DSN_Development_mat_0815_32/'
test_dataset = MyDataset(test_path)
test_loader = DataLoader(dataset=test_dataset,#数据集
                            batch_size=1,
                            shuffle=False)#两个线程进行数据读取
#
model = GATNet(net_params)
print(model)
# model.load_state_dict(torch.load('E:/weights/GAT_spetral_32/GATtensor(7.6631)_0.7673120364750382_0805_1.pth'))
#
# optim_1 = torch.optim.Adam(params=model.parameters(),lr=1e-3,betas=(0.9,0.999))  ##exp2
optim_1 = torch.optim.Adam(params=model.parameters(),lr=1e-3,betas=(0.9,0.999))

l1_loss = nn.MSELoss()
maeloss = nn.L1Loss()

# e = torch.ones([17*17,1])
'''
训练
'''
def train():
    for epoch in range(5000):
        for i in train_loader:
            train_feature_tensor, train_label_tensor = i
            input_data = train_feature_tensor.long()
            input_data = input_data.squeeze(dim=0)
            input_data = input_data.float()
            label = train_label_tensor.squeeze(dim=0)
            label = label[0,:]
            label = label.float()
            # print(label.shape)
            # input_data = torch.relu(input_data)
            # e = torch.ones([17 * 17, 1])
            # e = self_euclidean_dist(input_data)
            # e = torch.relu(pearsonr_tensor(input_data).long())
            G = build_graph(input_data)
            G = dgl.add_self_loop(G)
            # print(f'G:{G}')
            e = torch.ones(G.num_edges(),1).long()
            result_0 = model(G,input_data,e)
            label = label.reshape([1])
            result = torch.mean(result_0,dim=0)
            optim_1.zero_grad()
            loss_1 = l1_loss(result, label)
            loss_1.backward()
            optim_1.step()
            # break
    #
        predict_result = []
        true_result = []
        for test_x, test_y in test_loader:
            test_x = torch.squeeze(test_x,dim=0).float()
            test_y = test_y.squeeze(dim=0)
            test_y = test_y[0,:]
            # test_x = torch.relu(test_x)
            # e = self_euclidean_dist(test_x)
            # e = torch.relu(pearsonr_tensor(test_x).long())
            # e = e.reshape(-1,1).long()
            G = build_graph(test_x)
            e = torch.ones(G.num_edges(), 1).long()
            test_result = model.forward(G,test_x,e)
            temp_data = [i.item() for i in torch.mean(test_result,dim=0)]
            predict_result.extend(temp_data)
            true_result.extend([i.item() for i in test_y])
        predict_result_tensor = torch.Tensor(predict_result)
        true_result_tensor = torch.Tensor(true_result)
        rmse = l1_loss(predict_result_tensor,true_result_tensor)
        print(f"Epoch:{epoch}     RMSE:{rmse.pow(0.5)}")
        pcc = pearsonr(predict_result, true_result)
        print(f'pcc:{pcc[0]}')
        if rmse.pow(0.5)<7.8:
            print('*'*100)
            print(f"RMSE:{rmse.pow(0.5)}")
            pcc = pearsonr(predict_result,true_result)
            print(f'pcc:{pcc[0]}')
            save_pth = 'E:/weights/GAT_temporal_32/'
            torch.save(model.state_dict(),
                           os.path.join(save_pth, f'GAT{str(rmse.pow(0.5))}_{str(pcc[0])}_0805_1.pth'))
def ccc(x,y):
    #Concordance Correlation Coefficient
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc

def eval():
    # model.load_state_dict(torch.load('./weights/exp3_1/cfstensor(7.9008)_0.7270572537644348_0805_1.pth'))
    model.load_state_dict(torch.load('./weights1/GAT_temporal_32/GATtensor(7.7242)_0.7402907136654591_0805_1.pth'))
    #
    predict_result = []
    true_result = []
    for test_x, test_y in test_loader:
        test_x = torch.squeeze(test_x, dim=0).float()
        test_y = test_y.squeeze(dim=0)
        test_y = test_y[0, :]
        # test_x = torch.relu(test_x)
        # e = self_euclidean_dist(test_x)
        # e = torch.relu(pearsonr_tensor(test_x).long())
        # e = e.reshape(-1,1).long()
        G = build_graph(test_x)
        e = torch.ones(G.num_edges(), 1).long()
        test_result = model.forward(G, test_x, e)
        temp_data = [i.item() for i in torch.mean(test_result, dim=0)]
        predict_result.extend(temp_data)
        true_result.extend([i.item() for i in test_y])
    predict_result_tensor = torch.Tensor(predict_result)
    true_result_tensor = torch.Tensor(true_result)
    rmse = l1_loss(predict_result_tensor, true_result_tensor)
    print(f"RMSE:{rmse.pow(0.5)}")
    pcc = pearsonr(predict_result, true_result)
    print(f'pcc:{pcc[0]}')
    ccc_1 = ccc(np.array(predict_result),np.array(true_result))
    print(f'ccc:{ccc_1}')
    mae = maeloss(predict_result_tensor,true_result_tensor)
    print(f'mae:{mae}')
    print(f'predict result is {predict_result}')
    print(f'true result is {true_result}')

eval()