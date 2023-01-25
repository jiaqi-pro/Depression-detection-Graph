import sys

import torch

sys.path.append('..')
from DepressionDataloader.DSN_Dataset import DSN_Dataset
from torch.utils.data import DataLoader
import pandas as pd
from Model.DSN import DSN
from Preprocess.util_ccc import concordance_correlation_coefficient
import os
from os.path import join as pjoin
import torch.optim as optim
import torch.nn as nn
from scipy.stats import pearsonr
from Preprocess.sending_email import sendEmail
from tqdm import tqdm
import datetime

date = datetime.datetime.today()

## 文件夹保存
exp_time = str(date.strftime("%Y-%m-%d-%H:%M:%S"))

save_dir = '../DSN_results_v2'
exp_dir = pjoin(save_dir, exp_time)
folder_all = os.path.exists(save_dir)
folder_exp = os.path.exists(exp_dir)
if not folder_all:
    os.mkdir(save_dir)

if not folder_exp:
    os.mkdir(exp_dir)


## 实验结果的保存
def save_results(results, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, results)) + '\n')


results_header = '{}\t{}\t{}\t{}'.format('epoch', 'test_rmse', 'test_pcc', 'test_ccc')

result_txt = pjoin(exp_dir + 'result.txt')
with open(result_txt, 'w') as f:
    f.write(results_header + '\n')

### txt 数据读取 用作后续的数据集准备
file_path = '/hy-tmp/Feature_fusion_resnet_mta_save/'  ## MTA 提取特征保存路径
train_file_index = 'train_valid_index.npy'
test_file_index = 'test_index.npy'

### dataset 准备
#### train/valid/test -> batch size
train_batch_size = 1
test_batch_size = 1

#### train dataloader


train_dataset = DSN_Dataset(file_path=file_path,file_index=train_file_index)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1)

### test dataloader
test_dataset = DSN_Dataset(file_path=file_path,file_index=test_file_index)

test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)

### model 实例化

#### 模式实例化参数

model = DSN()

#### 优化器

lr = 1e-5
betas = (0.9, 0.999)
optimizer = optim.Adam(params=model.parameters(), lr=lr, betas=betas)

###  MSE 函数
loss_1 = nn.MSELoss()

### 设备选择

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loss = 0
epochs = 5000




### DSN 训练loss
# 计算编码之后与TPN提取之后的特征的相似值
class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)
        return simse


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, private_samples, shared_samples):
        batch_size = private_samples.size(0)
        private_samples_pt = private_samples.view(batch_size, -1)
        shared_samples_pt = shared_samples.view(batch_size,-1)
        private_samples_pt = torch.sub(private_samples_pt, torch.mean(private_samples_pt, dim=0, keepdim=True))
        shared_samples_pt = torch.sub(shared_samples_pt, torch.mean(shared_samples_pt, dim=0, keepdim=True))
        private_samples_pt = torch.nn.functional.normalize(private_samples_pt, p=2, dim=1, eps=1e-12)
        shared_samples_pt = torch.nn.functional.normalize(shared_samples_pt, p=2, dim=1, eps=1e-12)
        correlation_matrix_pt = torch.matmul(private_samples_pt.t(), shared_samples_pt)
        cost = torch.mean(correlation_matrix_pt.pow(2)) * 1.0
        cost = torch.where(cost > 0, cost, 0 * cost)
        #
        #
        #
        #
        # batch_size = input1.size(0)
        # input1 = input1.view(batch_size, -1)
        # input2 = input2.view(batch_size, -1)
        #
        #
        # result = torch.mul(input1,input2)
        # temp = torch.sum(result,dim=1)
        # diff_loss = torch.norm(input=temp,dim=0).pow(2)


        return cost#
    
## exp1  
alpha_weight = 0.1
beta_weight = 0.075
gamma_weight = 0.025

# ## exp2
# alpha_weight = 1
# beta_weight = 0.05
# gamma_weight = 0.005

loss_diff_f = DiffLoss()
loss_simse_f = SIMSE()

## 定义训练函数
def train(model, device, train_loader, optimizer, epoch):
    train_loss = 0
    model.train()
    model = model.to(device)

    for train_data in tqdm(train_loader):
        feature, label = train_data  ## feature: [1,rand,2048]  label:[1,rand,1]

        feature = feature.squeeze(dim=0)  ## [rand,2048]

        feature = feature.unsqueeze(dim=1)  ## [rand, 1, 2048]

        feature = feature.permute(0, 2, 1)  ## [rand, 2048 ,1 ]

        label = label.squeeze(dim=0)  ## [rand,1]

        feature = feature.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        unrealted_data, realted_data, predict_result, encode_result = model(feature)
        
        
        
        
        loss_mse = loss_1(predict_result,label)


        loss_diff_0 = loss_diff_f(realted_data, unrealted_data)
        loss_simse_0 = loss_simse_f(feature, encode_result)
        loss_all = (alpha_weight * loss_mse + beta_weight * loss_diff_0 + gamma_weight * loss_simse_0)

        train_loss += loss_all

        loss_all.backward()
        optimizer.step()

    train_loss_avg = train_loss / len(train_loader)

    return model


### 定义验证函数
def evaliation(epoch, model, device, loader, file_count):
    model = model.to(device)
    model.eval()
    total_valid_labels = torch.Tensor()

    total_valid_predicts = torch.Tensor()

    for valid_data in tqdm(loader):
        feature, label = valid_data  ## feature: [1,rand,2048]  label:[1,rand,1]

        feature = feature.squeeze(dim=0)  ## [rand,2048]

        feature = feature.unsqueeze(dim=1)  ## [rand, 1, 2048]

        feature = feature.permute(0, 2, 1)  ## [rand, 2048 ,1 ]

        label = label.squeeze(dim=0)  ## [rand,1]

        feature = feature.to(device)
        label = label.to(device)

        unrealted_data, realted_data, predict_result, encode_result = model(feature)

        predict_result = predict_result.to('cpu').data

        label = label.to('cpu').data
        ##  predict  result
        total_valid_predicts = torch.cat((total_valid_predicts, predict_result), 0)
        ## ground truthg
        total_valid_labels = torch.cat((total_valid_labels, label), 0)

    valid_num_df = pd.read_csv(file_count)
    valid_num = valid_num_df['num'].values.tolist()
    '''
    count the full video label
    '''
    full_video_predicts_list = []
    full_video_groud_truth_list = []
    start = 0
    for step in valid_num:
        full_video_preict = total_valid_predicts[start:start + step].mean().item()
        full_video_groud_truth = total_valid_labels[start:start + step].mean().item()

        full_video_predicts_list.append(full_video_preict)
        full_video_groud_truth_list.append(full_video_groud_truth)

        start += step
    full_video_predicts_th = torch.tensor(full_video_predicts_list)
    full_video_groud_truth_th = torch.tensor(full_video_groud_truth_list)

    ###  计算评估指标
    Metric_rmse = loss_1(full_video_predicts_th, full_video_groud_truth_th).pow(0.5).item()
    Metric_pcc = pearsonr(full_video_predicts_list, full_video_groud_truth_list)[0]
    Metric_ccc = concordance_correlation_coefficient(full_video_predicts_list, full_video_groud_truth_list)

    print(f"Epcoh:{epoch}:Rmse{Metric_rmse},Pcc:{Metric_pcc},CCC:{Metric_ccc}")
    weights_dir = exp_dir + '/weights'
    folder_weights = os.path.exists(weights_dir)

    if not folder_weights:
        os.mkdir(weights_dir)

    # torch.save(model.state_dict(), os.path.join(weights_dir,
    #                                             f'Epcoh:{str(epoch)}_Rmse:{str(Metric_rmse)}_PCC:{str(Metric_pcc)}_CCC:{str(Metric_ccc)}.pth'))
    return Metric_rmse, Metric_pcc, Metric_ccc



best_ccc = 0 


for epoch in range(epochs):
    model = train(model, device, train_loader, optimizer, epoch)

    test_rmse, test_pcc, test_ccc = evaliation(epoch=epoch, model=model, device=device, loader=test_loader,
                                               file_count='../../test_count.csv')

    results = [epoch, test_rmse, test_pcc, test_ccc]

    save_results(results, result_txt)

    weights_dir = exp_dir + '/weights'
    folder_weights = os.path.exists(weights_dir)

    if not folder_weights:
        os.mkdir(weights_dir)

    if best_ccc < test_ccc:
    
        torch.save(model.state_dict(), os.path.join(weights_dir,
                                        f'Epcoh:{str(epoch)}_Rmse:{str(test_rmse)}_PCC:{str(test_pcc)}_CCC:{str(test_ccc)}.pth'))
        best_ccc = test_ccc
    
    
    
    if test_rmse < 7.5 and test_ccc > 0.15:
        torch.save(model.state_dict(), os.path.join(weights_dir,
                                        f'Epcoh:{str(epoch)}_Rmse:{str(test_rmse)}_PCC:{str(test_pcc)}_CCC:{str(test_ccc)}.pth'))
        sendEmail(content="test_rmse:" + str(test_rmse) + '  test_ccc:' + str(test_ccc))

sendEmail(content= '实验结束')
torch.save(model.state_dict(), os.path.join(weights_dir,
                                        f'Fin_Epcoh:{str(epoch)}_Rmse:{str(test_rmse)}_PCC:{str(test_pcc)}_CCC:{str(test_ccc)}.pth'))