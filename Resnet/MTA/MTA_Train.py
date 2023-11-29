import sys

import torch

sys.path.append('..')
from DepressionDataloader.Dataset import DepressDataset
from torch.utils.data import DataLoader
import pandas as pd
from Model.MTA import MTA
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

# 文件夹保存
exp_time = str(date.strftime("%Y-%m-%d-%H:%M:%S"))

save_dir = '../MTA_results'
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
# train_file = '../../train_data.txt'
# valid_file = '../../valid_data.txt'

train_file = '../train_valid.txt'
test_file = '../test_data.txt'

### dataset 准备
#### train/valid/test -> batch size
train_batch_size = 24
 
test_batch_size = 1024

#### train dataloader
train_dataset = DepressDataset(txt_path=train_file)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers = 10)

 

### test dataloader
test_dataset = DepressDataset(txt_path=test_file)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,num_workers = 10)

### model 实例化

#### 模式实例化参数

in_channel = 2048
input_channels = [256, 512, 1024, 2048]
attention_channels = 2048
outchannels = 512
model = MTA(in_channel=in_channel, input_channels=input_channels, attention_channels=attention_channels,
            outchannels=outchannels)

#### 优化器

lr = 1e-5
betas = (0.9, 0.999)
optimizer = optim.Adam(params=model.parameters(), lr=lr, betas=betas)

###  MSE 函数
loss_1 = nn.MSELoss()

### 设备选择

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

 
epochs = 200


## 定义训练函数
def train(model, device, train_loader, optimizer, epoch):
    train_loss = 0
    model.train()
    model = model.to(device)

    for train_data in tqdm(train_loader):
        train_feature, train_label = train_data
        train_feature = train_feature.permute(0, 2, 1)  ## [batch,30,2048]

        train_feature = train_feature.to(device)
        train_label = train_label.to(device)

        optimizer.zero_grad()
        predict = model(train_feature)

        loss = loss_1(predict, train_label)
        train_loss += loss

        loss.backward()
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
        valid_feature, valid_label = valid_data
        valid_feature = valid_feature.permute(0, 2, 1)  ## [batch,30,2048]

        valid_feature = valid_feature.to(device)
        valid_label = valid_label.to(device)

        predict = model(valid_feature)

        predict = predict.to('cpu').data
        
        valid_label = valid_label.to('cpu').data
        ##  predict  result
        total_valid_predicts = torch.cat((total_valid_predicts, predict), 0)
        ## ground truthg
        total_valid_labels = torch.cat((total_valid_labels, valid_label), 0)

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
    
    print(f'predict result:{full_video_predicts_list}')
    
    print(f'predict result 的 std :{torch.std(full_video_predicts_th).data}')
    
    print(f'ground truth:{full_video_groud_truth_list}')
    
    
    print(f'ground result 的 std :{torch.std(full_video_groud_truth_th).data}')
    
    print(f"Epcoh:{epoch}:Rmse{Metric_rmse},Pcc:{Metric_pcc},CCC:{Metric_ccc}")
    weights_dir = exp_dir + '/weights'
    folder_weights = os.path.exists(weights_dir)

    if not folder_weights:
        os.mkdir(weights_dir)


    return Metric_rmse, Metric_pcc, Metric_ccc




best_ccc = 0


# model.load_state_dict(torch.load("/hy-tmp/Code/MTA_results/2022-08-26-23:10:38/weights/Epcoh:19_Rmse:6.7131147384643555_PCC:0.25348365460845307_CCC:0.13499412536052927.pth"))  #model.load_state_dict()函数把加载的权重复制到模型的权重中去



for epoch in range(epochs):
    
    model = train(model, device, train_loader, optimizer, epoch)
 

    test_rmse, test_pcc, test_ccc = evaliation(epoch=epoch, model=model, device=device, loader=test_loader,
                                               file_count='../test_count.csv')

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