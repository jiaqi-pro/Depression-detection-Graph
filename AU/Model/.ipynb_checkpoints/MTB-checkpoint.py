import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



# parameters
## input_channels = [256,512,1024,2048]
## attention_channels = 2048



class MTB(nn.Module):
    def __init__(self,in_channel,input_channels,attention_channels,outchannels):
        super(MTB, self).__init__()
        self.input_channels = input_channels
        ## 定义多个channels，得到多尺度特征【Batch，256，1】，【Batch，512，1】，【Batch，1024，1】，【Batch，2048，1】
        self.conv1 =  nn.ModuleList()
        for i in input_channels:
            temp_part = nn.Sequential(

                nn.Conv1d(in_channels=in_channel, out_channels=i, kernel_size=1),
                nn.BatchNorm1d(i),
                nn.ReLU(inplace=True)

            )
            self.conv1.append(temp_part)

        self.conv2 = nn.ModuleList()
        for i in input_channels:
            temp_part_2 = nn.Sequential(
                nn.Conv1d(in_channels=i, out_channels=attention_channels, kernel_size=1),
                nn.BatchNorm1d(attention_channels),
                nn.ReLU(inplace=True)

            )
            self.conv2.append(temp_part_2)
        self.conv4 = nn.ModuleList()
        for i in input_channels:
            temp_part_4 = nn.Sequential(
                nn.Conv1d(in_channels=attention_channels, out_channels=outchannels, kernel_size=1),
                nn.BatchNorm1d(outchannels),
                nn.ReLU(inplace=True)

            )
            self.conv4.append(temp_part_4)

        self.reg = nn.Sequential(
            nn.Linear(in_features=outchannels * len(self.input_channels) * 30, out_features= 4096 ),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            
            
            nn.Linear(in_features=4096 , out_features= 512 ),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),

            
            nn.Linear(512,1)
     
            
        )


    def  forward(self,x):
        outs = [in_channel(x) for in_channel in self.conv1]
        outs = [in_channel(outs[idx])for idx,in_channel in enumerate(self.conv2)]

        outs = [in_channel(outs[idx]) for idx, in_channel in enumerate(self.conv4)]

        input_feature = torch.cat(outs,dim = 1 )
        input_feature = input_feature.view(input_feature.shape[0],-1)
        outs = self.reg(input_feature)


        return outs


####　测试部分
## 现在我们得到了多个尺度的特征



# feature = torch.rand([2,2048,30])
# in_channel = 2048
# input_channels = [256,512,1024,2048]
# attention_channels = 2048
# outchannels = 1024
# model = MTB(in_channel = in_channel, input_channels=input_channels,attention_channels= attention_channels,outchannels=outchannels)
# print(model)
# 
# results = model(feature)
# #
# print(results.shape)

### 


# in_channel = 31
# input_channels = [4, 8, 16, 31]
# attention_channels = 16
# outchannels = 16
# model = MTB(in_channel=in_channel, input_channels=input_channels, attention_channels=attention_channels,
#             outchannels=outchannels)


