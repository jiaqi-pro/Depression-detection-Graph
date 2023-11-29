import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



# parameters
## input_channels = [256,512,1024,2048]
## attention_channels = 2048

class NonLocalBlock(nn.Module):
    """ NonLocalBlock Module"""

    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()

        conv_nd = nn.Conv1d

        self.in_channels = in_channels
        self.inter_channels = self.in_channels // 2

        self.ImageAfterASPP_bnRelu = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.DepthAfterASPP_bnRelu = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.R_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.R_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.R_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.R_W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)

        self.F_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.F_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.F_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.F_W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, self_fea, mutual_fea, alpha, selfImage):

        if selfImage:

            selfNonLocal_fea = self.ImageAfterASPP_bnRelu(self_fea)
            mutualNonLocal_fea = self.DepthAfterASPP_bnRelu(mutual_fea)

            batch_size = selfNonLocal_fea.size(0)
            g_x = self.R_g(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)
            # using mutual feature to generate attention
            theta_x = self.F_theta(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.F_phi(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)

            # using self feature to generate attention
            self_theta_x = self.R_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_theta_x = self_theta_x.permute(0, 2, 1)
            self_phi_x = self.R_phi(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_f = torch.matmul(self_theta_x, self_phi_x)
            # add self_f and mutual f
            f_div_C = F.softmax(alpha * f + self_f, dim=-1)
            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, *selfNonLocal_fea.size()[2:])
            W_y = self.R_W(y)
            z = W_y + self_fea
            return z

        else:
            selfNonLocal_fea = self.DepthAfterASPP_bnRelu(self_fea)## [30,2408,1]


            mutualNonLocal_fea = self.ImageAfterASPP_bnRelu(mutual_fea)##[30,2048,1]

            batch_size = selfNonLocal_fea.size(0) ##30

            g_x = self.F_g(selfNonLocal_fea).view(batch_size, self.inter_channels, -1) ##[30,1,1024]
            g_x = g_x.permute(0, 2, 1)

            # using mutual feature to generate attention
            theta_x = self.R_theta(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.R_phi(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)

            # using self feature to generate attention
            self_theta_x = self.F_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_theta_x = self_theta_x.permute(0, 2, 1)
            self_phi_x = self.F_phi(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_f = torch.matmul(self_theta_x, self_phi_x)

            # add self_f and mutual f
            f_div_C = F.softmax(alpha * f + self_f, dim=-1)
            print(g_x.shape)
            print(f_div_C.shape)
            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, *selfNonLocal_fea.size()[2:])
            W_y = self.F_W(y)
            z = W_y + self_fea
            return z

class MTA(nn.Module):
    def __init__(self,in_channel,input_channels,attention_channels,outchannels):
        super(MTA, self).__init__()
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
        ## 通过attetnion 需要将他们对其到同一个尺度 return list[Batch,2048,1] * 4
        self.conv3 = nn.Conv1d(in_channels= attention_channels *2,out_channels=2, kernel_size=1)
        self.nonblock = NonLocalBlock(in_channels= attention_channels)
        self.conv4 = nn.ModuleList()
        for i in input_channels:
            temp_part_4 = nn.Sequential(
                nn.Conv1d(in_channels=attention_channels, out_channels=outchannels, kernel_size=1),
                nn.BatchNorm1d(outchannels),
                nn.ReLU(inplace=True)

            )
            self.conv4.append(temp_part_4)

        self.reg = nn.Sequential(
            nn.Linear(in_features=outchannels * len(self.input_channels) * 30, out_features= 30 ),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(30,1)
        )

        # self.conv4 = nn.ModuleList([nn.Conv1d(in_channels= attention_channels,out_channels=outchannels,kernel_size=1) for i in range(len(input_channels))])

    def  forward(self,x):
        outs = [in_channel(x) for in_channel in self.conv1]
        outs = [in_channel(outs[idx])for idx,in_channel in enumerate(self.conv2)]

        if len(self.input_channels) == 4:
            conncat_tensor_01 = torch.cat([outs[0], outs[1]], dim=1)
            conncat_tensor_01_conv = self.conv3(conncat_tensor_01)
            alpha_01 = F.softmax(conncat_tensor_01_conv,dim=1)
            alpha_01_1 = alpha_01[:,1,:].unsqueeze(dim=2)
            feature_attention_0 = self.nonblock(outs[0], outs[1], alpha_01_1, True)

            conncat_tensor_12 = torch.cat([outs[1], outs[2]], dim=1)
            conncat_tensor_12_conv = self.conv3(conncat_tensor_12)
            alpha_12 = F.softmax(conncat_tensor_12_conv, dim=1)
            alpha_12_2 = alpha_12[:, 1, :].unsqueeze(dim=2)
            feature_attention_1 = self.nonblock(outs[1], outs[2], alpha_12_2, True)

            conncat_tensor_23 = torch.cat([outs[2], outs[3]], dim=1)
            conncat_tensor_23_conv = self.conv3(conncat_tensor_23)
            alpha_23 = F.softmax(conncat_tensor_23_conv, dim=1)
            alpha_23_3 = alpha_23[:, 1, :].unsqueeze(dim=2)
            feature_attention_2 = self.nonblock(outs[2], outs[3], alpha_23_3, True)

            conncat_tensor_30 = torch.cat([outs[3], outs[0]], dim=1)
            conncat_tensor_30_conv = self.conv3(conncat_tensor_30)
            alpha_30 = F.softmax(conncat_tensor_30_conv, dim=1)
            alpha_30_3 = alpha_30[:, 1, :].unsqueeze(dim=2)
            feature_attention_3 = self.nonblock(outs[3], outs[0], alpha_30_3, True)

            outs = [feature_attention_0, feature_attention_1, feature_attention_2, feature_attention_3]
        elif len(self.input_channels) == 3:
            conncat_tensor_01 = torch.cat([outs[0], outs[1]], dim=1)
            conncat_tensor_01_conv = self.conv3(conncat_tensor_01)
            alpha_01 = F.softmax(conncat_tensor_01_conv, dim=1)
            alpha_01_1 = alpha_01[:, 1, :].unsqueeze(dim=2)
            feature_attention_0 = self.nonblock(outs[0], outs[1], alpha_01_1, True)

            conncat_tensor_12 = torch.cat([outs[1], outs[2]], dim=1)
            conncat_tensor_12_conv = self.conv3(conncat_tensor_12)
            alpha_12 = F.softmax(conncat_tensor_12_conv, dim=1)
            alpha_12_2 = alpha_12[:, 1, :].unsqueeze(dim=2)
            feature_attention_1 = self.nonblock(outs[1], outs[2], alpha_12_2, True)

            conncat_tensor_20 = torch.cat([outs[2], outs[0]], dim=1)
            conncat_tensor_20_conv = self.conv2(conncat_tensor_20)
            alpha_20 = F.softmax(conncat_tensor_20_conv, dim=1)
            alpha_20_2 = alpha_20[:, 1, :].unsqueeze(dim=2)
            feature_attention_2 = self.nonblock(outs[2], outs[0], alpha_20_2, True)

            outs = [feature_attention_0, feature_attention_1, feature_attention_2]

        outs = [in_channel(outs[idx]) for idx, in_channel in enumerate(self.conv4)]

        input_feature = torch.cat(outs,dim = 1 )
        input_feature = input_feature.view(input_feature.shape[0],-1)
        outs = self.reg(input_feature)


        return outs


####　测试部分
## 现在我们得到了多个尺度的特征
#
# x1 = torch.rand([30,2048,1])
# x2 = torch.rand([30,2048,1])
# x3 = torch.rand([30,2048,1])
# x4 = torch.rand([30,2048,1])
# outs = [x1,x2,x3,x4]
# ### 1. concat feature
#
# conncat_tensor_01 = torch.cat([outs[0], outs[1]], dim=1)
# print(conncat_tensor_01.shape)
# conv1 = nn.Conv1d(in_channels=2048 *2 , out_channels=2, kernel_size=1)
# conncat_tensor_01_conv = conv1(conncat_tensor_01)
# alpha_01 = F.softmax(conncat_tensor_01_conv,dim=1)
# alpha_0 = alpha_01[:,0,:]
# alpha_1 = alpha_01[:,1,:].unsqueeze(dim=2)






# nonblock = NonLocalBlock(in_channels= 2048)
# temp_feature_0 = nonblock(outs[0], outs[1], alpha_1, False)



# conv1 = nn.Conv1d(in_channels=2048 * 2 ,out_channels=2,kernel_size=1)
# print(conv1(conncat_tensor_01).shape)

# feature = torch.rand([2,2048,30])
# in_channel = 2048
# input_channels = [256,512,1024,2048]
# attention_channels = 2048
# outchannels = 1024
# model = MTA(in_channel = in_channel, input_channels=input_channels,attention_channels= attention_channels,outchannels=outchannels)
# print(model)
#
# results = model(feature)
#
# print(results.shape)




