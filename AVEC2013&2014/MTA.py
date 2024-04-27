from MTA_dataloader import MTB_input
import torch
import torch.utils.data as Data
import os
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
from torch.utils.data import DataLoader
from mmaction.models.tenons.backbones.resnet import  ResNet
from mmcv import Config
from mmaction.models.tenons.necks.tpn_attention import TPN_attention
from mmaction.models.tenons.backbones.resnet_slow import ResNet_SlowFast
import torch
from mmcv import Config
cfg_1 = Config.fromfile('MTA_Resnet.py')

resnet_slowfast = ResNet_SlowFast(
                 depth= cfg_1.backbone.depth,
                 pretrained=None,
                 pretrained2d=True,
                 num_stages=4,
                 spatial_strides=(1, 2, 2, 2),
                 temporal_strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 out_indices=cfg_1.backbone.out_indices,
                 conv1_kernel_t=5,
                 conv1_stride_t=2,
                 pool1_kernel_t=1,
                 pool1_stride_t=2,
                 style='pytorch',
                 frozen_stages=-1,
                 inflate_freq=(0, 0, 1, 1),
                 inflate_stride=(1, 1, 1, 1),
                 inflate_style='3x1x1',
                 nonlocal_stages=(-1,),
                 nonlocal_freq=(0, 1, 1, 0),
                 nonlocal_cfg=None,
                 bn_eval=False,
                 bn_frozen=False,
                 partial_bn=False,
                 with_cp=False
)
cfg = Config.fromfile('MTA_TPN.py')

'''
2. 放入tpn提取特征
'''
tpn = TPN_attention( in_channels=cfg.necks.in_channels,
                 out_channels=1024,
                 spatial_modulation_config=cfg.necks.spatial_modulation_config,
                 temporal_modulation_config=cfg.necks.temporal_modulation_config,
                 upsampling_config=cfg.necks.upsampling_config,
                 downsampling_config=cfg.necks.downsampling_config,
                 level_fusion_config=cfg.necks.level_fusion_config,
                 aux_head_config=cfg.necks.aux_head_config,
                 )
class MTA(nn.Module):
    def __init__(self):
        super(MTA, self).__init__()
        self.tpn_resnet_slow = resnet_slowfast
        self.tpn = tpn
        self.fc1 = nn.Linear(100352,2048)
        self.dropout = nn.Dropout(p=0.1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2048,1)

    def forward(self,data,label):
        data_1  = data.permute(0,2,1,3,4)
        backbone = self.tpn_resnet_slow(data_1)
        out, loss_aux = self.tpn(backbone, target= label)
        tpn_feature = out.reshape(out.shape[0],-1)
        tpn_result_0 = self.fc1(tpn_feature)
        tpn_result_0 = self.dropout(tpn_result_0)
        tpn_result_0 = self.relu1(tpn_result_0)
        tpn_result = self.fc2(tpn_result_0)
        return tpn_result,loss_aux,tpn_result_0


# input_tensor = torch.rand([10,30,3,224,224])
# label_tensor = torch.rand([10]).long()
# print(label_tensor.shape)
# tpn_result,loss_aux,_ = MTA()(input_tensor,label_tensor)
# print(tpn_result.shape)