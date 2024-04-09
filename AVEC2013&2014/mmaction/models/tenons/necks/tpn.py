import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmcv import Config
import numpy as np

from ...registry import NECKS
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm2d as bn
from icecream import ic

class NonLocalBlock(nn.Module):
    """ NonLocalBlock Module"""

    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()

        conv_nd = nn.Conv3d

        self.in_channels = in_channels
        self.inter_channels = self.in_channels // 2

        self.ImageAfterASPP_bnRelu = nn.Sequential(
            nn.BatchNorm3d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.DepthAfterASPP_bnRelu = nn.Sequential(
            nn.BatchNorm3d(self.in_channels),
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
            selfNonLocal_fea = self.DepthAfterASPP_bnRelu(self_fea)
            mutualNonLocal_fea = self.ImageAfterASPP_bnRelu(mutual_fea)

            batch_size = selfNonLocal_fea.size(0)

            g_x = self.F_g(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
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

            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, *selfNonLocal_fea.size()[2:])
            W_y = self.F_W(y)
            z = W_y + self_fea
            return z


class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate):
        super(_DenseAsppBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=input_num, out_channels=num1, kernel_size=1)
        self.bn1 = bn(num1, momentum=0.0003)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(in_channels=num1, out_channels=num2, kernel_size=3,
                               dilation=dilation_rate, padding=dilation_rate)
        self.bn2 = bn(num2, momentum=0.0003)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, input):
        feature = self.relu1(self.bn1(self.conv1(input)))
        feature = self.relu2(self.bn2(self.conv2(feature)))

        return feature


class DASPPmodule(nn.Module):
    def __init__(self):
        super(DASPPmodule, self).__init__()
        num_features = 512
        d_feature1 = 176
        d_feature0 = num_features // 2

        self.AvgPool = nn.Sequential(
            nn.AvgPool2d([32, 32], [32, 32]),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(size=32, mode='nearest'),
        )
        self.ASPP_2 = _DenseAsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=2)

        self.ASPP_4 = _DenseAsppBlock(input_num=num_features + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=4)

        self.ASPP_8 = _DenseAsppBlock(input_num=num_features + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=8)

        self.afterASPP = nn.Sequential(
            nn.Conv3d(in_channels=512 * 2 + 176 * 3, out_channels=512, kernel_size=1))

    def forward(self, encoder_fea):
        imgAvgPool = self.AvgPool(encoder_fea)

        aspp2 = self.ASPP_2(encoder_fea)
        feature = torch.cat([aspp2, encoder_fea], dim=1)

        aspp4 = self.ASPP_4(feature)
        feature = torch.cat([aspp4, feature], dim=1)

        aspp8 = self.ASPP_8(feature)
        feature = torch.cat([aspp8, feature], dim=1)

        asppFea = torch.cat([feature, imgAvgPool], dim=1)
        AfterASPP = self.afterASPP(asppFea)

        return AfterASPP


class AttentionModel(nn.Module):
    def __init__(self):
        super(AttentionModel, self).__init__()
        self.NonLocal = NonLocalBlock(in_channels=2048)

        self.image_bn_relu = nn.Sequential(
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True))

        self.depth_bn_relu = nn.Sequential(
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True))

        self.affinityAttConv = nn.Sequential(
            nn.Conv3d(in_channels=1024, out_channels=2, kernel_size=1),
            nn.BatchNorm3d(2),
            nn.ReLU(inplace=True),
        )

    def forward(self, image_Input, depth_Input):
        image_feas = self.ImageBranchEncoder(image_Input)
        ImageAfterDASPP = self.ImageBranch_DASPP(self.ImageBranch_fc7_1(image_feas[-1]))

        depth_feas = self.DepthBranchEncoder(depth_Input)
        DepthAfterDASPP = self.DepthBranch_DASPP(self.DepthBranch_fc7_1(depth_feas[-1]))

        bs, ch, hei, wei = ImageAfterDASPP.size()

        affinityAtt = F.softmax(self.affinityAttConv(torch.cat([ImageAfterDASPP, DepthAfterDASPP], dim=1)))
        alphaD = affinityAtt[:, 0, :, :].reshape([bs, hei * wei, 1])
        alphaR = affinityAtt[:, 1, :, :].reshape([bs, hei * wei, 1])

        alphaD = alphaD.expand([bs, hei * wei, hei * wei])
        alphaR = alphaR.expand([bs, hei * wei, hei * wei])

        ImageAfterAtt1 = self.NonLocal(ImageAfterDASPP, DepthAfterDASPP, alphaD, selfImage=True)
        DepthAfterAtt1 = self.NonLocal(DepthAfterDASPP, ImageAfterDASPP, alphaR, selfImage=False)


# nonblock = NonLocalBlock(in_channels=2048)
# x = torch.rand([5, 2048, 8, 4, 4])
#
# alpha = torch.rand([5,128,128])
# z = nonblock(x,x,alpha,False)
# print(z.shape)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ConvModule(nn.Module):
    def __init__(
            self,
            inplanes,
            planes,
            kernel_size,
            stride,
            padding,
            bias=False,
            groups=1,
    ):
        super(ConvModule, self).__init__()

        self.conv = nn.Conv3d(inplanes, planes, kernel_size, stride, padding, bias=bias, groups=groups)
        # ic(self.conv)
        self.bn = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # ic(x.shape)
        out = self.relu(self.bn(self.conv(x)))
        return out


class AuxHead(nn.Module):
    def __init__(
            self,
            inplanes,
            planes,
            loss_weight=0.5
    ):
        super(AuxHead, self).__init__()
        self.convs = \
            ConvModule(inplanes, inplanes * 2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.loss_weight = loss_weight
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(inplanes * 2, planes)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv3d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)

    def forward(self, x, target=None):
        if target is None:
            return None
        loss = dict()
        x = self.convs(x)
        x = F.adaptive_avg_pool3d(x, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)

        loss['loss_aux'] = self.loss_weight * F.cross_entropy(x, target)
        return loss


class TemporalModulation(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 downsample_scale=8,
                 ):
        super(TemporalModulation, self).__init__()

        self.conv = nn.Conv3d(inplanes, planes, (3, 1, 1), (1, 1, 1), (1, 0, 0), bias=False, groups=32)
        self.pool = nn.MaxPool3d((downsample_scale, 1, 1), (downsample_scale, 1, 1), (0, 0, 0), ceil_mode=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class Upsampling(nn.Module):
    def __init__(self,
                 scale=(2, 1, 1),
                 ):
        super(Upsampling, self).__init__()
        self.scale = scale

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        return x


class Downampling(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=(3, 1, 1),
                 stride=(1, 1, 1),
                 padding=(1, 0, 0),
                 bias=False,
                 groups=1,
                 norm=False,
                 activation=False,
                 downsample_position='after',
                 downsample_scale=(1, 2, 2),
                 ):
        super(Downampling, self).__init__()

        self.conv = nn.Conv3d(inplanes, planes, kernel_size, stride, padding, bias=bias, groups=groups)
        self.norm = nn.BatchNorm3d(planes) if norm else None
        self.relu = nn.ReLU(inplace=True) if activation else None
        assert (downsample_position in ['before', 'after'])
        self.downsample_position = downsample_position
        self.pool = nn.MaxPool3d(downsample_scale, downsample_scale, (0, 0, 0), ceil_mode=True)

    def forward(self, x):
        if self.downsample_position == 'before':
            x = self.pool(x)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.downsample_position == 'after':
            x = self.pool(x)

        return x


class LevelFusion(nn.Module):
    def __init__(self,
                 in_channels=[1024, 1024],
                 mid_channels=[1024, 1024],
                 out_channels=2048,
                 ds_scales=[(1, 1, 1), (1, 1, 1)],
                 ):
        super(LevelFusion, self).__init__()
        self.ops = nn.ModuleList()
        num_ins = len(in_channels)
        for i in range(num_ins):
            op = Downampling(in_channels[i], mid_channels[i], kernel_size=(1, 1, 1), stride=(1, 1, 1),
                             padding=(0, 0, 0), bias=False, groups=32, norm=True, activation=True,
                             downsample_position='before', downsample_scale=ds_scales[i])
            self.ops.append(op)

        in_dims = np.sum(mid_channels)
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(in_dims, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        out = [self.ops[i](feature) for i, feature in enumerate(inputs)]
        out = torch.cat(out, 1)
        out = self.fusion_conv(out)
        return out


class SpatialModulation(nn.Module):
    def __init__(
            self,
            inplanes=[1024, 2048],
            planes=2048,
    ):
        super(SpatialModulation, self).__init__()

        self.spatial_modulation = nn.ModuleList()
        for i, dim in enumerate(inplanes):
            op = nn.ModuleList()
            ds_factor = planes // dim
            ds_num = int(np.log2(ds_factor))
            if ds_num < 1:
                op = Identity()
            else:
                for dsi in range(ds_num):
                    in_factor = 2 ** dsi
                    out_factor = 2 ** (dsi + 1)
                    op.append(ConvModule(dim * in_factor, dim * out_factor, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                         padding=(0, 1, 1), bias=False))
            self.spatial_modulation.append(op)

    def forward(self, inputs):
        out = []
        for i, feature in enumerate(inputs):
            if isinstance(self.spatial_modulation[i], nn.ModuleList):
                out_ = inputs[i]
                for III, op in enumerate(self.spatial_modulation[i]):
                    out_ = op(out_)
                out.append(out_)
            else:
                out.append(self.spatial_modulation[i](inputs[i]))
        return out


@NECKS.register_module
class TPN(nn.Module):

    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=256,
                 spatial_modulation_config=None,
                 temporal_modulation_config=None,
                 upsampling_config=None,
                 downsampling_config=None,
                 level_fusion_config=None,
                 aux_head_config=None,
                 ):
        super(TPN, self).__init__()
        assert isinstance(in_channels, list)
        assert isinstance(out_channels, int)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        # self.nonblock = NonLocalBlock(in_channels=1024)
        spatial_modulation_config = Config(spatial_modulation_config) if isinstance(spatial_modulation_config,
                                                                                    dict) else spatial_modulation_config
        temporal_modulation_config = Config(temporal_modulation_config) if isinstance(temporal_modulation_config,
                                                                                      dict) else temporal_modulation_config
        upsampling_config = Config(upsampling_config) if isinstance(upsampling_config, dict) else upsampling_config
        downsampling_config = Config(downsampling_config) if isinstance(downsampling_config,
                                                                        dict) else downsampling_config
        aux_head_config = Config(aux_head_config) if isinstance(aux_head_config, dict) else aux_head_config
        level_fusion_config = Config(level_fusion_config) if isinstance(level_fusion_config,
                                                                        dict) else level_fusion_config

        self.temporal_modulation_ops = nn.ModuleList()
        self.upsampling_ops = nn.ModuleList()
        self.downsampling_ops = nn.ModuleList()
        self.level_fusion_op = LevelFusion(**level_fusion_config)
        self.spatial_modulation = SpatialModulation(**spatial_modulation_config)
        for i in range(0, self.num_ins, 1):
            inplanes = in_channels[-1]
            planes = out_channels

            if temporal_modulation_config is not None:
                # overwrite the temporal_modulation_config
                temporal_modulation_config.param.downsample_scale = temporal_modulation_config.scales[i]
                temporal_modulation_config.param.inplanes = inplanes
                temporal_modulation_config.param.planes = planes
                temporal_modulation = TemporalModulation(**temporal_modulation_config.param)
                self.temporal_modulation_ops.append(temporal_modulation)

            if i < self.num_ins - 1:
                if upsampling_config is not None:
                    # overwrite the upsampling_config
                    upsampling = Upsampling(**upsampling_config)
                    self.upsampling_ops.append(upsampling)

                if downsampling_config is not None:
                    # overwrite the downsampling_config
                    downsampling_config.param.inplanes = planes
                    downsampling_config.param.planes = planes
                    downsampling_config.param.downsample_scale = downsampling_config.scales
                    downsampling = Downampling(**downsampling_config.param)
                    self.downsampling_ops.append(downsampling)

        out_dims = level_fusion_config.out_channels

        # Two pyramids
        self.level_fusion_op2 = LevelFusion(**level_fusion_config)

        self.pyramid_fusion_op = nn.Sequential(
            nn.Conv3d(out_dims * 2, 2048, 1, 1, 0, bias=False),
            nn.BatchNorm3d(2048),
            nn.ReLU(inplace=True)
        )

        # overwrite aux_head_config
        if aux_head_config is not None:
            aux_head_config.inplanes = self.in_channels[-2]
            self.aux_head = AuxHead(**aux_head_config)
        else:
            self.aux_head = None

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)

        if self.aux_head is not None:
            self.aux_head.init_weights()

    def forward(self, inputs, target=None):
        loss = None

        # Auxiliary loss
        if self.aux_head is not None:
            loss = self.aux_head(inputs[-2], target)

            # Spatial Modulation
        outs = self.spatial_modulation(inputs)


        # # Temporal Modulation
        outs = [temporal_modulation(outs[i]) for i, temporal_modulation in enumerate(self.temporal_modulation_ops)]

        temporal_modulation_outs = outs

        # Build top-down flow - upsampling operation
        if self.upsampling_ops is not None:
            for i in range(self.num_ins - 1, 0, -1):
                outs[i - 1] = outs[i - 1] + self.upsampling_ops[i - 1](outs[i])
        # Get top-down outs
        topdownouts = self.level_fusion_op2(outs)
        outs = temporal_modulation_outs

        # Build bottom-up flow - downsampling operation
        if self.downsampling_ops is not None:
            for i in range(0, self.num_ins - 1, 1):
                outs[i + 1] = outs[i + 1] + self.downsampling_ops[i](outs[i])

                # Get bottom-up outs
        outs = self.level_fusion_op(outs)

        # fuse two pyramid outs
        outs = self.pyramid_fusion_op(torch.cat([topdownouts, outs], 1))

        return outs, loss
