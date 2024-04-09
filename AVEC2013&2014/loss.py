import torch
import torch.nn as nn
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



        return cost

# 计算编码之后与TPN提取之后的特征的相似值
class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)
        return simse
