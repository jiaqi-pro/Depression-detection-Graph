from MTA import MTA
from icecream import ic
from NS import NS
import torch
from loss import SIMSE, Reconstruction, DiffLoss, NS_Regression_loss

class MTB_DFE(torch.nn.Module):
    def __init__(self):
        super(MTB_DFE, self).__init__()
        self.mta = MTA()
        self.ns = NS()
    def forward(self, x, random_label):
        _,loss_aux,mta_output = self.mta(x,random_label)
        mta_output = torch.unsqueeze(mta_output,2)
        unrealted_data, realted_data, predict_result, encode_result = self.ns(mta_output)

        return unrealted_data, realted_data, predict_result, encode_result,loss_aux,mta_output

# label can be a tensor of random
input_tensor = torch.rand([2,30,3,224,224])
label = torch.rand([2]).long()
model = MTB_DFE()
unrealted_data, realted_data, predict_result, encode_result,loss_aux,mta_output= model(input_tensor,label)

## Training loss

simse = SIMSE()
reconstruction = Reconstruction()
diff_simse = DiffLoss()
NS_Regression_loss = NS_Regression_loss()
w_1, w_2, w_3,w_4 = 1,1,1,1


indices = torch.randperm(realted_data.size(0))

# 使用索引来 shuffle 第一个维度
shuffled_data = realted_data[indices]
loss_mta = loss_aux['loss_aux']
loss0 = NS_Regression_loss(predict_result,label)
loss1 = simse(realted_data,shuffled_data)
loss2 = diff_simse(unrealted_data, realted_data)
loss3 = reconstruction(encode_result, mta_output)

ic(encode_result.shape,mta_output.shape,loss_aux)
loss = loss0 + w_1 * loss_mta + w_2 *loss1 + w_3 * loss2 + w_4 * loss3
print(loss)
