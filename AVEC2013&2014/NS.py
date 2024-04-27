import torch
import torch.nn as nn

# 计算无关特征网络
class unrelated_conv(nn.Module):
    def __init__(self):
        super(unrelated_conv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 2048, out_channels= 512, kernel_size=1)
        self.relu_1 = nn.ReLU(True)
        self.maxpool_1 = nn.MaxPool1d(kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=512,out_channels=128,kernel_size=1)
        self.relu_2 = nn.ReLU(True)
        self.maxpool_2 = nn.MaxPool1d(kernel_size=1)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=16, kernel_size=1)
        self.relu_3 = nn.ReLU(True)
        self.maxpool_3 = nn.MaxPool1d(kernel_size=1)

    def forward(self,X):
        X_dsn = X
        temp1_conv1 = self.conv1(X_dsn)
        temp1_relu = self.relu_1(temp1_conv1)
        temp1_maxpool = self.maxpool_1(temp1_relu)
        temp2_conv2 = self.conv2(temp1_maxpool)
        temp2_relu = self.relu_2(temp2_conv2)
        rough_result= self.maxpool_2(temp2_relu)

        rough_result = self.conv3(rough_result)
        rough_result = self.relu_3(rough_result)
        rough_result = self.maxpool_3(rough_result)




        return rough_result

# 计算精细特征网络
class related_conv(nn.Module):
    def __init__(self):
        super(related_conv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.relu_1 = nn.ReLU(True)
        self.maxpool_1 = nn.MaxPool1d(kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1)
        self.relu_2 = nn.ReLU(True)
        self.maxpool_2 = nn.MaxPool1d(kernel_size=1)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=16, kernel_size=1)
        self.relu_3 = nn.ReLU(True)
        self.maxpool_3 = nn.MaxPool1d(kernel_size=1)



    def forward(self,X):
        X_dsn = X
        temp1_conv1 = self.conv1(X_dsn)
        temp1_relu = self.relu_1(temp1_conv1)
        temp1_maxpool = self.maxpool_1(temp1_relu)
        temp2_conv2 = self.conv2(temp1_maxpool)
        temp2_relu = self.relu_2(temp2_conv2)
        smooth_result = self.maxpool_2(temp2_relu)

        smooth_result = self.conv3(smooth_result)
        smooth_result = self.relu_3(smooth_result)
        smooth_result = self.maxpool_3(smooth_result)


        return smooth_result
#输入数据为精细特征，预测结果
class predict_part(nn.Module):
    def __init__(self):
        super(predict_part, self).__init__()
        self.fc1 = nn.Linear(in_features=16,out_features=1)
        self.relu = nn.ReLU(True)
        self.fc2 = nn.Linear(in_features=64,out_features=1)
    def forward(self,x):
        x_in = x.reshape(x.shape[0],-1)
        x_in = self.relu(x_in)
        predict_temp1 = self.fc1(x_in)
        # predict_temp2 = self.relu(predict_temp1)
        # predict_result = self.relu(self.fc2(predict_temp2))

        return predict_temp1
#特征复原。。
class encoder_image(nn.Module):
    def __init__(self):
        super(encoder_image, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=16, out_channels=128, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=512, kernel_size=1)

        self.conv3 = nn.Conv1d(in_channels=512, out_channels=2048, kernel_size=1)

    def forward(self, smooth_feature, rough_feature):
        encode_feature = torch.add(smooth_feature, rough_feature)
        encode_1 = self.conv1(encode_feature)
        x_hat = self.conv2(encode_1)

        x_hat = self.conv3(x_hat)
        return x_hat


class NS(nn.Module):
    def __init__(self):
        super(NS, self).__init__()
        self.unrelated_conv = unrelated_conv()
        self.related_conv = related_conv()
        self.predict_part = predict_part()
        self.encoder = encoder_image()

    def forward(self,X_feature):

        unrealted_data = self.unrelated_conv(X_feature)


        realted_data = self.related_conv(X_feature)


        predict_result  = self.predict_part(realted_data)


        encode_result = self.encoder(unrealted_data, realted_data)

        return unrealted_data, realted_data, predict_result, encode_result

# model = NS()
# x = torch.rand([10,2048,1])
# unrealted_data, realted_data, predict_result, encode_result = model(x)
# print(f'unrealted_data:{unrealted_data.shape}')
# print(f'predict_Result :{predict_result}')
# print(f'related_data:{realted_data.shape}')
# print(f'encode_result:{encode_result.shape}')