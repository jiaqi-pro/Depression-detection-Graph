
'''
   实现功能:构建数据集.

    输入变量:2中txt路径

    返回内容: 四个参数:Glo_img, Fac_img, label_classifier, label_regression

    维度对应的是【batch_size,32,3,112,112】,【batch_size,32,3,112,112】,【batch_size,4】,[batch_size,1]
'''
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
transform = transforms.Compose([
        transforms.Resize([112,112]),
        transforms.ToTensor()]
    )



class MTB_input(Dataset):

    def __init__(self, txt_path, transform=transform):
        self.txt_path = txt_path
        self.transform = transform
        self.image_list = []
        with open(self.txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                images = [img_path for img_path in line.split(',')]
                # print(images)
                label = images[-1].replace('\n', '')
                self.image_list.append((images[:-1], label))
    def __getitem__(self, index):

        Glo_img, labels = self._concat_images(self.image_list[index])
        labels_regression = torch.tensor([float(labels)], dtype=torch.float32)

        return Glo_img,labels_regression
    def _concat_images(self, data):
        glo_img= []
        for i in range(0,len(data[0]),1):

            global_image = Image.open(data[0][i]).convert('RGB')

            if self.transform is not None:
                global_image = self.transform(global_image)
            glo_img.append(global_image)
        glo_img = torch.stack(glo_img, dim=0)
        return glo_img, data[1]


    def __len__(self):
        return len(self.image_list)
