import os
import torch
import torchvision.transforms
from PIL import Image
from torch import nn

root_dir = 'F:/01 Project/03 MachineLearning/02 MNIST/test_img'  # directory of the images to be tested
test_name = 'test_7.png'  # name of the image to be tested

img_path = os.path.join(root_dir, test_name)  # path of the image to be tested
print(img_path)  # printing the path of the image to be tested

img = Image.open(img_path)  # opening the image
img_1 = img.convert('1')  # converting the image to black and white，即灰度图
print(img)  # printing the image

# Testing the image
tran_pose = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32, 32)),
    torchvision.transforms.ToTensor(),
])


# ===============================定义网络=================================
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            # 输入为1通道，输出为6通道，卷积核大小为5*5，步长为1，补0
            nn.MaxPool2d(kernel_size=(2, 2)),  # 池化核大小为2*2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),  # 输入为6通道，输出为16通道，卷积核大小为5*5
            nn.MaxPool2d(kernel_size=(2, 2)),  # 池化核大小为2*2
            nn.Flatten(),  # 将输入数据展平，即将二维数组转化为一维数组
            nn.Linear(in_features=16 * 5 * 5, out_features=120),  # 输入为16*5*5，输出为120
            nn.Linear(in_features=120, out_features=84),  # 输入为120，输出为84
            nn.Linear(in_features=84, out_features=10)  # 输入为84，输出为10
        )

    def forward(self, x):
        x = self.model(x)
        return x
    # 定义前向传播过程


# ===============================测试网络===============================
mynet = torch.load('F:/01 Project/03 MachineLearning/02 MNIST/path/MNIST_19_Acc_0.981499969959259.pth',
                   map_location=torch.device('cpu'))
# loading the trained model and setting it to CPU
print(mynet)  # printing the model

img_1 = tran_pose(img_1)  # transforming the image
print(img_1.shape)  # 输出结果为torch.Size([4, 32, 32])，即4通道，32*32像素
# 但是模型输入为1通道，所以需要变换通道数
img_1 = torch.reshape(img_1, (1, 1, 32, 32))
print(img_1.shape)  # 输出结果为torch.Size([1, 32, 32])，即1通道，32*32像素

mynet.eval()  # setting the model to evaluation mode
with torch.no_grad():  # disabling the gradient calculation
    output = mynet(img_1)  # forwarding the image to the model
    number = output.argmax(axis=1).item()  # argmax(axis=1)返回每行最大值的索引，即预测的数字。item()返回索引对应的数值
    print(f'The predicted number is:{number}')
