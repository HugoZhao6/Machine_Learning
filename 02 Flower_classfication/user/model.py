# 花分类问题，基于AlexNet

import torch.nn as nn


# 该项目第一次使用激活函数
# 激活函数是为了增加模型的非线性，使得模型能够拟合复杂的函数关系
# 常见的激活函数有：ReLU、Sigmoid、Tanh、LeakyReLU、ELU、Softmax、Softplus、Swish
# 这里使用了ReLU激活函数，它是最常用的激活函数之一
# 本次网络构建均只使用了原论文的一半的节点个数，目的是为了节省计算资源，提高训练速度
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        # 卷积层，提取图像特征
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            # 原论文中输出通道数为64，这里改为48，目的是加快运算速度，对结果影响不大
            nn.ReLU(inplace=True),  # inplace=True 表示可以改变输入的数据，节省内存，可以载入更大的模型
            nn.MaxPool2d(kernel_size=3, stride=2),  # 最大池化层，降低维度
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 全连接层，将特征图转换为类别输出
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # 随机丢弃一些神经元，即使一部分神经元不工作，防止过拟合。p表示丢弃的比例，默认值为0.5
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

        #    初始化权重
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # 展平特征图，strat_dim=1表示从第二维开始展平，即从通道维度开始展平
        x = self.classifier(x)  # 分类器
        return x

    def _initialize_weights(self):
        for m in self.modules():  # 遍历模型中的所有层，然后进行判断
            if isinstance(m, nn.Conv2d):  # 如果是卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 使用kaiming初始化权重
                if m.bias is not None:  # 如果有偏置项
                    nn.init.constant_(m.bias, 0)  # 将偏置项初始化为0
            elif isinstance(m, nn.BatchNorm2d):  # 如果是批归一化层
                nn.init.constant_(m.weight, 1)  # 将权重初始化为1
                nn.init.constant_(m.bias, 0)  # 将偏置项初始化为0
            elif isinstance(m, nn.Linear):  # 如果是全连接层
                nn.init.normal_(m.weight, 0, 0.01)  # 使用正态分布初始化权重，均值为0，方差为0.01
                nn.init.constant_(m.bias, 0)  # 将偏置项初始化为0


# 实例化模型并输出模型结构
# my_model = AlexNet()
# print(my_model)
