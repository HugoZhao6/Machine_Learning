import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader

# ===============================设置设备==================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 判断是否有GPU
print(device)  # 输出设备

# =============================加载数据集==============================
train_dataset = torchvision.datasets.MNIST(root='dataset', train=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32, 32)),
    torchvision.transforms.ToTensor()
]), download=False)
# 下载数据集，第一次运行时需要，之后不需要。train=True表示加载训练集，False表示加载测试集。
# torchvision.transforms.Compose()表示对数据集进行预处理，这里将图片大小调整为32*32。
# torchvision.transforms.ToTensor()表示将图片转化为张量。

test_dataset = torchvision.datasets.MNIST(root='dataset', train=False, transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32, 32)),
    torchvision.transforms.ToTensor()
]), download=False)

# ==============================获取数据集大小============================
train_data_size = len(train_dataset)
test_data_size = len(test_dataset)

print(f'训练集长度为{train_data_size}')
print(f'测试集长度为{test_data_size}')

# ===============================加载数据=================================
train_data_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True, drop_last=True)
# 加载数据集，batch_size表示每次加载多少张图片，shuffle表示是否打乱顺序，drop_last表示是否丢弃最后一批
# 丢弃最后一批是为了保证数据集的长度是batch_size的整数倍。即当剩余数据量小于batch_size时，丢弃。

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


mynet = LeNet()  # 实例化网络
print(mynet)  # 打印网络结构
mynet = mynet.to(device)  # 将网络移至GPU设备，加速运算

loss_fn = nn.CrossEntropyLoss()  # 定义损失函数
loss_fn = loss_fn.to(device)  # 将损失函数移至GPU设备，加速运算

learning_rate = 1e-2  # 学习率
optim = torch.optim.SGD(mynet.parameters(), learning_rate)  # 定义优化器

# =============================训练部分核心代码============================
train_step = 0  # 训练次数计数器
epoch = 20  # 训练的轮数

if __name__ == '__main__':  # 训练部分
    for i in range(epoch):
        print(f'----------第{i + 1}轮训练----------')
        mynet.train()  # 防止过拟合（可去掉）
        for data in train_data_loader:  # 加载训练集
            imgs, targets = data  # 取出图片和标签
            # print(imgs.shape)
            imgs = imgs.to(device)  # 将图片移至GPU设备
            targets = targets.to(device)  # 将标签移至GPU设备

            outputs = mynet(imgs)  # 前向传播
            # print(outputs.shape)

            loss = loss_fn(outputs, targets)  # 损失函数计算
            optim.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optim.step()  # 更新优化器梯度参数

            train_step += 1  # 训练次数计数器加1
            if train_step % 100 == 0:  # 每100次训练打印一次loss
                print(f'第{train_step}次训练，loss={loss.item()}')

        # ===============================测试部分核心代码===========================
        mynet.eval()  # 测试部分开始标志防止过拟合（可去掉）
        accuracy = 0  # 准确率计数器
        total_accuracy = 0  # 总准确率计数器

        with torch.no_grad():  # 禁止梯度计算
            for data in test_data_loader:  # 加载测试集
                imgs, targets = data  # 取出图片和标签
                imgs = imgs.to(device)  # 将图片移至GPU设备
                targets = targets.to(device)  # 将标签移至GPU设备

                outputs = mynet(imgs)  # 前向传播

                accuracy = (outputs.argmax(axis=1) == targets).sum()
                # 计算准确率，argmax取最大值索引，axis=1表示按行计算，axis=0表示按列计算，==判断是否相等
                total_accuracy += accuracy  # 总准确率计数器加上当前批次的准确率

            rate = total_accuracy / test_data_size  # 计算准确率
            print(f'{i + 1}轮训练结束,准确率{rate}')

            file_name = f'path/MNIST_{i}_Acc_{rate}.pth'  # 设置保存模型文件名
            torch.save(mynet, file_name)  # 保存每一轮训练的模型
    torch.save(mynet, 'path/MNIST_Acc.pth')  # 保存最终的模型
