
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 加载数据集