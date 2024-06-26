import os
import shutil
import random

# 按照比例分割图片文件
# source_folder: 源文件夹
# target_folder1: 目标文件夹1
# target_folder2: 目标文件夹2
# ratio: 分割比例

def split_images(source_folder, target_folder1, target_folder2, ratio=0.1):
    # 创建目标文件夹
    if not os.path.exists(target_folder1):
        os.makedirs(target_folder1)
    if not os.path.exists(target_folder2):
        os.makedirs(target_folder2)

    # 获取源文件夹内的所有图片文件
    images = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    # 打乱图片文件顺序
    random.shuffle(images)

    # 按照比例分割图片文件
    split_index = int(len(images) * ratio)
    images1 = images[:split_index]
    images2 = images[split_index:]

    # 移动图片文件到目标文件夹
    for image in images1:
        shutil.move(os.path.join(source_folder, image), os.path.join(target_folder1, image))

    for image in images2:
        shutil.move(os.path.join(source_folder, image), os.path.join(target_folder2, image))

    print(f'Moved {len(images1)} images to {target_folder1}')
    print(f'Moved {len(images2)} images to {target_folder2}')


# 示例用法
source_folder = 'E:/01_Download/02_EdgeDownload/flower_photos/tulips'
target_folder1 = 'E:/01_Download/02_EdgeDownload/flower_photos/val/tulips'
target_folder2 = 'E:/01_Download/02_EdgeDownload/flower_photos/train/tulips'

split_images(source_folder, target_folder1, target_folder2)
