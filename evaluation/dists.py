# import os
# from PIL import Image
# import torch
# from piq import DISTS
# from torchvision import transforms
# from tqdm import tqdm

# # 初始化 DISTS 计算器
# dists_metric = DISTS()

# # 图像转换
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),  # 调整图像大小
#     transforms.ToTensor(),          # 转换为张量
# ])

# folder1 = '/path/to/folder1'
# folder2 = '/path/to/folder2'

# # 获取文件列表并排序
# images1 = sorted(os.listdir(folder1))
# images2 = sorted(os.listdir(folder2))

# # 确保两个文件夹中的图像数量相同
# assert len(images1) == len(images2), "两个文件夹中的图像数量不一致"

# total_score = 0.0
# num_images = len(images1)

# for img_name1, img_name2 in tqdm(zip(images1, images2), total=num_images):
#     # 构建完整路径
#     img_path1 = os.path.join(folder1, img_name1)
#     img_path2 = os.path.join(folder2, img_name2)

#     # 打开并预处理图像
#     img1 = Image.open(img_path1).convert('RGB')
#     img2 = Image.open(img_path2).convert('RGB')

#     img1 = transform(img1).unsqueeze(0)  # [1, C, H, W]
#     img2 = transform(img2).unsqueeze(0)

#     # 计算 DISTS 分数
#     with torch.no_grad():
#         score = dists_metric(img1, img2).item()

#     total_score += score

# # 计算平均 DISTS 分数
# average_score = total_score / num_images

# print(f'平均 DISTS 分数：{average_score}')



import torch
from PIL import Image
from piq import DISTS
from torchvision import transforms

# 初始化 DISTS 计算器
dists_metric = DISTS()

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 可根据需要调整大小
    transforms.ToTensor(),
])

# 加载并预处理第一张图片
img1 = Image.open('/home/liaojr/BBDM-main/results/SAMPLE/LBBDM-f4/sample_to_eval/condition/2s1_real_A_elevDeg_015_azCenter_010_22_serial_b01.png').convert('RGB')
img1 = transform(img1).unsqueeze(0)
0


# 加载并预处理第二张图片
img2 = Image.open('/home/liaojr/BBDM-main/results/SAMPLE/LBBDM-f4/sample_to_eval/ground_truth/2s1_synth_A_elevDeg_015_azCenter_010_22_serial_b01.png').convert('RGB')
img2 = transform(img2).unsqueeze(0)

# 计算 DISTS 分数
with torch.no_grad():
    score = dists_metric(img1, img2).item()

print(f'两张图片的 DISTS 分数：{score}')
