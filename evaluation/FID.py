import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score

# 加载预训练的Inception-v3模型
inception_model = torchvision.models.inception_v3(pretrained=True).to(torch.device('cuda:0'))


def calc_FID(input_path1, input_path2):
    fid_value = fid_score.calculate_fid_given_paths([input_path1, input_path2],
                                                    batch_size=1,
                                                    device=torch.device('cuda:0'),
                                                    dims=2048)  # 2048,768,192,64
    print('FID value:', fid_value)
    return fid_value

calc_FID(input_path1='/home/liaojr/BBDM-main/results/SAMPLE/LBBDM-f4/sample_to_eval/condition',
         input_path2='/home/liaojr/BBDM-main/results/SAMPLE/LBBDM-f4/sample_to_eval/ground_truth')

# import os
# import shutil
# import torch
# import torchvision
# from pytorch_fid import fid_score

# # 加载预训练的Inception-v3模型
# # inception_model = torchvision.models.inception_v3(pretrained=True).to(torch.device('cuda:0'))

# def copy_images_to_folder(src_folder, dst_folder):
#     """复制src_folder及其子文件夹中的所有图像到dst_folder."""
#     if not os.path.exists(dst_folder):
#         os.makedirs(dst_folder)
#     for dirpath, _, filenames in os.walk(src_folder):
#         for filename in filenames:
#             if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
#                 shutil.copy(os.path.join(dirpath, filename), dst_folder)

# def calc_FID(temp_folder, direct_folder):
#     # 计算FID值
#     fid_value = fid_score.calculate_fid_given_paths([temp_folder, direct_folder],
#                                                     batch_size=1,
#                                                     device=torch.device('cuda:0'),
#                                                     dims=2048)
#     print('FID value:', fid_value)
#     return fid_value

# # 创建临时目录
# temp_folder1 = '/home/liaojr/pytorch-CycleGAN-and-pix2pix-master/results/SAMPLE_pix2pix/test_latest/images'

# #
# # # 复制图像
# # copy_images_to_folder('/home/liaojr/BBDM-main/results/SAMPLE/LBBDM-f4/sample_to_eval/sample_folder', temp_folder1)
# #
# # 调用函数，计算FID
# calc_FID(temp_folder1, '/home/liaojr/BBDM-main/results/SAMPLE/LBBDM-f4/sample_to_eval/ground_truth')

