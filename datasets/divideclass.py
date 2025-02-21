import os
import shutil

# 定义源文件夹A的路径
source_folder = '/home/liaojr/BBDM-main/resultsA2B/SAMPLE/LBBDM-f4/sample_to_eval/ground_truth'  # 将此路径替换为实际的文件夹A路径
# 定义目标文件夹B的路径
target_root_folder = '/home/liaojr/BBDM-main/resultsA2B/SAMPLE/LBBDM-f4/sample_to_eval/ground_truth_class'  # 将此路径替换为实际的目标文件夹B路径

# 遍历源文件夹中的所有文件
for file_name in os.listdir(source_folder):
    # 检查文件是否是图片文件
    if file_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
        # 获取文件名前3个字符作为子文件夹名
        folder_name = file_name[:3]

        # 创建目标文件夹中的子文件夹路径
        folder_path = os.path.join(target_root_folder, folder_name)

        # 如果目标子文件夹不存在，创建它
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 构造源文件路径和目标文件路径
        source_file_path = os.path.join(source_folder, file_name)
        target_file_path = os.path.join(folder_path, file_name)

        # 复制文件到目标子文件夹
        shutil.copy(source_file_path, target_file_path)
        print(f"复制文件: {file_name} 到文件夹: {folder_path}")

print("图片分类完成！")
