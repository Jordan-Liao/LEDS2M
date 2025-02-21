import os
import shutil


def rename_and_organize_images(source_folder, target_folder):
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有子文件夹
    for subdir in os.listdir(source_folder):
        subdir_path = os.path.join(source_folder, subdir)

        # 确保是文件夹
        if os.path.isdir(subdir_path):
            # 遍历文件夹中的所有文件
            for file in os.listdir(subdir_path):
                # 构建完整的文件路径
                file_path = os.path.join(subdir_path, file)

                # 检查是否为图片，这里简单通过扩展名判断
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # 生成新的文件名和路径
                    new_file_name = subdir + '_' + file
                    new_file_path = os.path.join(target_folder, new_file_name)

                    # 复制并重命名图片到目标文件夹
                    shutil.copy(file_path, new_file_path)

                    # 获取新文件名的前5个字作为子文件夹名
                    subfolder_name = new_file_name[:5]
                    subfolder_path = os.path.join(target_folder, subfolder_name)

                    # 确保子文件夹存在
                    if not os.path.exists(subfolder_path):
                        os.makedirs(subfolder_path)

                    # 移动重命名的文件到相应的子文件夹
                    shutil.move(new_file_path, os.path.join(subfolder_path, new_file_name))


# 设置源文件夹和目标文件夹路径
source_folder = '/home/liaojr/BBDM-main/resultsA2B/SAMPLE/LBBDM-f4/sample_to_eval/condition'
target_folder = '/home/liaojr/BBDM-main/resultsA2B/SAMPLE/LBBDM-f4/sample_to_eval/condition_class'

# 调用函数
rename_and_organize_images(source_folder, target_folder)
