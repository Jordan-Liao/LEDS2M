import os
import shutil
import re

def distribute_images(src_directory, train_directory, val_directory):
    # 定义正则表达式匹配015, 016, 017
    train_pattern = re.compile(r".*(014_azCenter|015_azCenter|016_azCenter).*\.png$")
    val_pattern = re.compile(r".*017_azCenter.*\.png$")

    # 遍历源目录中的所有子目录
    for subdir in os.listdir(src_directory):
        current_dir = os.path.join(src_directory, subdir)
        if os.path.isdir(current_dir):
            # 创建相应的train和val子目录
            train_subdir = os.path.join(train_directory, subdir)
            val_subdir = os.path.join(val_directory, subdir)
            if not os.path.exists(train_subdir):
                os.makedirs(train_subdir)
            if not os.path.exists(val_subdir):
                os.makedirs(val_subdir)

            # 遍历子目录中的所有文件
            for file in os.listdir(current_dir):
                file_path = os.path.join(current_dir, file)
                if train_pattern.match(file):
                    # 复制到train目录
                    shutil.copy(file_path, train_subdir)
                elif val_pattern.match(file):
                    # 复制到val目录
                    shutil.copy(file_path, val_subdir)

# 设置路径
src_directory = '/home/liaojr/BBDM-main/resultsA2B/SAMPLE/LBBDM-f4/sample_to_eval/sample_folder'
train_directory = '/home/storageSDA1/liaojr/sampleA2B_for_az/train'
val_directory = '/home/storageSDA1/liaojr/sampleA2B_for_az/val'

# 调用函数
distribute_images(src_directory, train_directory, val_directory)
