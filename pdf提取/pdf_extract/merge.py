import os

def merge_txt_files(directories, output_file):
    """
    将多个指定目录下的所有 .txt 文件合并到一个输出文件中。
    每个文件的内容前会添加文件名作为标识。

    :param directories: 包含 .txt 文件的目录路径列表
    :param output_file: 合并后的输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for directory in directories:
            for filename in os.listdir(directory):
                if filename.endswith('.txt'):
                    file_path = os.path.join(directory, filename)
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(f"{filename} (from {directory}):\n{infile.read()}\n{'-'*40}\n")

# 示例用法
directories = [
    './txt-ftw/',
    './txt-y/',
    './txt-z/'
]
output_path = 'merged_output.txt'
merge_txt_files(directories, output_path)