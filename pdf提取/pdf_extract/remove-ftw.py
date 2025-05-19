
import os

def remove_string_from_files(directory, string_to_remove):
    # 遍历指定目录中的所有 .txt 文件
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # 检查文件中是否包含指定字符串
            if string_to_remove in content:
                # 替换指定字符串为空
                new_content = content.replace(string_to_remove, '')
                # print(new_content)
                # 将新内容写回文件
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(new_content)
                print(f"Processed {filename}: Removed '{string_to_remove}'")
            else:
                print(f"Skipped {filename}: '{string_to_remove}' not found")

def remove_publish_time_from_directory(directory_path):
    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):  # 只处理 .txt 文件
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # 过滤掉包含“发布时间”的行
            filtered_lines = [line for line in lines if not line.startswith("发布时间：")]
            # print(filtered_lines)

            # 将结果写回到文件
            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(filtered_lines)


directory_path = './txt-ftw/'  
# 1.去除"本文档……"
string_to_remove = '本文档为2024CCFBDCI比赛用语料的一部分。部分文档使用大语言模型改写生成，内容可能与现实情况\n不符，可能不具备现实意义，仅允许在本次比赛中使用。'  
remove_string_from_files(directory_path, string_to_remove)

# 2.去除发布时间
remove_publish_time_from_directory(directory_path)






