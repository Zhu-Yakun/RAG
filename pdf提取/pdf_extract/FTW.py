import os
import PyPDF2

# 定义要处理的目录
input_directory = "../data"
output_directory = "./txt-ftw/"

# 遍历目录中的所有文件
for filename in os.listdir(input_directory):
    # 检查文件是否以指定前缀开头且是PDF文件
    if filename.lower().endswith(".pdf") and any(filename.startswith(prefix) for prefix in ["AF", "AT", "AW", "BF", "BT", "BW"]):
        pdf_path = os.path.join(input_directory, filename)
        txt_filename = f"{os.path.splitext(filename)[0]}.txt"  # 创建对应的TXT文件名
        txt_path = os.path.join(output_directory, txt_filename)  # 输出路径

        # 打开PDF文件
        with open(pdf_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            # 打开对应的TXT文件
            with open(txt_path, "w", encoding="utf-8") as f:
                # 提取每一页的文本
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        f.write(text + "\n")

print("文本提取完成，所有PDF文件已处理。")
