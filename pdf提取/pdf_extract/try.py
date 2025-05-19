# # from pdf2image import convert_from_path
# # import pytesseract
# # from PIL import Image

# # # 指定 Tesseract 的路径（如有必要）
# # # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # # 将 PDF 转换为图像
# # pdf_path = '../data/BY04.pdf'
# # images = convert_from_path(pdf_path)

# # # 提取每页的文本
# # for i, image in enumerate(images):
# #     # 使用 OCR 识别文本，指定简体中文
# #     text = pytesseract.image_to_string(image, lang='chi_sim')
    
# #     # 打印或保存识别的文本
# #     print(f'第 {i + 1} 页文本:')
# #     print(text)

# from pdf2image import convert_from_path
# import pytesseract
# from PIL import Image

# # 指定 Tesseract 的路径（如有必要）
# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # 将 PDF 转换为图像
# pdf_path = '../data/BY04.pdf'
# images = convert_from_path(pdf_path)

# # 提取每页的文本
# for i, image in enumerate(images):
#     # 使用 OCR 识别文本
#     text = pytesseract.image_to_string(image, lang='chi_sim')
    
#     # 打印整页文本
#     print(f'第 {i + 1} 页文本:')
#     print(text)

#     # 识别页眉和页脚
#     width, height = image.size
#     header = image.crop((0, 0, width, height * 0.1))  # 假设页眉在顶部10%
#     footer = image.crop((0, height * 0.9, width, height))  # 假设页脚在底部10%

#     # 识别页眉和页脚文本
#     header_text = pytesseract.image_to_string(header, lang='chi_sim')
#     footer_text = pytesseract.image_to_string(footer, lang='chi_sim')

#     print(f'第 {i + 1} 页页眉:')
#     print(header_text)
#     print(f'第 {i + 1} 页页脚:')
#     print(footer_text)

import os
import PyPDF2

# 定义要处理的目录
input_directory = "../data"
output_directory = "./txt-y/"

# 遍历目录中的所有文件
for filename in os.listdir(input_directory):
    # 检查文件是否以指定前缀开头且是PDF文件
    if filename.lower().endswith(".pdf") and any(filename.startswith(prefix) for prefix in ["BY"]):
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