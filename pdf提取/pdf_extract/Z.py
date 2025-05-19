import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from pdfminer.high_level import extract_text

def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    return images

def ocr_image(image):
    text = pytesseract.image_to_string(image, lang='chi_sim')  # 使用简体中文语言模型
    return text

def extract_text_from_pdf(pdf_path):
    try:
        # 尝试使用 pdfminer.six 提取文本
        text = extract_text(pdf_path)
        if not text.strip():
            # 如果 pdfminer.six 提取的文本为空，使用 OCR 技术
            images = pdf_to_images(pdf_path)
            text = ""
            for image in images:
                text += ocr_image(image)
        return text
    except Exception as e:
        print(f"提取文本时发生错误: {e}")
        return None

def batch_extract_text_from_pdfs(directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 指定需要处理的文件列表
    files_to_process = [f"AZ{i:02d}.pdf" for i in range(1, 11)]

    for filename in files_to_process:
        pdf_path = os.path.join(directory, filename)
        if os.path.exists(pdf_path):
            output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.txt")
            extracted_text = extract_text_from_pdf(pdf_path)
            if extracted_text:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)
                print(f"成功提取文本并保存到 {output_path}")
            else:
                print(f"无法提取 {pdf_path} 中的文本")
        else:
            print(f"文件 {pdf_path} 不存在")

# 示例调用
input_directory = '../data'
output_directory = './txt-z'
batch_extract_text_from_pdfs(input_directory, output_directory)