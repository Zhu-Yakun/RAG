import os
import re
import warnings
import logging

import pdfplumber
import PyPDF2
from pdfminer.layout import LTTextContainer
from pdfminer.high_level import extract_pages

# ---------------- 配置 ----------------
HEADER_CUTOFF = 0.85
NAV_KEYWORDS = [
    "基本概念", "发展历程", "应用领域", "发展趋势",
    "课程安排", "相关拓展", "目录"," 基本问题","分词规范","方法概述","机械分词","分词演示","分词歧义的类型","歧义字段的类型","歧义字段的发现方法","歧义字段的处理方法","基于规则的歧义消解","基于统计的歧义消解","语料库",
    "机器学习分词","机器学习分词","分词总结","分词作业","自动标注","标注分类","词性标注歧义","标注准确性评价","词典","概述","非上下文 预训练模型","上下文 预训练模型","预训练模型概述","非上下文预训练模型","上下文预训练模型"
]
# 如果页面包含此短语，则整页跳过
DROP_PHRASE = ["同济大学计算机科学与技术系","谢谢观看"]

# ---------------- 日志 & 警告 ----------------
logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def clean_extracted_text(text: str) -> str:
    text = re.sub(r'\b[nN]\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(?<=[。！])', '\n', text)

    lines = []
    for line in text.splitlines():
        l = line.strip()
        if not l:
            continue
        if re.fullmatch(r'\d+', l):   # 纯数字行（页码）
            continue
        if any(k in l for k in NAV_KEYWORDS):
            continue
        if '|' in l:
            continue
        lines.append(l)
    return "\n".join(lines)


def extract_tables_from_page(pdf_path: str, page_num: int):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        ph = page.bbox[3]
        settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "intersection_tolerance": 3,
        }
        tables = page.find_tables(table_settings=settings)

    valid = []
    for t in tables:
        x0, y0, x1, y1 = t.bbox
        if y1 > ph * HEADER_CUTOFF:
            continue
        data = t.extract()
        if not any((cell or "").strip() for cell in data[0]):
            continue
        blob = " ".join(cell or "" for row in data for cell in row)
        if any(k in blob for k in NAV_KEYWORDS):
            continue
        valid.append(t)

    valid.sort(key=lambda t: (-t.bbox[3], t.bbox[0]))
    return valid


def table_to_markdown(table):
    clean = [
        [(cell or "").strip().replace("\n", " ") for cell in row]
        for row in table
    ]
    col_w = [max(len(r[i]) for r in clean) for i in range(len(clean[0]))]
    sep = "|" + "|".join("-" * w for w in col_w) + "|"

    lines = []
    for i, row in enumerate(clean):
        padded = [row[j].ljust(col_w[j]) for j in range(len(row))]
        lines.append("|" + "|".join(padded) + "|")
        if i == 0:
            lines.append(sep)
    return "\n".join(lines)


def pdf_trans(pdf_path: str) -> str:
    reader = PyPDF2.PdfReader(open(pdf_path, "rb"))
    parts = []

    # 打开 pdfplumber 文档一次性读取页面文本
    with pdfplumber.open(pdf_path) as plumber_doc:
        for pnum, layout in enumerate(extract_pages(pdf_path)):
            # 1. 检查是否包含 DROP_PHRASE
            page_text = plumber_doc.pages[pnum].extract_text() or ""
            if DROP_PHRASE[0] in page_text:
                continue  # 整页跳过
            if DROP_PHRASE[1] in page_text:
                continue  # 整页跳过
            ph = layout.bbox[3]

            # 2️⃣ 提取表格
            for tbl in extract_tables_from_page(pdf_path, pnum):
                md = table_to_markdown(tbl.extract())
                parts.append(md)

            # 3️⃣ 提取正文（跳过顶部导航区域）
            raw = []
            for elem in layout:
                if isinstance(elem, LTTextContainer) and elem.y1 <= ph * HEADER_CUTOFF:
                    raw.append(elem.get_text())
            cleaned = clean_extracted_text("".join(raw))
            # 去掉本页末尾可能的页码及符号
            cleaned = re.sub(r'\d+\W*$', '', cleaned).strip()
            if cleaned and not cleaned.endswith("。"):
                cleaned += "。"
            parts.append(cleaned)

    # 合并并去重段落
    text = "\n\n".join(p.strip() for p in parts if p.strip())
    paras = [p for p in text.split("\n\n") if p]
    seen = set()
    final = []
    for p in paras:
        if p in seen:
            continue
        seen.add(p)
        final.append(p)
    return "\n\n".join(final)


def batch_convert_pdfs_to_txt(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for fn in os.listdir(input_dir):
        if not fn.lower().endswith(".pdf"):
            continue
        src = os.path.join(input_dir, fn)
        dst = os.path.join(output_dir, os.path.splitext(fn)[0] + ".txt")
        print(f"Processing {fn} → {os.path.basename(dst)}")
        txt = pdf_trans(src)
        with open(dst, "w", encoding="utf-8") as f:
            f.write(txt)
    print("Done.")


if __name__ == "__main__":
    batch_convert_pdfs_to_txt("../data", "./output_txt")
