import os
import csv
from create_db import read_csv_and_create
from search_method4 import search_by_embedding
# from li_search import search_by_embedding
import jieba

# 停用词列表
STOPWORDS = set(["的", "了", "是", "和", "在", "有", "我", "也", "就", "不", "人", "都", "一个", "这", "那", "它", "你"
                 ,"根据","什么","哪些","指出","多少","哪个","知道","关于","中国联通"])

def extract_keywords_from_text(text):
    """
    使用jieba进行中文分词，并去除停用词，返回有效关键词列表
    """
    # 使用jieba分词
    words = jieba.cut(text)

    # 过滤停用词并返回关键词列表
    keywords = [word for word in words if word not in STOPWORDS and len(word.strip()) > 1]
    return keywords

def extract_keywords_from_question(question):
    """
    从用户问题中提取关键词
    """
    keywords = extract_keywords_from_text(question)
    return keywords

# 读取问题向量
def read_vector_question_csv(file_path):
    vectors = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for row in reader:
            index = row[0]
            original_question = row[1]
            embedding = [float(x) for x in row[2].strip('[]').split(',')]
            vectors.append((index, original_question, embedding))
    return vectors

# 根据问题向量检索答案并导出
def retrieve_and_export_answers():
    # 创建数据库
    read_csv_and_create()
    # 读取问题向量
    vector_question_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vector_question.csv")
    vectors = read_vector_question_csv(vector_question_file)
    # 检索并导出结果
    results = []
    for index, original_question, embedding in vectors:
        # 动态提取关键词
        keywords = extract_keywords_from_question(original_question)

        document_content, document_embedding = search_by_embedding(original_question,embedding,keywords=keywords)
        if document_content:
            results.append((index, original_question, embedding, document_content, document_embedding))

    # 将结果写入CSV文件
    csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrieved_answers.csv")
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["ques_id", "question", "answer", "embedding"])
        for index, original_question, embedding, retrieved_document, retrieved_embedding in results:
            # 将嵌入向量转换为字符串形式
            retrieved_embedding_str = ','.join(map(str, retrieved_embedding))
            writer.writerow([index, original_question, retrieved_document, retrieved_embedding_str])

    print(f"结果已保存到 {csv_file_path}")

if __name__ == "__main__":
    retrieve_and_export_answers()