import chromadb
import pandas as pd
import os
import re

# 读取CSV文件并写入数据库
def read_csv_and_create():
    # 指定集合名称
    collection_name = "AnswerDataBase"
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 初始化Chroma数据库
    client = chromadb.PersistentClient(path=current_dir)
    # 尝试创建集合（collection）
    try:
        collection = client.create_collection(name=collection_name)
    except:
        return
        answer = input("检测到已存在同名数据库，是否需要删除并重新建立？（yes/no）\n")
        if answer == "yes":
            client.get_collection(name=collection_name)
            client.delete_collection(name=collection_name)
            collection = client.create_collection(name=collection_name)
        else:
            print("已放弃重新建立数据库")
            return

    # 读取CSV文件
    vector_passages_file = os.path.join(current_dir, "vector_passages.csv")
    data = pd.read_csv(vector_passages_file)
    # 写入至数据库
    for Index, row in data.iterrows():
        doc_id = row["Index"]  # id列名
        text = row["Original Passage"]  # 文本列名
        embedding = row["Embedding"]  # 向量
        # 将嵌入从字符串转换为实际的数值列表（假设用逗号分隔）
        embedding_list = list(map(float, embedding.strip().split(',')))  # 确保是浮点数列
        # 提取年份
        year = extract_year(text)
        # 将记录插入到Chroma数据库
        if year:
            metadata = {"year": year}
            collection.add(documents=[text], embeddings=[embedding_list], ids=[str(doc_id)], metadatas=[metadata])
        else:
            collection.add(documents=[text], embeddings=[embedding_list], ids=[str(doc_id)])

    print("数据导入完成！已存储至：", current_dir)
    # 验证数据是否成功存入
    # validate_data_insertion(collection, data)

# 提取年份的函数
def extract_year(text):
    # 尝试匹配多种格式的年份
    patterns = [
        r'\d{4}',  # 匹配四位数字的年份
        r'\d{4}年',  # 匹配四位数字后面带“年”的年份
        r'\d{4}-\d{2}-\d{2}',  # 匹配日期格式中的年份
        r'\d{4}/\d{2}/\d{2}',  # 匹配日期格式中的年份
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            year = match.group(0)
            # 去除年份后面的“年”或其他字符
            year = re.sub(r'年$', '', year)
            year = re.sub(r'-\d{2}-\d{2}$', '', year)
            year = re.sub(r'/\d{2}/\d{2}$', '', year)
            return int(year)
    
    return None

def validate_data_insertion(collection, data):
    # 选择一些记录进行验证
    sample_ids = data["Index"].sample(5).tolist()  # 随机选择5个记录进行验证

    # 查询这些记录
    results = collection.get(ids=[str(id) for id in sample_ids], include=['embeddings', 'documents', 'metadatas'])

    # 打印查询结果
    for i, doc_id in enumerate(sample_ids):
        if str(doc_id) in results['ids']:
            index = results['ids'].index(str(doc_id))
            document_content = results['documents'][index]
            document_embedding = results['embeddings'][index] if len(results['embeddings']) > index else None
            document_metadata = results['metadatas'][index] if len(results['metadatas']) > index else None

            print(f"验证记录 {doc_id}:")
            print(f"  文本: {document_content}")
            if document_embedding is not None:
                print(f"  嵌入: {document_embedding}")
            else:
                print("  嵌入: 未找到嵌入向量")
            if document_metadata is not None:
                print(f"  元数据: {document_metadata}")
            else:
                print("  元数据: 未找到元数据")
        else:
            print(f"未找到记录 {doc_id}")

if __name__ == "__main__":
    read_csv_and_create()