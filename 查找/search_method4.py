# 年份筛选，拼接后的文本块重新转向量
import chromadb
import os
import numpy as np
import re
model_dir = r"D:\丁楚桐\bge"
from FlagEmbedding import FlagModel

model = FlagModel(model_dir,
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True)

# 设置numpy输出格式，确保不省略任何部分
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# 提取年份的函数
def extract_year(question):
    # 尝试匹配多种格式的年份
    patterns = [
        r'\d{4}',  # 匹配四位数字的年份
        r'\d{4}年',  # 匹配四位数字后面带“年”的年份
        r'\d{4}-\d{2}-\d{2}',  # 匹配日期格式中的年份
        r'\d{4}/\d{2}/\d{2}',  # 匹配日期格式中的年份
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            year = match.group(0)
            # 去除年份后面的“年”或其他字符
            year = re.sub(r'年$', '', year)
            year = re.sub(r'-\d{2}-\d{2}$', '', year)
            year = re.sub(r'/\d{2}/\d{2}$', '', year)
            return int(year)
    
    return None

def clean_content(content):
    # 定义需要替换或删除的目标字符列表
    replace_chars = {
        "None": "",  # 将"None"替换为空字符串
        "｜": "",  # 将全角竖线转换为半角竖线
        "|": "",  # 将竖线转换为空字符串"
        " ": ""  # 可选：去除空格，根据需要保留或注释掉这一行
    }
    
    # 使用replace方法替换特定字符串
    for old, new in replace_chars.items():
        content = content.replace(old, new)
    
    # 删除多余的竖线，假设连续两个竖线是不需要的
    while "||" in content:
        content = content.replace("||", "|")
    
    # 去除首尾可能存在的多余竖线
    content = content.strip("|")
    
    return content

def filter_by_keywords(ids,documents,embeddings, keywords):
    """
    根据给定的关键词过滤文档，只返回包含至少一半关键词的文档。
    """
    filtered_docs = []
    filtered_ids = []
    filtered_embeddings = []
    for i in range(len(ids)):
        doc = documents[i]
        id = ids[i]
        embedding = embeddings[i]
        # 计算匹配到的关键词数量
        matched_keywords = [keyword for keyword in keywords if keyword in doc]
        # print(len(doc))
        if len(matched_keywords) >= len(keywords) * 0.4 and len(doc) < 256:  # 如果匹配到的关键词数量不小于总关键词数量的一半
            filtered_ids.append(id)
            filtered_docs.append(doc)
            filtered_embeddings.append(embedding)
    return filtered_ids,filtered_docs,filtered_embeddings

# 根据嵌入向量搜索最相似文档
def search_by_embedding(original_question, embedding, keywords=None, k=5, metadata_filter=None, keyword=None, similarity_threshold=0.2):
    # 提取年份
    year = extract_year(original_question)
    
    # 更新元数据过滤条件
    if year and metadata_filter is None:
        metadata_filter = {"year": year}
    elif year:
        metadata_filter["year"] = year

    # 指定集合名称
    collection_name = "AnswerDataBase"
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 初始化Chroma数据库
    client = chromadb.PersistentClient(path=current_dir)
    try:
        collection = client.get_collection(name=collection_name)  # 确保使用与保存时相同的名称
    except Exception as e:
        print(f"无指定数据库，需先建立: {e}")
        return None, None
    
    # 查询相似的文档
    results = collection.query(
        query_embeddings=[embedding],
        n_results=10,  # 返回更多文档以便后续筛选
        include=["documents", "embeddings", "metadatas"]
    )
    
    if results and results['ids']:
        # 过滤和排序结果
        filtered_results = []
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            doc_content = results['documents'][0][i]
            doc_embedding = results['embeddings'][0][i]
            doc_metadata = results['metadatas'][0][i]
            
            # 元数据过滤
            if metadata_filter and doc_metadata:
                if not all(doc_metadata.get(key) == value for key, value in metadata_filter.items()):
                    continue
            
            # 计算相似度
            similarity = np.dot(embedding, doc_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(doc_embedding))
            if similarity < similarity_threshold:
                continue
            
            filtered_results.append((doc_id, doc_content, doc_embedding, similarity))
        
        # 按相似度排序
        filtered_results.sort(key=lambda x: x[3], reverse=True)
        
        # 取前k个结果
        top_ids = [result[0] for result in filtered_results[:k]]
        top_contents = [result[1] for result in filtered_results[:k]]
        top_embeddings = [result[2] for result in filtered_results[:k]]

        if keywords:
            temp_ids,temp_contents,temp_embeddings = filter_by_keywords(top_ids, top_contents,top_embeddings, keywords)
            if len(temp_ids) == 0:
                top_ids = top_ids[:1]
                top_contents = top_contents[:1]
                top_embeddings = top_embeddings[:1]
            else :
                top_ids = temp_ids
                top_contents = temp_contents
                top_embeddings = temp_embeddings
        
        combined_content = " ".join(top_contents)
        # combined_embedding = np.sum(top_embeddings, axis=0).tolist()
        clean_combined_content = clean_content(combined_content)
        combined_embedding = model.encode(clean_combined_content)
        return clean_combined_content, combined_embedding
    else:
        print("\n------------------------------\n"
              "未找到相似内容。")
        return None, None