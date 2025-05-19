# 关键词过滤和特定字符筛选
import chromadb
import os
import numpy as np
import re


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

def filter_by_keywords(ids,documents, keywords):
    """
    根据给定的关键词过滤文档，只返回包含至少一半关键词的文档。
    """
    filtered_ids = []
    filtered_docs = []
    for i in range(len(ids)):
        doc = documents[i]
        id = ids[i]
        # 计算匹配到的关键词数量
        matched_keywords = [keyword for keyword in keywords if keyword in doc]
        print(len(doc))
        if len(matched_keywords) >= len(keywords) * 0.4 and len(doc) < 256:  # 如果匹配到的关键词数量不小于总关键词数量的一半
            filtered_ids.append(id)
            filtered_docs.append(doc)
    return filtered_ids,filtered_docs
# 设置numpy输出格式，确保不省略任何部分
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# 根据嵌入向量搜索最相似文档
def search_by_embedding(embedding, keywords=None):
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
        n_results=1, # 返回最相似的3个文档
        include=["documents", "embeddings"]
    )
    if results and results['ids']:
        # 获取前两个最相似的文档的ID和内容
        top_ids = results['ids'][0] # 获取前k个文档的ID
        top_contents = results['documents'][0] # 获取前k个文档的内容
        # print(top_contents)
        # 如果有关键词过滤，进行过滤
        if keywords:
            temp_ids,temp_contents = filter_by_keywords(top_ids, top_contents, keywords)
            if len(temp_ids) == 0:
                top_ids = top_ids[:1]
                top_contents = top_contents[:1]
            else :
                top_ids = temp_ids
                top_contents = temp_contents
        # print(top_contents)
        # exit()
        combined_content = []
        combined_embeddings = []

        for document_id, document_content in zip(top_ids, top_contents):
            # 根据document_id用get获取对应的嵌入向量
            detailed_results = collection.get(ids=[document_id], include=['embeddings'])
            detailed_embedding = detailed_results['embeddings'][0]  # 获取对应的嵌入向量

            # 拼接文档内容
            combined_content.append(document_content)
            combined_embeddings.append(detailed_embedding)

            # 拼接内容
        combined_content_str = " ".join(combined_content)
        # 向量相加，确保转为numpy数组相加
        combined_embedding = np.sum(combined_embeddings, axis=0).tolist()  # 转为列表

        return clean_content(combined_content_str), combined_embedding
    else:
        print("\n------------------------------\n"
              "未找到相似内容。")
        return None, None
