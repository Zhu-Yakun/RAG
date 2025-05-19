# 每个question选择与其最相近的两个chunk，而非一个，再对这两个chunk文本拼接，向量相加
import chromadb
import os
import numpy as np

# 设置numpy输出格式，确保不省略任何部分
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def cosine_similarity(a, b):
    # 将输入转换为numpy数组
    a = np.array(a)
    b = np.array(b)
    # 计算点积和范数
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        # 防止除以0的情况
        return 0.0
    return dot_product / (norm_a * norm_b)

# 根据嵌入向量搜索最相似文档
def search_by_embedding(embedding):
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
        n_results=3, # 返回最相似的3个文档
        include=["documents", "embeddings"]
    )
    if results and results['ids']:
        # 获取前两个最相似的文档的ID和内容
        top_ids = results['ids'][0]
        top_contents = results['documents'][0]

        combined_content = []
        combined_embeddings = []
        simi = []

        for document_id, document_content in zip(top_ids, top_contents):
            # 根据document_id用get获取对应的嵌入向量
            detailed_results = collection.get(ids=[document_id], include=['embeddings'])
            detailed_embedding = detailed_results['embeddings'][0]  # 获取对应的嵌入向量

            # 拼接文档内容
            combined_content.append(document_content)
            combined_embeddings.append(detailed_embedding)
            simi.append(cosine_similarity(embedding, detailed_embedding))

            # 拼接内容
        res_content = []
        res_embeddings = []

        res_content.append(combined_content[0])
        res_content.append(combined_content[1])
        if (simi[1] < (simi[0] + simi[2]) / 2):
            res_content.append(combined_content[2])

        res_embeddings.append(combined_embeddings[0])
        res_embeddings.append(combined_embeddings[1])
        if (simi[1] < (simi[0] + simi[2]) / 2):
            res_embeddings.append(combined_embeddings[2])

        combined_content_str = " ".join(res_content)
        # 向量相加，确保转为numpy数组相加
        combined_embedding = np.sum(res_embeddings, axis=0).tolist()  # 转为列表

        return combined_content_str, combined_embedding
    else:
        print("\n------------------------------\n"
              "未找到相似内容。")
        return None, None