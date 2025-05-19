# 基准查找模型
import chromadb
import os
import numpy as np

# 设置numpy输出格式，确保不省略任何部分
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

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
        n_results=1, # 返回最相似的一个文档
        include=["documents", "embeddings"]
    )
    if results and results['ids']:
        document_id = results['ids'][0][0]
        document_content = results['documents'][0][0]
        document_embedding = results['embeddings'][0][0] # 直接用query获取embedding
        # 根据document_id用get获取对应的嵌入向量
        detailed_results = collection.get(ids=[document_id], include=['embeddings'])
        detailed_embedding = detailed_results['embeddings'][0]  # 用query获取id，用get得到id对应的embedding
        # 两种方法得到的embedding并不相同
        # return document_content, detailed_embedding
        return document_content, document_embedding
    else:
        print("\n------------------------------\n"
              "未找到相似内容。")
        return None, None

# def test_searchByEmbedding():
#     # 示例嵌入向量，长度为1024
#     example_embedding = np.random.rand(1024).tolist()
#     # 调用搜索函数
#     document_content, detailed_embedding = searchByEmbedding(example_embedding)
#     # 打印结果
#     if document_content is not None:
#         print(f"找到的文档内容: {document_content}")
#         if detailed_embedding is not None:
#             print(f"找到的嵌入向量 (详细结果): {detailed_embedding}")
#         else:
#             print("找到的嵌入向量 (详细结果): 未找到嵌入向量")
#     else:
#         print("未找到相似内容。")

# if __name__ == "__main__":
#     test_searchByEmbedding()
