import chromadb
import os

def searchByIndex():
    # 指定集合名称
    collection_name = "AnswerDataBase"

    # 获取当前文件所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 初始化 Chroma 数据库
    client = chromadb.PersistentClient(path=current_dir)

    try:
        collection = client.get_collection(name=collection_name)  # 确保使用与保存时相同的名称
    except:
        print("无指定数据库，需先建立")
        return

    # 根据 Index 查询目标内容
    target_index = input("input target index, 0 to escape\n")  # 替换为你要查询的索引
    if target_index == '0':
        return 0

    # 查询文档
    document = collection.get(ids=[target_index], include=['embeddings', 'documents'])

    # print(document)
    if document['ids']:
        print("\n------------------------------\n"
              "ID：", document['ids'], "\n内容：", document['documents'],
              "\n向量：", document['embeddings'], "\n")
    else:
        print("\n------------------------------\n"
              "未找到目标索引内容。")

    return 1

flag=1
while(flag):
    flag=searchByIndex()