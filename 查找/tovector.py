import pandas as pd
from FlagEmbedding import FlagModel

# 设置文件路径
input_csv = r"D:\丁楚桐\5 数据挖掘\CCF大数据\DataminingTeamRepository-main\RetrieveResults\test1.csv"
output_csv = r"D:\丁楚桐\5 数据挖掘\CCF大数据\DataminingTeamRepository-main\RetrieveResults\test1_true_emb.csv"
model_dir = r"D:\丁楚桐\bge"

# 读取CSV文件
df = pd.read_csv(input_csv)
print("read completed")

# 初始化模型
model = FlagModel(
    model_dir,
    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
    use_fp16=True
)

# 定义一个函数将答案编码为向量
def encode_answer(answer):
    embedding = model.encode(answer)
    # 将向量转换为字符串，以便存储在CSV中
    embedding_str = ','.join(map(str, embedding))
    return embedding_str

# 对每个答案进行编码并创建新的embedding列
df['embedding'] = df['answer'].apply(encode_answer)

# 选择需要的列
df_output = df[['ques_id', 'question', 'answer', 'embedding']]

# 保存到新的CSV文件
df_output.to_csv(output_csv, index=False, encoding='utf-8-sig')

print("向量编码完成，结果已保存到", output_csv)
