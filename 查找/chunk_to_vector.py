import os
import pandas as pd

# Function to read every 3rd line starting from the 2nd line
def read_every_3rd_line_from_second(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        i = 0
        for line in file:
            if i % 3 == 2:  # Check if it's the 2nd, 5th, 8th, etc. line
                lines.append(line.strip())
            i += 1
    return lines

def modelInit():
    # Step 3: Initialize and use FlagModel for encoding
    model_dir = r"E:\Personal\Academic\mainCourse\数据挖掘\toVector\bge"
    from FlagEmbedding import FlagModel

    model = FlagModel(model_dir,
                      query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                      use_fp16=True)

    return model
    # Now q_embeddings and p_embeddings contain the encoded representations of queries and passages respectively.

def read_blocks(file_path):
    # 打开文件并读取内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用标识符分块内容，假设分块通过 "--- 分块 X ---" 来标识
    blocks = content.split('\n---')

    # 将分块内容存储在列表中
    block_list = []
    for block in blocks:
        if block.strip():  # 忽略空块
            # 获取分块的内容，分块标题可以忽略
            lines = block.strip().split('\n', 1)
            if len(lines) > 1:
                block_content = lines[1].strip()  # 获取内容并去除前后空白字符
                # 去除块内的回车符和多余的空白字符
                block_content = block_content.replace('\n', ' ').replace('\r', '').strip()
                block_list.append(block_content)

    return block_list

def textConvertion(chunk_dir, model, save_dir):
    # Step 2: Read passages from every 3rd line starting from the 2nd line in txt files in chunk directory
    passages = []

    # Iterate through each txt file in the directory
    for filename in os.listdir(chunk_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(chunk_dir, filename)
            # selected_lines = read_every_3rd_line_from_second(file_path)
            selected_lines = read_blocks(file_path)
            passages.extend(selected_lines)
            # break
    # print(passages)

    p_embeddings = model.encode(passages)  # [382,1024]
    print(p_embeddings.shape)

    # Step 6: Create DataFrame for vector_passages.csv
    vector_passages = pd.DataFrame({
        'Index': range(1, len(passages) + 1),  # Index from 1 to number of passages
        'Original Passage': passages,  # Original data
        'Embedding': [",".join(map(str, p)) for p in p_embeddings]  # Convert each embedding to a comma-separated string
    })

    # Step 7: Save vector_passages.csv
    vector_passages.to_csv(save_dir, index=False)

    print("CSV files generated successfully!")

def questConvertion(question_file, model,questSaveDir):
    df = pd.read_csv(question_file)
    queries = df['question'].tolist()

    # Encode queries and passages
    q_embeddings = model.encode_queries(queries)  # [100,1024]
    print(q_embeddings.shape)

    # Step 4: Create DataFrame for vector_question.csv
    vector_question = pd.DataFrame({
        'Index': range(1, len(queries) + 1),  # Index from 1 to 100
        'Original Question': queries,  # Original data
        'Embedding': [",".join(map(str, q)) for q in q_embeddings]  # Convert each embedding to a comma-separated string
    })

    # Step 5: Save vector_question.csv
    vector_question.to_csv(questSaveDir, index=False)

if __name__ == '__main__':
    chunk_dir = r"E:\Personal\Academic\mainCourse\数据挖掘\toVector\lda_chunk_AB"
    save_dir = r"E:\Personal\Academic\mainCourse\数据挖掘\toVector\vector_passages_lda_AB.csv"

    # Step 1: Read queries from question.csv
    # question_file = r"E:\Personal\Academic\mainCourse\数据挖掘\toVector\B_question.csv"
    # questSaveDir = r"E:\Personal\Academic\mainCourse\数据挖掘\toVector\vector_questionB.csv"

    model = modelInit()
    textConvertion(chunk_dir, model, save_dir)
    # questConvertion(question_file, model,questSaveDir)

