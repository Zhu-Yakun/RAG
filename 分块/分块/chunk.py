import os
import re
import jieba
import gensim
from gensim import corpora
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
import torch
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
import dashscope

chunk_num=0
            
class MyCorpus():
    def __init__(self, input_folder,output_folder,method):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.method=method
        
    def process_files(self):
        if self.method=="charactor":
            # 按固定字符大小分块
            self.process_files_in_folder_1(self.input_folder, self.output_folder)
        elif self.method=="bert":
            # 基于BERT的分块
            self.process_folder_bert(self.input_folder, self.output_folder,similarity_threshold=0.85, max_chunk_length=3)
        elif self.method=="lda":
            # 基于LDA的分块
            self.process_folder_lda(self.input_folder, self.output_folder)
        elif self.method=="llm":
            # 基于语言模型的分块
            self.process_directory(self.input_folder, self.output_folder)
        else:
            print("请输入正确的分块方法")
    
    def chunk_character(self,text, chunk_size=200, chunk_overlap=50):
        text_splitter = CharacterTextSplitter(
            separator = "。",
            chunk_size = chunk_size,
            chunk_overlap  = chunk_overlap
        )
        docs = text_splitter.create_documents([text])

        chunks = [doc.page_content for doc in docs]
        return chunks
        
    def process_files_in_folder_1(self,input_folder,output_folder):
        """批量读取文件夹中的 .txt 文件，并使用 chunk_character 进行分块处理"""
        # 确保输出文件夹 "chunk" 存在
        os.makedirs(output_folder, exist_ok=True)
        
        for filename in os.listdir(input_folder):
            if filename.endswith('.txt'):
                file_path = os.path.join(input_folder, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read().replace('\n', '')

                # 使用 chunk_character 函数进行分块
                chunks = self.chunk_character(text, chunk_size=200, chunk_overlap=50)
                # 输出文件路径，将文件保存到 "chunk" 文件夹中
                output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_chunk.txt")
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    for i, chunk in enumerate(chunks):
                        output_file.write(f"Chunk {i + 1}:\n{chunk}\n\n")
                
                print(f"处理完成：{filename}，分块结果已写入 {output_file_path}")
                
    # 使用正则表达式进行分句
    def split_sentences(self,text):
        sentence_endings = re.compile(r'(?<=[。！？])')
        sentences = sentence_endings.split(text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]
    
    # 中文文本分词，去停用词
    def preprocess_text(self,text):
        sentences = self.split_sentences(text)   
        processed_sentences = []
        for sentence in sentences:
            words = jieba.cut(sentence)  # 使用jieba进行中文分词
            filtered_words = [word for word in words if  len(word) > 1] 
            processed_sentences.append(filtered_words)
        return processed_sentences, sentences
    
    # 基于语义相似度的递归合并分块
    def recursive_similarity_chunking(self,sentences, embeddings, similarity_threshold=0.7, max_chunk_length=3):
        chunks = []
        current_chunk = [sentences[0]]
        current_chunk_length = 1

        for i in range(1, len(sentences)):
            similarity = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
            if similarity >= similarity_threshold and current_chunk_length < max_chunk_length:
                current_chunk.append(sentences[i])
                current_chunk_length += 1
            else:
                chunks.append(current_chunk)
                current_chunk = [sentences[i]]
                current_chunk_length = 1

        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)

        return chunks
    
    # 获取BERT句子嵌入
    def get_sentence_embedding(self,sentence):
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        model = BertModel.from_pretrained('bert-base-chinese')
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
        with torch.no_grad():
            outputs = model(**inputs)
        # 提取最后一层的池化输出作为句子嵌入
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    # 批量处理文件夹中的所有.txt文件
    def process_folder_bert(self,input_folder, output_folder, similarity_threshold=0.7, max_chunk_length=5):
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)

        # 遍历文件夹中的所有.txt文件
        for filename in os.listdir(input_folder):
            if filename.endswith('.txt'):
                input_file_path = os.path.join(input_folder, filename)
                output_file_name = f"{os.path.splitext(filename)[0]}_chunk.txt"
                output_file_path = os.path.join(output_folder, output_file_name)

                # 读取文本文件
                with open(input_file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    text = text.replace('\n', '')

                # 获取每个句子的BERT向量表示
                original_sentences = self.split_sentences(text)
                embeddings = [self.get_sentence_embedding(sentence) for sentence in original_sentences]

                # 基于语义相似度进行递归合并分块
                chunks = self.recursive_similarity_chunking(original_sentences, embeddings, similarity_threshold, max_chunk_length)
                global chunk_num
                # 将分块结果保存到新文件中
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    for idx, chunk in enumerate(chunks):
                        f.write(f"\n--- 分块 {idx + 1} ---\n")
                        chunk_num += 1
                        for sentence in chunk:
                            f.write(f"{sentence}")
                        f.write("\n")
                print(chunk_num)
                print(f"已处理文件: {filename}, 结果保存在: {output_file_path}")
                
    # 读取.txt文件并进行处理
    def process_file(self,file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            text = text.replace('\n', '')
        return text
    
    # 创建LDA模型并进行主题建模
    def perform_lda(self,text):
        # 预处理文本
        processed_sentences, original_sentences = self.preprocess_text(text)
        
        # 创建字典和语料库
        dictionary = corpora.Dictionary(processed_sentences)
        corpus = [dictionary.doc2bow(sentence) for sentence in processed_sentences]
        
        # 训练LDA模型，选择合适的主题数
        lda_model = gensim.models.LdaMulticore(corpus, num_topics=20, id2word=dictionary, passes=15, workers=4)
        
        # 输出每个主题的关键词
        topics = lda_model.print_topics(num_words=5)
        print("LDA Topics:")
        for topic in topics:
            print(topic)
        
        # 根据主题分块
        topic_assigned_sentences = defaultdict(list)
        
        # 为每个句子分配主题
        for i, sentence in enumerate(processed_sentences):
            bow = dictionary.doc2bow(sentence)
            topic_distribution = lda_model.get_document_topics(bow)
            topic_id = max(topic_distribution, key=lambda x: x[1])[0]  # 获取最有可能的主题ID
            topic_assigned_sentences[topic_id].append(original_sentences[i])
        
        return topic_assigned_sentences

    # 输出每个主题的句子
    def save_topic_blocks_to_file(self,topic_assigned_sentences, output_file='lda_output.txt'):
        with open(output_file, 'w', encoding='utf-8') as f:
            for topic_id, sentences in topic_assigned_sentences.items():
                f.write(f"\n--- Topic {topic_id + 1} ---\n")
                for sentence in sentences:
                    f.write(f"{sentence}")
                f.write("\n")
        print(f"主题块已保存到文件: {output_file}")
    
    # 批量处理文件夹中的所有.txt文件
    def process_folder_lda(self,input_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)  # 如果文件夹不存在，创建它

        # 遍历文件夹中的所有.txt文件
        for filename in os.listdir(input_folder):
            if filename.endswith('.txt'):
                input_file_path = os.path.join(input_folder, filename)
                output_file_name = f"{os.path.splitext(filename)[0]}_chunk.txt"
                output_file_path = os.path.join(output_folder, output_file_name)
                try:
                    text = self.process_file(input_file_path)
                    topic_assigned_sentences = self.perform_lda(text)
                    self.save_topic_blocks_to_file(topic_assigned_sentences, output_file_path)
                    print(f"已处理文件: {filename}, 结果保存在: {output_file_path}")
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")
    # 读取文档内容
    def read_document(self,file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().replace('\n', '')

    # 调用LLM判断句子是否属于当前块
    def should_continue_in_block(self,current_block, new_sentence):
        content = f"当前块内容: {current_block}\n新句子: {new_sentence}\n问题: 我正在将一段内容进行分块处理，后续要搭建问答系统，块的大小不要超过三句话，新句子是否与当前块的内容分到一个块中？请回答是或否。"
        messages = [
            {'role': 'user', 'content': content}
        ]
        response = dashscope.Generation.call(
            api_key="sk-7556cc482a004993aa5bfa796c10d5bd",  # 替换为你的 API Key
            model="qwen-turbo",  # 使用的模型
            messages=messages,
            result_format='message'
        )
        if 'output' in response and len(response['output']) > 0:
            content = response['output']['choices'][0]['message']['content']  # 获取 content 字段
            return '是' in content.strip()  # 根据 content 判断是否继续
        else:
            print(f"错误：返回数据没有 'choices' 键，或者选择为空。响应内容: {response}")

    # 使用LLM进行文档分块
    def llm_based_chunking(self,document):
        sentences = document.split('。')  # 简单按句号分割文本
        sentences = [sentence.strip() + '。' for sentence in sentences if sentence.strip()]
        
        blocks = []
        current_block = []

        # 从头开始遍历句子
        for sentence in sentences:
            if not current_block:
                current_block.append(sentence)
            else:
                # 判断当前句子是否属于当前块
                current_block_text = ' '.join(current_block)
                if self.should_continue_in_block(current_block_text, sentence):
                    current_block.append(sentence)
                    # 检查块的句子数量是否超过3
                    if len(current_block) > 3:
                        # 将当前块加入结果，开始新块
                        blocks.append(current_block)
                        current_block = []  # 清空当前块
                else:
                    # 将当前块加入结果，开始新块
                    blocks.append(current_block)
                    current_block = [sentence]

        # 添加最后的块
        if current_block:
            blocks.append(current_block)

        return blocks

    # 输出分块结果
    def output_blocks(self,chunks, filename, output_dir):
        output_file_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_chunks.txt")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for idx, chunk in enumerate(chunks):
                f.write(f"\n--- 分块 {idx + 1} ---\n")
                for sentence in chunk:
                    f.write(f"{sentence}")
                f.write("\n")
        print(f"已处理文件: {filename}, 结果保存在: {output_file_path}")

    # 批量处理文件夹中的 .txt 文件
    def process_directory(self,input_dir, output_dir):
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 遍历文件夹中的所有 .txt 文件
        for filename in os.listdir(input_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(input_dir, filename)
                document = self.read_document(file_path)
                blocks = self.llm_based_chunking(document)
                self.output_blocks(blocks, filename, output_dir)
    

# 主函数
def main():
    input_folder = "E:\\数据挖掘\\A榜\\ftw" 
    output_folder = "E:\\数据挖掘\\A榜\\chunk" 
    # method可以为"charactor"、"bert"、"lda"、"llm"
    method="bert"
    Corpus=MyCorpus(input_folder,output_folder,method)
    Corpus.process_files()

if __name__ == "__main__":
    main()
