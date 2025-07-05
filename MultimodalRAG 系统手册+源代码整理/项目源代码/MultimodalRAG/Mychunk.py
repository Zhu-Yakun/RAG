import os
import re
import json
import jieba
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models
import torch

class MyCorpus:
    def __init__(self, input_folder, output_folder, method,
                 similarity_threshold=0.2, max_chunk_length=5,
                 lda_num_topics=20, lda_passes=15):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.method = method.lower()
        self.similarity_threshold = similarity_threshold
        self.max_chunk_length = max_chunk_length
        self.lda_num_topics = lda_num_topics
        self.lda_passes = lda_passes
        self.all_chunks = []  # 聚合集合

        # 准备句子嵌入模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sentencizer = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2',
            device=self.device
        )

    def process_files(self):
        os.makedirs(self.output_folder, exist_ok=True)
        # 根据 method 处理
        if self.method == "charactor":
            self._process_character_folder()
        elif self.method == "bert":
            self._process_bert_folder()
        elif self.method == "lda":
            self._process_lda_folder()
        else:
            print("请输入正确的分块方法: charactor, bert, lda")
            return

        # 输出所有块到一个 JSON
        out_path = os.path.join(self.output_folder, "all_chunks.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(self.all_chunks, f, ensure_ascii=False, indent=2)
        print(f"已输出所有块至 {out_path}, 共 {len(self.all_chunks)} 条记录")

    # ---------- 固定字符分块 ----------
    def chunk_character(self, text, chunk_size=200, chunk_overlap=50):
        from langchain.text_splitter import CharacterTextSplitter
        splitter = CharacterTextSplitter(
            separator="。",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        docs = splitter.create_documents([text])
        return [doc.page_content for doc in docs]

    def _process_character_folder(self):
        for fn in os.listdir(self.input_folder):
            if not fn.endswith('.txt'): continue
            text = open(os.path.join(self.input_folder, fn), encoding='utf-8').read().replace('\n','')
            chunks = self.chunk_character(text)
            self._add_chunks(fn, chunks)

    # ---------- BERT/Sentence-BERT 分块 ----------
    def split_sentences(self, text):
        pattern = re.compile(r'(?<=[。！？])')
        parts = pattern.split(text)
        return [p.strip() for p in parts if p.strip()]

    def _process_bert_folder(self):
        for fn in os.listdir(self.input_folder):
            if not fn.endswith('.txt'): continue
            text = open(os.path.join(self.input_folder, fn), encoding='utf-8').read().replace('\n','')
            sentences = self.split_sentences(text)
            embeddings = self.sentencizer.encode(
                sentences, convert_to_numpy=True, show_progress_bar=False
            )
            chunks = []
            idxs = [0]
            for idx in range(1, len(sentences)):
                block_emb = embeddings[idxs].mean(axis=0, keepdims=True)
                sim = cosine_similarity(block_emb, embeddings[idx:idx+1])[0][0]
                if sim >= self.similarity_threshold and len(idxs) < self.max_chunk_length:
                    idxs.append(idx)
                else:
                    chunks.append([sentences[i] for i in idxs])
                    idxs = [idx]
            if idxs:
                chunks.append([sentences[i] for i in idxs])
            self._add_chunks(fn, chunks)

    # ---------- LDA 分块 ----------
    def preprocess_text(self, text):
        sents = self.split_sentences(text)
        processed = []
        for s in sents:
            words = jieba.cut(s)
            processed.append([w for w in words if len(w)>1])
        return processed, sents

    def perform_lda(self, text):
        proc, orig = self.preprocess_text(text)
        dictionary = corpora.Dictionary(proc)
        corpus = [dictionary.doc2bow(s) for s in proc]
        lda = models.LdaMulticore(
            corpus, num_topics=self.lda_num_topics,
            id2word=dictionary, passes=self.lda_passes, workers=4
        )
        topic_sents = defaultdict(list)
        for i, bow in enumerate(corpus):
            dist = lda.get_document_topics(bow)
            topic = max(dist, key=lambda x:x[1])[0]
            topic_sents[topic].append(orig[i])
        return list(topic_sents.values())

    def _process_lda_folder(self):
        for fn in os.listdir(self.input_folder):
            if not fn.endswith('.txt'): continue
            text = open(os.path.join(self.input_folder, fn), encoding='utf-8').read()
            chunks = self.perform_lda(text)
            self._add_chunks(fn, chunks)

    # ---------- 添加并聚合块 ----------
    def _add_chunks(self, filename, chunks):
        base = os.path.splitext(filename)[0]
        for i, chunk in enumerate(chunks, start=1):
            self.all_chunks.append({
                "name": f"{base}{i}",
                "description": ''.join(chunk)
            })

# 主函数
if __name__ == "__main__":
    input_folder = "output_txt"
    output_folder = "chunk"
    method = "bert"  # 可选: charactor, bert, lda
    MyCorpus(input_folder, output_folder, method).process_files()
