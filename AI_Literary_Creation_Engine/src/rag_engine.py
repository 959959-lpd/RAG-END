



import faiss
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import MODEL_CONFIG, RAG_CONFIG

class KnowledgeBase:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        print(f"Loading Embedding Model ({MODEL_CONFIG['embedding_model']})...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['embedding_model'])
        self.model = AutoModel.from_pretrained(MODEL_CONFIG['embedding_model']).to(self.device).eval()

        self.dim = MODEL_CONFIG['vector_dim']


        if os.path.exists(RAG_CONFIG['index_path']):
            print(f"Loading existing FAISS index from {RAG_CONFIG['index_path']}...")
            self.index = faiss.read_index(RAG_CONFIG['index_path'])
        else:
            print("Creating new Flat FAISS index (inner product)...")

            self.index = faiss.IndexFlatIP(self.dim)

        self.metadata = []
        if os.path.exists(RAG_CONFIG['metadata_path']):
            try:
                with open(RAG_CONFIG['metadata_path'], 'r', encoding='utf-8') as f:
                    for line in f:
                        self.metadata.append(json.loads(line))
            except Exception as e:
                print(f"Warning: Failed load metadata: {e}")

    def embed_text(self, texts, batch_size=32):
        """批量 Embedding"""
        if isinstance(texts, str): texts = [texts]

        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(self.device)

            with torch.no_grad():
                model_output = self.model(**encoded_input)

                sentence_embeddings = model_output.last_hidden_state[:, 0]


            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            all_embeds.append(sentence_embeddings.cpu().numpy())

        return np.concatenate(all_embeds, axis=0)

    def add_documents(self, data_path):
        """从 JSONL 读取数据并添加进 FAISS"""
        print(f"Adding documents from {data_path}...")

        texts = []
        metadata = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)


                text = item.get('text', '')
                if not text: continue


                if 'metadata' in item:
                    meta = item['metadata']
                elif 'fine_grained_emotion' in item:

                    meta = {
                        "sentiment": item['fine_grained_emotion']['primary_emotion'],
                        "intensity": item['fine_grained_emotion']['intensity_score'],
                        "imagery": [m['imagery'] for m in item['fine_grained_emotion']['imagery_emotion_map']]
                    }
                else:
                    meta = {}

                meta['original_text'] = text

                texts.append(text)
                metadata.append(meta)

        if not texts: return


        embeddings = self.embed_text(texts)


        self.index.add(embeddings)


        self.metadata.extend(metadata)


        self._save()
        print(f"Added {len(texts)} documents. Total in index: {self.index.ntotal}")

    def _save(self):
        index_dir = os.path.dirname(RAG_CONFIG['index_path'])
        if index_dir and not os.path.exists(index_dir):
            os.makedirs(index_dir, exist_ok=True)

        faiss.write_index(self.index, RAG_CONFIG['index_path'])
        with open(RAG_CONFIG['metadata_path'], 'w', encoding='utf-8') as f:
            for m in self.metadata:
                f.write(json.dumps(m) + "\n")

    def retrieve(self, query, top_k=3, sentiment_guidance=None):
        """
        检索增强:
        query: 用户输入或当前生成片段
        sentiment_guidance: (Optional) 期望的情感对齐 (e.g. "sadness")

        返回: List[Dict] 包含文本和元数据
        """
        if not self.index.ntotal: return []

        query_vec = self.embed_text([query])


        D, I = self.index.search(query_vec, top_k * 3)

        results = []
        indices = I[0]
        scores = D[0]

        for idx in range(len(indices)):
            db_idx = indices[idx]
            if db_idx < 0: continue

            meta = self.metadata[db_idx]
            similarity = scores[idx]



            if sentiment_guidance:
                doc_sentiment = meta.get('sentiment')

                if doc_sentiment != sentiment_guidance and meta.get('confidence', 0) > 0.8:

                    continue

            results.append({
                "text": meta['original_text'],
                "metadata": meta,
                "score": float(similarity)
            })

            if len(results) >= top_k:
                break

        return results

if __name__ == "__main__":




    print("KnowledgeBase initialized.")