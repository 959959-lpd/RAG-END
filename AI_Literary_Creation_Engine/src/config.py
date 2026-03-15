import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

EMOTION_LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}


IMAGERY_KEYWORDS = {
    "sadness": ["tear", "darkness", "rain", "shadow", "grave", "moon", "silence", "crow"],
    "joy": ["sun", "light", "smile", "flower", "bird", "morning", "spring", "gold"],
    "love": ["heart", "rose", "kiss", "embrace", "dream", "star", "nightingale", "dove"],
    "anger": ["fire", "storm", "blood", "thunder", "knife", "fist", "red", "wolf"],

}

MODEL_CONFIG = {
    "base_model": "/root/autodl-tmp/hf_cache/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1",
    "embedding_model": "/root/autodl-tmp/hf_cache/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
    "sentiment_model": "/root/autodl-tmp/hf_cache/hub/models--bhadresh-savani--bert-base-uncased-emotion/snapshots/04e32b0ce2cd9c6cc36daffabeda36857058da63",
    "vector_dim": 1024,
    "max_length": 2048,
    "quantization": "4bit"
}

RAG_CONFIG = {
    "top_k": 3,
    "index_path": os.path.join(BASE_DIR, "knowledge_base/literary_faiss.index"),
    "metadata_path": os.path.join(BASE_DIR, "knowledge_base/metadata.jsonl")
}



DECODING_PARAMS = {
    "guidance_weight": 4.5,
    "penalty_weight": 2.5,
    "top_p": 0.9,
    "temperature": 0.7
}