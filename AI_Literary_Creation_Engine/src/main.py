






import torch
import sys
import os
import argparse


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import MODEL_CONFIG, RAG_CONFIG
from src.model_loader import LLMLoader
from src.rag_engine import KnowledgeBase
from src.decoding import CollaborativeDecoding, SentimentConstraintLogitsProcessor

def main():
    parser = argparse.ArgumentParser(description="AI Literary Engine")
    parser.add_argument("--prompt", type=str, default="The moonlight reflects on the lake", help="The literary prompt")
    parser.add_argument("--target_sentiment", type=str, default="sadness", help="Target sentiment (joy, sadness, etc.)")
    parser.add_argument("--data_path", type=str, default="processed_data/output.jsonl", help="Path to JSONL data for RAG (if empty, skip indexing)")
    parser.add_argument("--skip_indexing", action="store_true", help="Skip adding docs to FAISS index")
    parser.add_argument("--adapter_path", type=str, default="/root/paper/AI_Literary_Creation_Engine/results_checkpoints", help="Path to LoRA adapter")
    args = parser.parse_args()

    print("=== Initializing AI Literary Creation Engine ===")


    loader = LLMLoader()
    model, tokenizer = loader.load_base_model(use_lora=True, adapter_path=args.adapter_path)
    model.eval()


    kb = KnowledgeBase()
    if not args.skip_indexing and os.path.exists(args.data_path):

        if kb.index.ntotal == 0:
            print(f"Indexing data from {args.data_path}...")
            kb.add_documents(args.data_path)


    decoder = CollaborativeDecoding(model, tokenizer, kb)


    print(f"\nUser Prompt: {args.prompt}")
    print(f"Target Sentiment: {args.target_sentiment}")


    print("\n>>> Generating with Factual Enhancement Decoding...")
    generated_text = decoder.generate_literary_piece(
        prompt=args.prompt,
        target_sentiment=args.target_sentiment,
        max_new_tokens=400
    )


    if "### Response:\n" in generated_text:
        final_output = generated_text.split("### Response:\n")[-1].strip()
    else:
        final_output = generated_text

    print("\n" + "="*50)
    print("FINAL OUTPUT:")
    print("="*50)
    print(final_output)
    print("="*50)

if __name__ == "__main__":
    main()