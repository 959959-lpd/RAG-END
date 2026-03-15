



import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
import sys
import os
import nltk
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer


try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"

from src.config import MODEL_CONFIG, EMOTION_LABELS

import argparse
from src.model_loader import LLMLoader
from src.rag_engine import KnowledgeBase
from src.decoding import CollaborativeDecoding

class ExperimentEvaluator:
    def __init__(self, model_loader=None, knowledge_base=None, adapter_path=None, use_lora=True, use_collaborative=True):
        print("Initializing Evaluation Metrics Models...")
        self.device = 0 if torch.cuda.is_available() else -1
        self.use_collaborative = use_collaborative



        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_name = "/root/autodl-tmp/hf_cache/hub/models--bhadresh-savani--bert-base-uncased-emotion/snapshots/04e32b0ce2cd9c6cc36daffabeda36857058da63"
        print(f"Loading Emotion Model from {model_name} (Offline mode)...")

        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)

        self.eval_model = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=None,
            device=self.device
        )


        self.generator = None
        if model_loader:
             print(f"Loading Generation Model for Benchmark (LoRA={use_lora})...")
             model, tokenizer = model_loader.load_base_model(use_lora=use_lora, adapter_path=adapter_path)
             model.eval()
             self.generator = CollaborativeDecoding(model, tokenizer, knowledge_base)

        print("Evaluation models ready.")

    def generate_text(self, prompt, target_sentiment):
        """调用生成模型生成单条文本"""
        if not self.generator:
            return "No generator loaded."

        full_output = self.generator.generate_literary_piece(
            prompt,
            target_sentiment=target_sentiment,
            max_new_tokens=400,
            use_collaborative=self.use_collaborative
        )

        marker = "Assistant:"
        if marker in full_output:
            generated = full_output.split(marker)[-1].strip()
        elif "### Response:" in full_output:
            generated = full_output.split("### Response:")[-1].strip()
        else:
            generated = full_output

        return generated

    def calculate_metrics(self, generated_texts, target_sentiments, reference_texts=None):
        """
        核心评估函数 - Updated with BLEU, ROUGE, Distinct-N
        :param generated_texts: List[str] 模型生成的文本列表
        :param target_sentiments: List[str] 对应的目标情感标签 (e.g. "sadness")
        :param reference_texts: List[str] (Optional) 参考文本列表 (Ground Truth)
        :return: Dict 包含 ECFR, EER, BLEU, ROUGE, Distinct-2 等
        """

        total_samples = len(generated_texts)
        if total_samples == 0: return {}

        metrics = {}


        correct_sentiment_count = 0
        extreme_emotion_count = 0
        emotional_drift_scores = []

        print(f"Evaluating Emotions & Diversity for {total_samples} samples...")

        for text, target in tqdm(zip(generated_texts, target_sentiments), total=total_samples, desc="Emotion Analysis"):

            trunc_text = text[:512]

            try:

                results = self.eval_model(trunc_text)[0]



                top_pred = max(results, key=lambda x: x['score'])
                pred_label = top_pred['label']
                pred_score = top_pred['score']


                if self._is_emotion_match(pred_label, target):
                    correct_sentiment_count += 1


                if not self._is_emotion_match(pred_label, target) and pred_score > 0.9:
                     extreme_emotion_count += 1


                target_score = next((x['score'] for x in results if x['label'] == target), 0.0)
                emotional_drift_scores.append(1.0 - target_score)
            except Exception as e:
                print(f"Error classifying text: {e}")

        metrics["ECFR"] = (correct_sentiment_count / total_samples) * 100
        metrics["EER"] = (extreme_emotion_count / total_samples) * 100
        metrics["Avg_Drift"] = np.mean(emotional_drift_scores) if emotional_drift_scores else 0.0


        metrics["Distinct-1"] = self._calculate_distinct_n(generated_texts, n=1)
        metrics["Distinct-2"] = self._calculate_distinct_n(generated_texts, n=2)


        if reference_texts and len(reference_texts) == total_samples:
            print("Calculating Reference-based Metrics (BLEU, ROUGE)...")
            try:





                refs_tokenized = [[nltk.word_tokenize(ref)] for ref in reference_texts]
                hyps_tokenized = [nltk.word_tokenize(hyp) for hyp in generated_texts]

                bleu_score = corpus_bleu(refs_tokenized, hyps_tokenized)
                metrics["BLEU-4"] = bleu_score * 100


                scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
                rouge_scores = {'rouge1': [], 'rougeL': []}

                for ref, hyp in zip(reference_texts, generated_texts):
                    scores = scorer.score(ref, hyp)
                    rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                    rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)

                metrics["ROUGE-1"] = np.mean(rouge_scores['rouge1']) * 100
                metrics["ROUGE-L"] = np.mean(rouge_scores['rougeL']) * 100

            except Exception as e:
                print(f"Error calculating reference metrics: {e}")

        return metrics

    def _calculate_distinct_n(self, texts, n=2):
        """Calculate Distinct-N ratio"""
        all_ngrams = []
        for text in texts:
            tokens = nltk.word_tokenize(text.lower())
            if len(tokens) < n: continue
            ngrams = list(nltk.ngrams(tokens, n))
            all_ngrams.extend(ngrams)

        if not all_ngrams: return 0.0

        unique_ngrams = set(all_ngrams)
        return len(unique_ngrams) / len(all_ngrams)


    def _is_emotion_match(self, pred, target):
        """宽松的情感匹配逻辑"""
        if pred == target: return True


        mapping = {
            "joy": ["love", "joy", "surprise"],
            "sadness": ["sadness", "fear", "anger"],
            "anger": ["anger", "disgust"],
            "fear": ["fear", "sadness"]
        }
        return pred in mapping.get(target, [])

    def run_benchmark(self, test_file_path, num_samples=10, run_name="experiment"):
        """
        运行自动化 Benchmark - extracting prompts, targets, and references
        :param test_file_path: 包含 prompt 和 target_sentiment 的 JSONL 文件
        :param run_name: 实验名称
        """
        print(f"Loading test data from {test_file_path}...")
        test_data = []
        with open(test_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples: break
                try:
                    test_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        generated_texts = []
        target_sentiments = []
        reference_texts = []

        print(f"Starting Generation for {len(test_data)} samples...")
        for item in tqdm(test_data):

            prompt = ""
            target = ""
            reference = ""

            if 'architecture_binding' in item:
                prompt = item['architecture_binding']['query_intent']
                target = item['architecture_binding']['decoder_constraints']['target_emotion']

            elif 'instruction' in item and 'input' in item:
                prompt = item['instruction']


                input_text = item['input']
                if "Target Emotion:" in input_text:
                    try:
                        target = input_text.split("Target Emotion:")[1].split("\n")[0].strip()
                    except:
                        target = "joy"
                else:
                    target = "joy"

                reference = item.get('output', '')
            else:
                prompt = item.get('prompt', 'Write a poem')
                target = item.get('target_sentiment', 'joy')



            for key in ['completion', 'text', 'response', 'output', 'target_text', 'content']:
                 if key in item and isinstance(item[key], str):
                     reference = item[key]
                     break
            if not reference and 'messages' in item:
                 for msg in item['messages']:
                      if msg['role'] == 'assistant':
                           reference = msg['content']
                           break


            basic_target = self._map_to_basic_emotion(target)


            if self.generator:
                print(f"\nPrompt: {prompt[:50]}... | Target: {target} ({basic_target})")
                try:
                    gen_text = self.generate_text(prompt, basic_target)
                except Exception as e:
                    print(f"Error generating text: {e}")
                    gen_text = ""

                if gen_text.startswith(prompt):
                    gen_text = gen_text[len(prompt):]
                print(f"  Generated: {gen_text[:50]}...")
            else:
                gen_text = item.get('generated_text', '')

            generated_texts.append(gen_text)
            target_sentiments.append(basic_target)
            if reference:
                reference_texts.append(reference)



        refs_to_pass = reference_texts if len(reference_texts) == len(generated_texts) else None
        if len(reference_texts) > 0 and len(reference_texts) != len(generated_texts):
             print(f"Warning: Found {len(reference_texts)} references for {len(generated_texts)} samples. Skipping overlap metrics.")

        metrics = self.calculate_metrics(generated_texts, target_sentiments, reference_texts=refs_to_pass)


        import csv
        results_file = "/root/experiment_results.csv"
        file_exists = os.path.isfile(results_file)

        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Experiment", "ECFR", "EER", "Avg_Drift", "Distinct-1", "Distinct-2"])

            writer.writerow([
                run_name,
                f"{metrics.get('ECFR', 0):.2f}",
                f"{metrics.get('EER', 0):.2f}",
                f"{metrics.get('Avg_Drift', 0):.4f}",
                f"{metrics.get('Distinct-1', 0):.4f}",
                f"{metrics.get('Distinct-2', 0):.4f}"
            ])
        print(f"Results appended to {results_file}")

        print("\n" + "="*40)
        print("Experiment Results (Benchmark)")
        print("="*40)

        if metrics:
            print(f"ECFR (Emotional Correctness): {metrics.get('ECFR', 0):.2f}%")
            print(f"EER  (Extreme Emotion Rate):  {metrics.get('EER', 0):.2f}%")
            print(f"Drift (Avg Deviation):       {metrics.get('Avg_Drift', 0):.4f}")
            print(f"Distinct-2 (Diversity):      {metrics.get('Distinct-2', 0):.4f}")

            if 'BLEU-4' in metrics:
                print(f"BLEU-4 (Overlap):            {metrics.get('BLEU-4', 0):.2f}")
                print(f"ROUGE-L (Structure):         {metrics.get('ROUGE-L', 0):.2f}")
        print("="*40)


        metrics['Experiment'] = run_name
        output_file = "experiment_results.csv"

        header = not os.path.exists(output_file)

        df = pd.DataFrame([metrics])

        if 'Experiment' in df.columns:
            cols = ['Experiment'] + [c for c in df.columns if c != 'Experiment']
            df = df[cols]

        df.to_csv(output_file, mode='a', header=header, index=False)
        print(f"Results saved to {output_file} (Run: {run_name})")

    def _map_to_basic_emotion(self, emotion):
        """Map complex emotions to Ekman's 6 basic emotions"""
        emotion = emotion.lower()
        mapping = {
            "gratitude": "love",
            "resilience": "joy",
            "despair": "sadness",
            "hostility": "anger",
            "connection": "love",
            "reflection": "joy",
            "steadfastness": "joy",
            "trepidation": "fear",
            "awe": "surprise",
            "defiance": "anger",
            "remorse": "sadness",
            "penitent": "sadness",
            "penitent remorse": "sadness",
            "formality": "joy",
            "analytical": "joy",
            "neutral": "joy",
            "scholarly analysis": "joy",
            "trust": "joy",
            "somber reflection": "sadness",
            "creative anticipation": "joy",
            "longing": "sadness",
            "grief": "sadness",
            "ambivalence": "fear",
            "historical awe": "surprise",
            "romanticism": "love",
            "contemplation": "joy",
            "contemplative": "joy",
            "disapproval": "anger",
            "reverence": "love",
            "melancholy": "sadness"
        }

        if emotion in mapping:
            return mapping[emotion]




        try:
            if hasattr(self, 'eval_model') and self.eval_model:

                result = self.eval_model(emotion[:512])[0]
                return result['label']
        except Exception as e:

            pass

        return emotion

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to test dataset (JSONL)")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument("--run_name", type=str, default="experiment", help="Name of the experiment run")
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA (use base model only)")
    parser.add_argument("--standard_decoding", action="store_true", help="Disable Collaborative Decoding (use standard generation)")

    args = parser.parse_args()


    loader = LLMLoader()
    kb = KnowledgeBase()


    print(f"Running Experiment: {args.run_name}")
    print(f"Config: LoRA={not args.no_lora}, Collaborative={not args.standard_decoding}")

    evaluator = ExperimentEvaluator(
        model_loader=loader,
        knowledge_base=kb,
        adapter_path=args.adapter_path,
        use_lora=not args.no_lora,
        use_collaborative=not args.standard_decoding
    )


    evaluator.run_benchmark(args.data_path, num_samples=args.num_samples, run_name=args.run_name)