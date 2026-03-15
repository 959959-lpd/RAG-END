



import torch
from transformers import LogitsProcessor
import numpy as np
import warnings
from typing import List, Optional, Tuple, Dict
import math
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DECODING_PARAMS

class SentimentConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self,
                 tokenizer,
                 target_sentiment: str,
                 imagery_keywords: List[str],
                 guidance_scale: float = None,
                 penalty_scale: float = None,
                 device="cuda"):
        """
        :param tokenizer: 当前 LLM 的 Tokenizer
        :param target_sentiment: "joy", "sadness" 等
        :param imagery_keywords: 从 RAG 检索到的与当前情感强相关的意象词 (e.g. ["tear", "rain"])
        :param guidance_scale: 正向激励权重 (Defaults to DECODING_PARAMS['guidance_weight'])
        :param penalty_scale: 反向惩罚权重 (Defaults to DECODING_PARAMS['penalty_weight'])
        """
        self.tokenizer = tokenizer
        self.target_sentiment = target_sentiment

        self.guidance_scale = guidance_scale if guidance_scale is not None else DECODING_PARAMS.get("guidance_weight", 2.5)
        self.penalty_scale = penalty_scale if penalty_scale is not None else DECODING_PARAMS.get("penalty_weight", 1.5)

        self.imagery_keywords = imagery_keywords
        self.device = device



        self.positive_token_ids = self._get_tokens(imagery_keywords)





        opposite_map = {"sadness": ["joy", "smile", "laugh"],
                        "joy": ["sadness", "tear", "cry"],
                        "anger": ["calm", "peace"],
                        "fear": ["brave", "safe"]}

        neg_keywords = opposite_map.get(target_sentiment, [])
        self.negative_token_ids = self._get_tokens(neg_keywords)

        print(f"Initialized LogitsProcessor for {target_sentiment}. "
              f"Pos tokens: {len(self.positive_token_ids)}, Neg tokens: {len(self.negative_token_ids)}")

    def _get_tokens(self, words: List[str]) -> List[int]:



        ids = set()
        for w in words:

            tokens_space = self.tokenizer.encode(" " + w, add_special_tokens=False)
            if len(tokens_space) <= 2:
                ids.update(tokens_space)


            tokens = self.tokenizer.encode(w, add_special_tokens=False)
            if len(tokens) <= 2:
                ids.update(tokens)

        return list(ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:





        current_seq = input_ids[0].tolist()

        if self.positive_token_ids:

            recent_tokens = set(current_seq[-20:])
            effective_pos_tokens = [t for t in self.positive_token_ids if t not in recent_tokens]

            if effective_pos_tokens:

               pos_indices = torch.tensor(effective_pos_tokens, dtype=torch.long, device=scores.device)
               scores[:, pos_indices] += self.guidance_scale



        if self.negative_token_ids:

            neg_indices = torch.tensor(self.negative_token_ids, dtype=torch.long, device=scores.device)
            scores[:, neg_indices] -= self.penalty_scale

        return scores

class CollaborativeDecoding:

    def __init__(self, model, tokenizer, kb_module):
        self.model = model
        self.tokenizer = tokenizer
        self.kb = kb_module

    def generate_literary_piece(self, prompt, target_sentiment, max_new_tokens=400, use_collaborative=True):

        context_imagery = []
        if use_collaborative and self.kb:


            retrieved_docs = self.kb.retrieve(prompt, sentiment_guidance=target_sentiment)


            for doc in retrieved_docs:
                if 'imagery' in doc['metadata']:
                    context_imagery.extend(doc['metadata']['imagery'])


            context_imagery = list(set(context_imagery))





        prompt_str = prompt.strip()
        is_instruction = any(prompt_str.lower().startswith(kw) for kw in ["write", "describe", "analyze", "explain", "provide"])

        if use_collaborative:
            if is_instruction:
                full_prompt = (
                    f"### Instruction:\n{prompt_str} "
                    f"Tone: {target_sentiment}. Imagery: {', '.join(context_imagery[:3])}.\n\n"
                    f"### Response:\n"
                )
            else:
                full_prompt = (
                    f"### Instruction:\nWrite a short, poetic literary paragraph about \"{prompt_str}\". "
                    f"Evoke {target_sentiment}. Use imagery: {', '.join(context_imagery[:3])}.\n\n"
                    f"### Response:\n"
                )
        else:

            if is_instruction:
                full_prompt = (
                    f"### Instruction:\n{prompt_str} "
                    f"Tone: {target_sentiment}.\n\n"
                    f"### Response:\n"
                )
            else:
                full_prompt = (
                    f"### Instruction:\nWrite a short, poetic literary paragraph about \"{prompt_str}\". "
                    f"Evoke {target_sentiment}.\n\n"
                    f"### Response:\n"
                )

        inputs = self.tokenizer(full_prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)


        logits_processor_list = []
        if use_collaborative:
            logits_processor = SentimentConstraintLogitsProcessor(
                tokenizer=self.tokenizer,
                target_sentiment=target_sentiment,
                imagery_keywords=context_imagery,
                guidance_scale=DECODING_PARAMS.get("guidance_weight", 2.0),
                penalty_scale=DECODING_PARAMS.get("penalty_weight", 1.5)
            )
            logits_processor_list.append(logits_processor)

        from transformers import LogitsProcessorList


        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=400,
            logits_processor=LogitsProcessorList(logits_processor_list),
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3
        )


        new_tokens = output[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return generated_text

if __name__ == "__main__":
    print("Decoding strategy ready.")