


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import MODEL_CONFIG

class LLMLoader:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

    def load_base_model(self, use_lora=True, adapter_path=None):
        print(f"Loading Base Model: {MODEL_CONFIG['base_model']}...")


        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CONFIG["base_model"],
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:


            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id




        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG["base_model"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        if use_lora:
            if adapter_path and os.path.exists(adapter_path):
                print(f"Loading Fine-Tuned LoRA Adapter from: {adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
            else:
                print("Applying New LoRA Adapter configuration (for training)...")


                peft_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                self.model = get_peft_model(self.model, peft_config)

            self.model.print_trainable_parameters()

        return self.model, self.tokenizer

if __name__ == "__main__":
    loader = LLMLoader()

    print("LLM Loader initialized.")