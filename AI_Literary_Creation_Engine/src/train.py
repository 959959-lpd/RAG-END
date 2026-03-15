import os
import sys
import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import MODEL_CONFIG

def formatting_prompts_func(example):
    """
    Format the training data into the prompt style expected by the model.
    Change this function depending on your model's chat template.
    Current template: simple instruction-following.
    """
    output_texts = []

    for i in range(len(example['instruction'])):
        instruction = example['instruction'][i]
        input_context = example['input'][i]
        response = example['output'][i]



        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_context}\n\n### Response:\n{response}"
        output_texts.append(text)
    return output_texts

def train(args):
    """
    Main training function using QLoRA for memory efficiency.
    """
    print(f"Loading Base Model: {args.model_name}")


    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )


    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"": 0}
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1


    model = prepare_model_for_kbit_training(model)


    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "up_proj",
            "o_proj",
            "k_proj",
            "down_proj",
            "gate_proj",
            "v_proj",
        ],
    )


    print(f"Loading Dataset from: {args.dataset_path}")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=25,
        learning_rate=args.learning_rate,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,

        lr_scheduler_type="constant",
        save_total_limit=2,
        report_to="tensorboard"
    )


    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
        packing=False,
    )


    print("Starting Training...")
    trainer.train()


    print(f"Saving Fine-Tuned Model to {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek/Llama for Literary Creation")



    parser.add_argument("--model_name", type=str, default=MODEL_CONFIG.get("base_model", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"), help="Hugging Face model ID or local path")
    parser.add_argument("--dataset_path", type=str, default="processed_data/training/fine_tuning_data.jsonl", help="Path to the JSONL training data")
    parser.add_argument("--output_dir", type=str, default="./results_checkpoints", help="Directory to save checkpoints and final model")


    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")

    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)

    train(args)
