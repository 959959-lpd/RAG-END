
export HF_HOME=/root/autodl-tmp/hf_cache
DATA_PATH="/root/paper/AI_Literary_Creation_Engine/processed_data/selected_test_prompts.jsonl"
ADAPTER="/root/paper/AI_Literary_Creation_Engine/results_checkpoints"

echo "=== Starting Experiments - SOTA & Ablation ==="


rm -f /root/experiment_results.csv /root/benchmark_debug.log


echo "Running Baseline (No LoRA)..." >> /root/benchmark_debug.log
/root/miniconda3/envs/ai_writing_v2/bin/python -u /root/paper/AI_Literary_Creation_Engine/src/evaluate.py \
    --data_path "$DATA_PATH" \
    --adapter_path "$ADAPTER" \
    --num_samples 10 \
    --run_name "Baseline_ZeroShot" \
    --no_lora \
    --standard_decoding >> /root/benchmark_debug.log 2>&1


echo "Running Ablation (No Collaborative)..." >> /root/benchmark_debug.log
/root/miniconda3/envs/ai_writing_v2/bin/python -u /root/paper/AI_Literary_Creation_Engine/src/evaluate.py \
    --data_path "$DATA_PATH" \
    --adapter_path "$ADAPTER" \
    --num_samples 10 \
    --run_name "Ablation_Standard" \
    --standard_decoding >> /root/benchmark_debug.log 2>&1


echo "Running Ours (Collaborative)..." >> /root/benchmark_debug.log
/root/miniconda3/envs/ai_writing_v2/bin/python -u /root/paper/AI_Literary_Creation_Engine/src/evaluate.py \
    --data_path "$DATA_PATH" \
    --adapter_path "$ADAPTER" \
    --num_samples 10 \
    --run_name "Ours_Collaborative" >> /root/benchmark_debug.log 2>&1

echo "Experiments Complete. Results:"
cat /root/experiment_results.csv

