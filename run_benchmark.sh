
set -e


export HF_HOME=/root/autodl-tmp/hf_cache
export DATA_PATH=/root/paper/AI_Literary_Creation_Engine/processed_data/annotated/expert_annotated_dataset.jsonl
export ADAPTER=/root/autodl-tmp/results_checkpoints/checkpoint-68700
PYTHON_EXEC=/root/miniconda3/envs/ai_writing_v2/bin/python
SCRIPT_PATH=/root/paper/AI_Literary_Creation_Engine/src/evaluate.py


if [ -f experiment_results.csv ]; then
    mv experiment_results.csv experiment_results_backup_$(date +%s).csv
    echo "Backed up old results."
fi

echo "========================================================"
echo "STARTING FULL BENCHMARK (100 Samples per Experiment)"
echo "Estimated time: ~75 minutes"
echo "========================================================"


run_exp() {
    NAME=$1
    ARGS=$2
    echo "[$(date)] Starting Experiment: $NAME"
    $PYTHON_EXEC -u $SCRIPT_PATH \
        --data_path $DATA_PATH \
        --adapter_path $ADAPTER \
        --num_samples 100 \
        --run_name "$NAME" \
        $ARGS \
        > "log_${NAME}.txt" 2>&1

    if [ $? -eq 0 ]; then
        echo "[$(date)] Finished Successfully: $NAME"
    else
        echo "[$(date)] FAILED: $NAME. Check log_${NAME}.txt"
        exit 1
    fi
}


run_exp "Ours_Collaborative" ""


run_exp "Ablation_StandardDecoding" "--standard_decoding"


run_exp "Baseline_BaseModel" "--no_lora --standard_decoding"

echo "========================================================"
echo "ALL EXPERIMENTS COMPLETED."
echo "Results saved to: experiment_results.csv"
echo "========================================================"
