

PYTHON_EXEC="/root/miniconda3/envs/ai_writing_v2/bin/python"
SCRIPT_PATH="/root/paper/AI_Literary_Creation_Engine/src/train.py"
LOG_FILE="/root/paper/AI_Literary_Creation_Engine/train_log.txt"


mkdir -p /root/paper/AI_Literary_Creation_Engine/processed_data/training

echo "Starting training process..."
echo "Python: $PYTHON_EXEC"
echo "Script: $SCRIPT_PATH"
echo "Logging to: $LOG_FILE"

nohup "$PYTHON_EXEC" -u "$SCRIPT_PATH" \
    --dataset_path "/root/paper/AI_Literary_Creation_Engine/processed_data/training/fine_tuning_data.jsonl" \
    --output_dir "/root/paper/AI_Literary_Creation_Engine/results_checkpoints" \
    > "$LOG_FILE" 2>&1 &
echo "Training process started in background."
