

PYTHON_EXEC="/root/miniconda3/envs/ai_writing_v2/bin/python"
SCRIPT_PATH="/root/paper/AI_Literary_Creation_Engine/src/data_processing.py"
LOG_FILE="/root/paper/AI_Literary_Creation_Engine/annotation_log.txt"

echo "Starting annotation process..."
echo "Python: $PYTHON_EXEC"
echo "Script: $SCRIPT_PATH"
echo "Logging to: $LOG_FILE"

nohup "$PYTHON_EXEC" -u "$SCRIPT_PATH" > "$LOG_FILE" 2>&1 &
echo "Process started in background."
