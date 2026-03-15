
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ai_writing_v2

echo "Converting processed annotations to training format..."
python /root/paper/AI_Literary_Creation_Engine/src/prepare_training_data.py
echo "Done."
