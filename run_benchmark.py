import os
import subprocess

ENV = os.environ.copy()
ENV["HF_HOME"] = "/root/autodl-tmp/hf_cache"
ENV["DATA_PATH"] = "/root/paper/AI_Literary_Creation_Engine/processed_data/training/fine_tuning_data.jsonl"
ENV["ADAPTER"] = "/root/autodl-tmp/results_checkpoints/checkpoint-68700"

experiments = [
    ("Ours_Collaborative", []),
    ("Ablation_StandardDecoding", ["--standard_decoding"]),
    ("Baseline_BaseModel", ["--no_lora", "--standard_decoding"])
]

BASE_CMD = [
    "/root/miniconda3/envs/ai_writing_v2/bin/python",
    "-u",
    "/root/paper/AI_Literary_Creation_Engine/src/evaluate.py",
    "--data_path", ENV["DATA_PATH"],
    "--adapter_path", ENV["ADAPTER"],
    "--num_samples", "35"
]

print("Starting Full Benchmark (35 samples per experiment)...")
for name, args in experiments:
    print(f">>> Running Experiment: {name}")
    cmd = BASE_CMD + ["--run_name", name] + args
    try:
        with open(f"log_{name}.txt", "w") as f:
            subprocess.run(cmd, env=ENV, stdout=f, stderr=subprocess.STDOUT, check=True)
        print(f"    Finished {name} successfully.")
    except subprocess.CalledProcessError as e:
        print(f"    Error running {name}: {e}")

print("All experiments completed. Check experiment_results.csv")
