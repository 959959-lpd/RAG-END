import json
import os
from pathlib import Path
from typing import List, Dict


BASE_DIR = Path(__file__).resolve().parent.parent

ANNOTATED_FILE = BASE_DIR / "processed_data" / "output_v2.jsonl"
TRAINING_DATA_DIR = BASE_DIR / "processed_data" / "training"
OUTPUT_FILE = TRAINING_DATA_DIR / "fine_tuning_data.jsonl"

def construct_prompt(entry: Dict) -> Dict:
    """
    Constructs a training example from an annotation entry.
    Formats the constraints into the instruction/input to guide the model.
    """


    metadata = entry.get("metadata", {})


    base_instruction = "Write a poem."
    target_emotion = metadata.get("sentiment", "Neutral")
    intensity = metadata.get("intensity", 3)

    if target_emotion and target_emotion.lower() != "neutral":

        instruction = f"{base_instruction} Ensure the tone is distinctly {target_emotion} (Intensity: {intensity}/5)."
    else:
        instruction = base_instruction



    input_context_parts = []
    if target_emotion:
        input_context_parts.append(f"Target Emotion: {target_emotion}")


    imagery = metadata.get("imagery", [])
    if imagery:

        kw_str = ", ".join(imagery[:5])
        input_context_parts.append(f"Key Imagery: {kw_str}")

    input_context = "\n".join(input_context_parts)


    output_text = entry.get("text", "")



    return {
        "instruction": instruction,
        "input": input_context,
        "output": output_text
    }

def main():
    if not ANNOTATED_FILE.exists():
        print(f"Error: Annotated file not found at {ANNOTATED_FILE}")
        return


    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

    print(f"Reading from: {ANNOTATED_FILE}")
    print("Converting data...")

    processed_count = 0
    with open(ANNOTATED_FILE, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:

        for line in infile:
            if not line.strip():
                continue

            try:
                entry = json.loads(line)
                training_example = construct_prompt(entry)


                outfile.write(json.dumps(training_example, ensure_ascii=False) + "\n")
                processed_count += 1


                if processed_count <= 3:
                    print(f"\n--- Example {processed_count} ---")
                    print(f"INSTRUCTION: {training_example['instruction']}")
                    print(f"INPUT:\n{training_example['input']}")
                    print(f"OUTPUT: {training_example['output'][:100]}...")

            except json.JSONDecodeError:
                continue

    print(f"\nSuccessfully converted {processed_count} examples.")
    print(f"Training data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
