import os
import json
import random
import re
import requests
import time
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
ANNOTATED_DATA_DIR = PROCESSED_DATA_DIR / "annotated"
CLEANED_FILES_PATTERN = "cleaned_*.txt"
OUTPUT_FILE = ANNOTATED_DATA_DIR / "expert_annotated_dataset.jsonl"
CHECKPOINT_FILE = ANNOTATED_DATA_DIR / "annotation_checkpoint.txt"


DEEPSEEK_API_KEY = "sk-57e2bd5be39946d4ae10876aa10c16f4"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"


MAX_WORKERS = 200
API_RATE_LIMIT_DELAY = 0.02


os.makedirs(ANNOTATED_DATA_DIR, exist_ok=True)

class AnnotationSchema:
    """
    Defines the structure for the dataset annotations, strictly following the
    architectural requirements of the AI Literary Creation Engine.
    """

    @staticmethod
    def create_entry(
        file_id: str,
        text_segment: str,
        meta: Dict[str, Any],
        architecture_binding: Dict[str, Any],
        fine_grained_emotion: Dict[str, Any],
        dual_constraints: Dict[str, Any],
        hallucination_labels: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "id": file_id,
            "text": text_segment,
            "meta": meta,



            "architecture_binding": architecture_binding,



            "fine_grained_emotion": fine_grained_emotion,



            "dual_constraints": dual_constraints,



            "hallucination_labels": hallucination_labels
        }

class DeepSeekAnnotator:
    """
    Uses DeepSeek V3 API to annotate literary texts with high precision.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.system_prompt = """
You are an expert literary critic and dataset annotator for an AI training pipeline.
 Your task is to analyze the provided poetry/literary text segment and generate a strict JSON object.

The Output MUST follow this exact JSON schema:
{
  "architecture_binding": {
    "query_intent": "User query that would prompt this text (e.g., 'Write a sad poem about nature')",
    "retrieval_keys": {
      "semantic_vector_input": "Key phrase for vector search",
      "entity_extraction": ["List", "of", "Entities"]
    },
    "decoder_constraints": {
      "target_emotion": "Primary Emotion identified",
      "avoid_sentiments": ["Opposite Emotion 1", "Opposite Emotion 2"]
    }
  },
  "fine_grained_emotion": {
    "primary_emotion": "Emotion Name (e.g., Melancholy, Awe)",
    "secondary_emotion": "Sub-emotion",
    "intensity_score": 1, // Integer 1-5 (5 is highest)
    "imagery_emotion_map": [
      {"imagery": "word/phrase", "aligned_emotion": "Emotion context", "correlation_score": 0.9}
    ]
  },
  "dual_constraints": {
    "compliance": {
      "safety_filter_passed": true, // Boolean
      "bias_level": "None/Low/Medium/High",
      "nsfw_score": 0.0 // Float 0.0-1.0
    },
    "factual_consistency": {
      "imagery_logic_check": "Pass/Fail",
      "context_flow_coherence": "High/Medium/Low",
      "logic_chain": "Step1 -> Step2 -> Step3 (e.g. Grief -> Acceptance)"
    }
  },
  "hallucination_labels": {
    "has_hallucination": false, // Boolean. Only set true if the text has blatant logic errors. Usually False for human text.
    "type": "None", // Or 'Emotional_Logic_Conflict' if true
    "description": "Short explanation",
    "severity": "None",
    "correction_target": null
  }
}
Note on "hallucination_labels": Since the input is human-written literature, "has_hallucination" should usually be false.
However, if the text contains archaic contradictions or metaphorical paradoxes that an AI might misinterpret as hallucination, label it as such but note it is intentional in description.
"""

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Calls DeepSeek API to analyze text.
        """
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Analyze this text:\n\n{text}"}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.3,
            "max_tokens": 1024
        }

        max_retries = 3

        time.sleep(API_RATE_LIMIT_DELAY + random.random() * 0.5)

        for attempt in range(max_retries):
            try:
                response = requests.post(DEEPSEEK_API_URL, headers=self.headers, json=payload, timeout=30)


                if response.status_code == 429:
                    wait_time = (attempt + 1) * 2 + random.uniform(1, 3)

                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                result = response.json()
                content = result['choices'][0]['message']['content']
                return json.loads(content)
            except Exception as e:

                time.sleep(2)


        return self._fallback_response()

    def _fallback_response(self):
        return {
            "architecture_binding": {"query_intent": "Error", "retrieval_keys": {}, "decoder_constraints": {}},
            "fine_grained_emotion": {"primary_emotion": "Error", "intensity_score": 0},
            "dual_constraints": {},
            "hallucination_labels": {"has_hallucination": False}
        }

def process_file(file_path: Path, annotator: DeepSeekAnnotator) -> List[Dict]:
    """
    Reads a cleaned file and produces annotated segments.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []



    segments = [s.strip() for s in content.split("\n\n") if len(s.strip()) > 50]

    annotated_entries = []
    file_id = file_path.stem.replace("cleaned_", "")



    segments_to_process = segments[:5]

    annotated_entries = []
    file_id = file_path.stem.replace("cleaned_", "")

    for idx, segment in enumerate(segments_to_process):
        analysis = annotator.analyze_text(segment)


        entry = AnnotationSchema.create_entry(
            file_id=f"{file_id}_{idx}",
            text_segment=segment,
            meta={"source_file": file_path.name},
            architecture_binding=analysis.get("architecture_binding", {}),
            fine_grained_emotion=analysis.get("fine_grained_emotion", {}),
            dual_constraints=analysis.get("dual_constraints", {}),
            hallucination_labels=analysis.get("hallucination_labels", {})
        )
        annotated_entries.append(entry)

    return annotated_entries

def main():
    print("Starting Expert Annotation Process (DeepSeek-V3 Powered + Multi-threaded)...")
    print(f"Reading from: {PROCESSED_DATA_DIR}")
    print(f"Writing to:   {OUTPUT_FILE}")
    print(f"Checkpoint:   {CHECKPOINT_FILE}")
    print(f"Max Workers:  {MAX_WORKERS}")


    processed_files = set()
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            processed_files = set(line.strip() for line in f if line.strip())
    print(f"Found {len(processed_files)} previously processed files in checkpoint.")


    all_files = list(PROCESSED_DATA_DIR.glob(CLEANED_FILES_PATTERN))
    files_to_process = [f for f in all_files if f.name not in processed_files]

    print(f"Total available files: {len(all_files)}")
    print(f"Remaining to process:  {len(files_to_process)}")

    if not files_to_process:
        print("All files have been processed! Exiting.")
        return

    annotator = DeepSeekAnnotator(DEEPSEEK_API_KEY)
    total_new_entries = 0


    try:



        with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f, \
             open(CHECKPOINT_FILE, "a", encoding="utf-8") as ckpt_f, \
             ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:


            future_to_file = {executor.submit(process_file, f, annotator): f for f in files_to_process}

            progress_bar = tqdm(as_completed(future_to_file), total=len(files_to_process), desc="Annotating")

            for future in progress_bar:
                file_path = future_to_file[future]
                try:
                    entries = future.result()

                    if not entries:

                        pass


                    for entry in entries:
                        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        total_new_entries += 1
                    out_f.flush()


                    ckpt_f.write(f"{file_path.name}\n")
                    ckpt_f.flush()

                except Exception as e:
                    progress_bar.write(f"CRITICAL ERROR processing {file_path.name}: {e}")

                    continue

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Progress saved up to last completed file.")
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception as e:
        print(f"\nFatal error: {e}")

    print(f"Session complete. Added {total_new_entries} new annotated segments.")
    print(f"Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
