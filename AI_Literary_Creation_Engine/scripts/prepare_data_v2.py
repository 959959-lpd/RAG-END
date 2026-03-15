import sys
import os
import json
import base64


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.data_processing import DatasetAnnotator
except ImportError:

    sys.path.append("/root/paper/AI_Literary_Creation_Engine")
    from src.data_processing import DatasetAnnotator

def main():
    input_path = "/root/paper/dataset"
    output_path = "/root/paper/AI_Literary_Creation_Engine/processed_data/output.jsonl"

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading DatasetAnnotator...")
    try:
        annotator = DatasetAnnotator()
        print(f"Annotator loaded. Processing {input_path}...")
        annotator.process_file(input_path, output_path)
        print(f"Data processing complete. Saved to {output_path}")
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()