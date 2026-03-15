import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import DatasetAnnotator

def main():
    if not os.path.exists("dataset/input.txt"):
        print("Error: dataset/input.txt not found.")
        return

    os.makedirs("processed_data", exist_ok=True)

    annotator = DatasetAnnotator()
    annotator.process_file("dataset/input.txt", "processed_data/output.jsonl")
    print("Data processing complete. Saved to processed_data/output.jsonl")

if __name__ == "__main__":
    main()