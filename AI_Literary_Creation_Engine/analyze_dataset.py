import json
import matplotlib.pyplot as plt
from collections import Counter
import re
from tqdm import tqdm
import os

INPUT_FILE = "/root/paper/AI_Literary_Creation_Engine/processed_data/output_v2.jsonl"
OUTPUT_DIR = "/root/paper/AI_Literary_Creation_Engine/analysis_results_v2"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def analyze_dataset():
    sentiment_counts = Counter()
    imagery_counts = Counter()
    intensity_counts = Counter()

    total_lines = 0
    truncated_candidates = []
    gutenberg_boilerplate_count = 0


    gutenberg_pattern = re.compile(r"(Project Gutenberg|eBook|License|Release date)", re.IGNORECASE)


    valid_endings = ('.', '!', '?', '"', "'", ')', ']', '}', ';', ':', '—', '-')

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for i, line in tqdm(enumerate(f), desc="Analyzing lines"):
            total_lines += 1
            try:
                data = json.loads(line)
                text = data.get('text', '').strip()
                metadata = data.get('metadata', {})


                if gutenberg_pattern.search(text):
                    gutenberg_boilerplate_count += 1



                if text and not text.endswith(valid_endings):



                    if len(truncated_candidates) < 20:
                        truncated_candidates.append(text)


                sentiment = metadata.get('sentiment')
                if sentiment:
                    sentiment_counts[sentiment] += 1

                imagery = metadata.get('imagery', [])
                for img in imagery:
                    imagery_counts[img] += 1

                intensity = metadata.get('intensity')
                if intensity is not None:
                    intensity_counts[str(intensity)] += 1

            except json.JSONDecodeError:
                print(f"JSON Error on line {i+1}")
                continue


    print("Generating plots...")


    plt.figure(figsize=(10, 6))
    sentiments, counts = zip(*sentiment_counts.most_common())
    plt.bar(sentiments, counts, color='skyblue')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sentiment_distribution.png")
    plt.close()


    plt.figure(figsize=(12, 6))
    imageries, img_counts = zip(*imagery_counts.most_common(20))
    plt.bar(imageries, img_counts, color='lightgreen')
    plt.title('Top 20 Imagery Words')
    plt.xlabel('Imagery')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/top_imagery_distribution.png")
    plt.close()


    plt.figure(figsize=(8, 5))
    intensities, int_counts = zip(*sorted(intensity_counts.items(), key=lambda x: x[0]))
    plt.bar(intensities, int_counts, color='salmon')
    plt.title('Emotion Intensity Distribution')
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/intensity_distribution.png")
    plt.close()


    print(f"\nAnalysis Complete.")
    print(f"Total entries: {total_lines}")
    print(f"Entries containing 'Gutenberg' boilerplate (approx): {gutenberg_boilerplate_count}")
    print(f"Potential truncated/unpunctuated lines (sample 5):")
    for t in truncated_candidates[:5]:
        print(f" - {t}")

    print(f"\nTop Sentiments: {sentiment_counts.most_common(5)}")
    print(f"Top Imagery: {imagery_counts.most_common(5)}")
    print(f"Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    analyze_dataset()
