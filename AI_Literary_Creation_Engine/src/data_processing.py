











import spacy
from transformers import pipeline
from tqdm import tqdm
import json
import re
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import IMAGERY_KEYWORDS, MODEL_CONFIG

class DatasetAnnotator:
    def __init__(self):
        print("Loading annotation models...")

        try:

            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            print("Please run: python -m spacy download en_core_web_sm")
            sys.exit(1)


        self.imagery_stopwords = {
            "project", "gutenberg", "ebook", "chapter", "page", "volume",
            "edition", "editor", "author", "title", "content", "contents",
            "copyright", "license", "text", "file", "http", "www", "com",
            "org", "net", "html", "htm", "illustration", "ind", "ii", "iii",
             "iv", "vi", "vii", "viii", "ix", "xi", "xii", "xiii", "xiv",
             "xv", "xvi", "xvii", "xviii", "xix", "xx", "footnote", "note",
             "release", "date", "updated", "language", "english", "character",
             "set", "encoding", "ascii", "start", "end", "produced", "distribute",
             "proofreading", "team", "online"
        }



        self.sentiment_analyzer = pipeline(
            "text-classification",
            model=MODEL_CONFIG["sentiment_model"],
            tokenizer=MODEL_CONFIG["sentiment_model"],
            top_k=None,
            device=0,
            batch_size=32
        )

    def text_cleaning(self, text):
        """文本清洗"""

        text = re.sub(r'\n+', ' ', text)

        text = re.sub(r'\s+', ' ', text).strip()


        return text

    def get_split_pattern(self, content):
        """Determine paragraph splitting pattern based on file content statistics"""

        single_count = len(re.findall(r'(?<!\n)\n(?!\n)', content))

        double_count = len(re.findall(r'\n\s*\n', content))



        if single_count > double_count:
            return r'(?:\n\s*){2,}'
        else:

            return r'(?:\n\s*){3,}'

    def extract_imagery(self, doc):
        """提取核心意象 (名词/专有名词中过滤)"""

        candidates = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop]





        unique_imagery = list(set(candidates))
        unique_imagery = [w for w in unique_imagery if w not in self.imagery_stopwords]


        unique_imagery = [w for w in unique_imagery if len(w) > 2]

        return unique_imagery

    def process_file_list(self, files_list, output_path):
        """处理文件列表"""
        print(f"Processing {len(files_list)} files...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        count = 0
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for file_path in tqdm(files_list, desc="Files"):
                self.process_one_file(file_path, out_f, count)

    def process_one_file(self, file_path, out_f, count):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            pattern = self.get_split_pattern(content)
            try:
                paragraphs = re.split(pattern, content)
            except Exception as e:

                paragraphs = re.split(r'\n\s*\n', content)

            valid_paragraphs = []
            for para in paragraphs:
                clean_text = self.text_cleaning(para)
                if len(clean_text) >= 50:
                    valid_paragraphs.append(clean_text)

            if not valid_paragraphs: return


            docs = list(self.nlp.pipe(valid_paragraphs))


            sentiment_results = self.sentiment_analyzer(valid_paragraphs, truncation=True, max_length=512)

            for text, doc, res in zip(valid_paragraphs, docs, sentiment_results):

                imagery = self.extract_imagery(doc)

                top_sentiment = max(res, key=lambda x: x['score'])
                label = top_sentiment['label']
                score = top_sentiment['score']

                if score > 0.9: intensity = 5
                elif score > 0.7: intensity = 4
                elif score > 0.5: intensity = 3
                else: intensity = 2

                entry = {
                    "text": text,
                    "metadata": {
                        "source_file": os.path.basename(file_path),
                        "sentiment": label,
                        "intensity": intensity,
                        "confidence": score,
                        "imagery": imagery
                    },
                    "rag_key": f"{label}_{intensity}"
                }

                out_f.write(json.dumps(entry) + "\n")

        except Exception as e:

            pass

    def process_file_original(self, input_path, output_path):
        """处理整个文件或文件夹 (Original method)"""



        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        count = 0


        with open(output_path, 'w', encoding='utf-8') as out_f:


            for file_path in tqdm(files_to_process, desc="Files"):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()


                    pattern = self.get_split_pattern(content)


                    try:
                        paragraphs = re.split(pattern, content)
                    except Exception as e:
                        print(f"Regex error split {pattern}: {e}")
                        paragraphs = re.split(r'\n\s*\n', content)

                    valid_paragraphs = []
                    for para in paragraphs:
                        clean_text = self.text_cleaning(para)
                        if len(clean_text) >= 50:
                            valid_paragraphs.append(clean_text)

                    if not valid_paragraphs: continue


                    docs = list(self.nlp.pipe(valid_paragraphs))


                    sentiment_results = self.sentiment_analyzer(valid_paragraphs, truncation=True, max_length=512)

                    for text, doc, res in zip(valid_paragraphs, docs, sentiment_results):

                        imagery = self.extract_imagery(doc)



                        top_sentiment = max(res, key=lambda x: x['score'])
                        label = top_sentiment['label']
                        score = top_sentiment['score']

                        if score > 0.9: intensity = 5
                        elif score > 0.7: intensity = 4
                        elif score > 0.5: intensity = 3
                        elif score > 0.3: intensity = 2
                        else: intensity = 1


                        entry = {
                            "text": text,
                            "metadata": {
                                "source_file": os.path.basename(file_path),
                                "sentiment": label,
                                "intensity": intensity,
                                "confidence": float(score),
                                "imagery": imagery
                            },

                            "rag_key": f"{label}_{intensity}"
                        }

                        out_f.write(json.dumps(entry) + "\n")
                        count += 1

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

        print(f"Saved {count} processed entries to {output_path}")

if __name__ == "__main__":
    annotator = DatasetAnnotator()


    CLEAN_DATASET_DIR = "/root/paper/AI_Literary_Creation_Engine/processed_data/clean_dataset"
    OUTPUT_FILE = "/root/paper/AI_Literary_Creation_Engine/processed_data/output_v2.jsonl"

    if os.path.exists(CLEAN_DATASET_DIR):

        files = [os.path.join(CLEAN_DATASET_DIR, f) for f in os.listdir(CLEAN_DATASET_DIR) if f.endswith('.txt')]

        files = files


        print(f"Processing {len(files)} files...")
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        count = 0


        with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
            for file_path in tqdm(files, desc="Files"):
                try:
                    annotator.process_one_file(file_path, out_f, count)
                    count += 1
                except:
                    pass
    else:
        print(f"Clean dataset directory not found: {CLEAN_DATASET_DIR}")