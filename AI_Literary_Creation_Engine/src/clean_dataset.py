import os
import re
import sys
from tqdm import tqdm


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


SOURCE_DIR = "/root/paper/dataset"
DEST_DIR = "/root/paper/AI_Literary_Creation_Engine/processed_data/clean_dataset"
MIN_FILE_SIZE = 1024


os.makedirs(DEST_DIR, exist_ok=True)

try:
    from langdetect import detect
except ImportError:
    print("langdetect not installed. Please run: pip install langdetect")
    sys.exit(1)


NON_ENGLISH_MARKERS = [
    "(French)", "(Chinese)", "(German)", "(Italian)", "(Spanish)", "(Dutch)",
    "(Finnish)", "(Latin)", "(Portuguese)", "(Swedish)", "(Greek)", "(Russian)",
    "(Esperanto)", "(Welsh)", "(Afrikaans)", "(Hungarian)", "(Polish)",
    "(Tagalog)", "(Catalan)", "(Danish)", "(Norwegian)", "(Japanese)",
    "(Western Frisian)", "(Galician)", "(Icelandic)", "(Aleut)", "(Arabic)",
    "(Hebrew)", "(Slovenian)", "(Tibetan)", "(Old English)", "(Provençal)",
    "(Frisian)", "(Cebuano)", "(Iloko)", "(Navajo)"
]


ENGLISH_COMMON_WORDS = {"the", "and", "of", "to", "in", "a", "is", "that", "for", "it", "with", "as", "his", "he", "be"}

def is_english(text, filename):
    """
    Check if text is English based on filename and content.
    """

    for marker in NON_ENGLISH_MARKERS:
        if marker.lower() in filename.lower():
            return False



    try:
        if detect(text[:5000]) != 'en':
            return False
    except:
        pass

    sample = text[:5000].lower()
    words = re.findall(r'\b[a-z]{2,}\b', sample)

    if not words:
        return False

    english_word_count = sum(1 for w in words if w in ENGLISH_COMMON_WORDS)
    ratio = english_word_count / len(words)


    if ratio < 0.15:
        return False

    return True

def remove_gutenberg_headers(text):
    """
    Remove Project Gutenberg headers and footers.
    """


    start_markers = [
        r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\* ?START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\* ?START OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*",
    ]

    start_pos = 0
    for marker in start_markers:
        matches = list(re.finditer(marker, text, re.IGNORECASE | re.DOTALL))
        if matches:
            start_pos = max(start_pos, matches[-1].end())


    end_markers = [
        r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\* ?END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\* ?END OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*"
    ]

    end_pos = len(text)
    for marker in end_markers:
        match = re.search(marker, text, re.IGNORECASE | re.DOTALL)
        if match:
            end_pos = min(end_pos, match.start())


    if start_pos >= end_pos:
        return text

    body = text[start_pos:end_pos].strip()



    lines = body.split('\n')
    cleaned_lines = []
    skip_next = False

    for line in lines:
        l_lower = line.lower()
        if "project gutenberg" in l_lower or "gutenberg license" in l_lower:
            continue
        if "distributed proofreading" in l_lower:
            continue
        if re.search(r'\[footnote \d+:.*?\]', l_lower):
            continue
        if re.search(r'\[illustration:.*?\]', l_lower):
            continue

        cleaned_lines.append(line)

    body = '\n'.join(cleaned_lines)

    return body

def clean_toc_and_front_matter(text):
    """
    Heuristic to remove Table of Contents and Front Matter.
    Looks for "Contents" header and attempts to find where the body starts.
    """
    lines = text.split('\n')


    contents_idx = -1
    for i in range(min(len(lines), 300)):
        line = lines[i].strip()

        if re.search(r'^(Table of )?Contents$', line, re.IGNORECASE):
            contents_idx = i
            break

    if contents_idx != -1:


        start_of_toc_items = contents_idx + 1
        first_toc_item = None


        for i in range(start_of_toc_items, min(len(lines), start_of_toc_items + 50)):
            item = lines[i].strip()
            if item and len(item) > 4:
                first_toc_item = item
                break

        if first_toc_item:


            search_start_idx = contents_idx + 50

            for j in range(search_start_idx, min(len(lines), search_start_idx + 2000)):
                if lines[j].strip() == first_toc_item:


                    return "\n".join(lines[j:])

    return text

def clean_text_formatting(text):
    """
    General text cleaning: normalize whitespace.
    """

    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Source directory not found: {SOURCE_DIR}")
        return

    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.txt')]
    print(f"Found {len(files)} files in {SOURCE_DIR}")

    processed_count = 0
    skipped_lang_count = 0
    error_count = 0


    for filename in tqdm(files, desc="Processing files", unit="file"):
        file_path = os.path.join(SOURCE_DIR, filename)

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if len(content) < MIN_FILE_SIZE:
                continue


            if not is_english(content, filename):
                skipped_lang_count += 1
                continue


            text = remove_gutenberg_headers(content)


            text = clean_toc_and_front_matter(text)


            text = clean_text_formatting(text)

            if not text:
                continue


            output_filename = f"cleaned_{filename}"
            output_path = os.path.join(DEST_DIR, output_filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)

            processed_count += 1

        except Exception as e:
            error_count += 1


    print(f"\nProcessing complete.")
    print(f"Total files scanned: {len(files)}")
    print(f"Processed English files: {processed_count}")
    print(f"Skipped non-English files: {skipped_lang_count}")
    print(f"Errors: {error_count}")

if __name__ == "__main__":
    main()
