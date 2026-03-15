import json
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import ast

def get_basic_emotion(text, classifier):
    try:
        result = classifier(text[:512])[0]
        return result['label']
    except Exception as e:
        print(f"Error mapping '{text}': {e}")
        return "joy"

def main():
    try:
        with open('/root/unique_emotions.txt', 'r') as f:
            content = f.read().strip()
            try:
                unique_emotions = ast.literal_eval(content)
            except:
                unique_emotions = set(content.split('\n'))
    except FileNotFoundError:
        print("unique_emotions.txt not found.")
        return

    print(f"Found {len(unique_emotions)} unique emotions.")

    print("Loading BERT model...")
    device = 0 if torch.cuda.is_available() else -1
    model_name = "bhadresh-savani/bert-base-uncased-emotion"
    classifier = pipeline("text-classification", model=model_name, device=device)

    mapping = {}
    for emo in unique_emotions:
        if not emo or not isinstance(emo, str): continue
        basic = get_basic_emotion(emo, classifier)
        mapping[emo.lower()] = basic
        mapping[emo] = basic

    mapping['admiration'] = 'love'
    mapping['trust'] = 'love'
    mapping['neutral'] = 'joy'

    output_path = '/root/paper/AI_Literary_Creation_Engine/src/emotion_mapping.json'
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=4)

    print(f"Saved mapping to {output_path}")

if __name__ == "__main__":
    main()
