import re

file_path = "/root/paper/dataset/397_L'Allegro, Il Penseroso, Comus, and Lycidas.txt"

with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()


print(f"Content repr: {repr(content[:500])}")

paragraphs = re.split(r'\n\s*\n', content)
print(f"Num paragraphs: {len(paragraphs)}")

for i, p in enumerate(paragraphs[:5]):
    print(f"Paragraph {i} length: {len(p)}")
    print(f"Paragraph {i} text: {repr(p)}")
    cleaned = p.replace('\n', ' ')
    print(f"Cleaned {i}: {cleaned}")
