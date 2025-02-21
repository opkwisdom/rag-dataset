import json
from tqdm import tqdm

max_len = 0
with open('passages.jsonl', 'r') as f:
    for d in tqdm(f):
        data = json.loads(d)
        max_len = max(max_len, len(data['contents']))

print(max_len)
