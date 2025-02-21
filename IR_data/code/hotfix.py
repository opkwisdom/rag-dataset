import json
import jsonlines
from typing import List, Dict, Any
from tqdm import tqdm

JsonType = Dict[str, Any]

def read_jsonl(filepath: str) -> List[JsonType]:
    data = []
    print(f'Reading {filepath}...')
    with jsonlines.open(filepath, 'r') as jsonl_reader:
        for json_item in jsonl_reader:
            data.append(json_item)
    return data

source_file = "../../data/qas.jsonl"
in_file = "../data/query_updated.jsonl"
out_file = "../data/query_updated2.jsonl"

source_data = read_jsonl(source_file)
target_data = read_jsonl(in_file)

source_data = sorted(source_data, key=lambda x: x['id'])
target_data = sorted(target_data, key=lambda x: x['query_id'])

i, j = 0, 0
total_data = []
pbar = tqdm(target_data, desc="Processing ")
while i < len(source_data) and j < len(target_data):
    source = source_data[i]
    target = target_data[j]
    
    if source['id'] == target['query_id']:
        data = {
            'query_id': target['query_id'],
            'question': target['question'],
            'answer': source['answer'],
            'query': target['query'],
            'gold_id': target['gold_id']
        }
        total_data.append(data)
        i += 1; j += 1
        pbar.update(1)
    else:
        i += 1

with open(out_file, 'w', encoding='utf-8') as outfile:
    for item in total_data:
        outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'{len(total_data)} saved')