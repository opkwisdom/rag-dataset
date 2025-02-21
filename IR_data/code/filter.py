import json
from tqdm import tqdm
import os
from typing import List, Dict, Any
from argparse import ArgumentParser

JsonType = Dict[str, Any]

def read_file(filepath: str) -> List[JsonType]:
    _, ext = os.path.splitext(filepath)
    
    print(f'Reading from {filepath}...')
    if ext == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
    elif ext == '.jsonl':
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return data

max_keywords = 3
min_question_len = 10
max_question_len = 200

def main():
    parser = ArgumentParser()
    parser.add_argument('--in_path', required=True, help='infile path')
    parser.add_argument('--out_path', required=True, help='outfile path')
    parser.add_argument('--min_relevant_passages', required=True, help='min # of relevant passages', type=int, default=3)
    parser.add_argument('--max_relevant_passages', required=True, help='max # of relevant passages', type=int, default=200)
    args = parser.parse_args()
    
    data = read_file(args.in_path)
    filtered = []
    for d in tqdm(data, desc="Processing "):
        if len(d['positive_id']) >= args.min_relevant_passages and len(d['positive_id']) <= args.max_relevant_passages and\
            len(d['query']) <= max_keywords and\
            len(d['question']) >= min_question_len and len(d['question']) <= max_question_len:
            filtered.append(d)

    with open(args.out_path, 'w') as jsonl_file:
        for item in filtered:
            jsonl_file.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"{len(filtered)} data has saved.")
    
if __name__ == "__main__":
    main()