import json
from tqdm import tqdm
from typing import List, Dict, Any
import os
import jsonlines

inf = float('inf')
JsonType = Dict[str, Any]

def save_stats_to_file(tp, *args):
    filepath = f"../stat/{tp}.txt"
    with open(filepath, 'w') as f:
        for arg in args:
            f.write(f"{arg}\n")
            
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

def main():
    in_path = "../data/ir_data_SM_filter_2.jsonl"
    out_path = "../data/ir_data_SM_final_2.jsonl"

    data = read_file(in_path)
    # 비교 및 분류
    processed_data = []
    for d in tqdm(data):
        pos_id, neg_id = [], []
        for query_id in d["positive_id_by_query"]:
            if query_id in d["positive_id_by_answer"]:
                pos_id.append(query_id)
            neg_id.append(query_id)
        
        if not pos_id or not neg_id:
            continue
        question_pos_id = d['question_positive_id']
        d.pop('question_positive_id')
        
        d['positive_id'] = pos_id
        d['question_positive_id'] = question_pos_id
        d['query_positive_id'] = neg_id
        d.pop('positive_id_by_query')
        d.pop('positive_id_by_answer')
        
        processed_data.append(d)
        
    with jsonlines.open(out_path, mode="w") as jsonl_writer:
        jsonl_writer.write_all(processed_data)
        
    print(f'{len(processed_data)} saved')
    
if __name__ == "__main__":
    main()