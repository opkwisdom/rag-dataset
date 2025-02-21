import json
from argparse import ArgumentParser
import os
from typing import List, Dict, Any
from tqdm import tqdm
from collections import defaultdict
import math

inf = float('inf')
JsonType = Dict[str, Any]

def save_stats_to_file(tp, n_total, **kwargs):
    filepath = f"../stat/{tp}.txt"
    stats = kwargs
    with open(filepath, 'w') as f:
        f.write(f"******** {tp} Dataset Statistics ********\n")
        f.write(f"Total Length: {n_total}\n\n")
        
        for field, stat in stats.items():
            f.write(f"Filed: {field}\n")
            for key, value in stat.items():
                f.write(f"   {key}: {value:.2f}\n")
            f.write("\n")
        f.write(f"{'*' * 40}\n\n")

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

def compute_stats(data):
    # 필드별 통계 저장소 초기화
    stats = defaultdict(lambda: {
        "min_len": math.inf,
        "max_len": 0,
        "avg_len": 0
    })
    
    n_total = len(data)

    # 데이터 순회
    for d in tqdm(data):
        for field, value in d.items():
            # 문자열, 리스트, 또는 딕셔너리인지 확인하여 처리
            if isinstance(value, (str, list, dict)):
                length = len(value)
                stats[field]["avg_len"] += length
                stats[field]["min_len"] = min(stats[field]["min_len"], length)
                stats[field]["max_len"] = max(stats[field]["max_len"], length)

    # 평균 계산
    for field, stat in stats.items():
        if "avg_len" in stat:
            stat["avg_len"] /= n_total
    
    return stats, n_total

def make_stat(filepath, tp):
    if os.path.exists(f"../stat/{tp}.txt"):
        print(f"Path for {tp} already exist")
        return -1
    
    data = read_file(filepath)
    
    stats, n_total = compute_stats(data)
    
    save_stats_to_file(tp, n_total, **stats)

def main():
    parser = ArgumentParser()
    parser.add_argument('--file_path', required=True, nargs='+', help='List of file paths')
    parser.add_argument('--file_type', required=True, nargs='+', help='List of file types corrresponding to each file')
    args = parser.parse_args()
    
    print("File paths:", args.file_path)
    print("File types:", args.file_type)
    
    assert len(args.file_path) == len(args.file_type),\
        f'{len(args.file_path)}, {len(args.file_type)}'
        
    for file_path, file_type in zip(args.file_path, args.file_type):
        make_stat(file_path, file_type)
    
if __name__ == "__main__":
    main()