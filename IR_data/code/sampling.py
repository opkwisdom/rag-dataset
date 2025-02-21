import json
from tqdm import tqdm
import random
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("--infile_path", default="./data/ir_data_v1.jsonl")
    parser.add_argument("--outfile_path", default="./data/ir_data_v1_sampled.jsonl")
    parser.add_argument("--n_samples", type=int, default=50000)
    args = parser.parse_args()
    
    with open(args.infile_path, 'r', encoding='utf-8') as file:
        total_lines = sum(1 for _ in file)
        file.seek(0)  # 파일 포인터를 다시 시작 부분으로 이동
        passages = [json.loads(line) for line in tqdm(file, total=total_lines, desc="Loading Passages")]

    random.shuffle(passages)

    passages = passages[:args.n_samples]
    with open(args.outfile_path, encoding="utf-8", mode="w") as writer:
        for i in tqdm(passages, desc="Save Result "):
            writer.write(json.dumps(i, ensure_ascii=False) + '\n')

    print(f"Data successfully saved to {args.outfile_path}: length={len(passages)}")
    
if __name__ == "__main__":
    main()