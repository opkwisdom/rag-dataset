import json
from tqdm import tqdm

input_file = "../data/qas.jsonl"
output_file = "../data/qas_hard.jsonl"

def save_to_jsonl(total_data, outfile_path):
    with open(outfile_path, 'w', encoding='utf-8') as outfile:
        for entry in total_data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')  # 줄바꿈 추가
    print(f"{len(total_data)} data successfully saved to {outfile_path}")

with open(input_file, 'r') as f:
    data = f.readlines()

filtered = []
for d in tqdm(data, desc='Processing '):
    d = json.loads(d)
    if d['level'] == 2:
        filtered.append(d)

save_to_jsonl(filtered, output_file)