import json
from tqdm import tqdm
from collections import defaultdict
from prompt_ranker import idf_computation

path1 = "../data/qas_rag_candidate.jsonl"
path2 = 'output/qas_rag_idf_candidate.json'
output_path = 'output/qas_rag_idf_candidate_m.json'

passages = []
all_candidates = set()
candidate_dict = defaultdict(set)

print("Reading files...")
with open(path1, "r") as f:
    data = [json.loads(line) for line in f]
    passages = [item['context'] for item in data]

with open(path2, "r") as f:
    candidate_data = json.load(f)


for item in candidate_data:
    pid = item['id']
    for kp in item['keyphrases']:
        candidate_dict[pid].add(kp)
        all_candidates.add(kp)

all_candidates = list(all_candidates)


print("Compute IDF...")
idf_dict = idf_computation(passages, all_candidates)
print("IDF Done")


result_data = []
for line in tqdm(data):
    pid = line['id']

    each_candidates = candidate_dict.get(pid, [])
    # IDF 기준 내림차순 정렬 후 top-10
    sorted_candidates = sorted(
        each_candidates,
        key=lambda x: idf_dict.get(x, 0),  # 없는 경우 idf=0으로 처리
        reverse=True
    )
    top10_candidates = sorted_candidates[:10]

    line['keyphrases'] = top10_candidates
    line['gold_passage_id'] = line['gold_id']
    line['answer'] = line['gpt-4o-mini_answer']
    line.pop('level')
    line.pop('gold_id')
    line.pop('keyword')
    line.pop('candidates')
    line.pop('gpt-4o-mini_answer')
    result_data.append(line)

with open(output_path, "w") as f:
    json.dump(result_data, f, ensure_ascii=False, indent=4)
    