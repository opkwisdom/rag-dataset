import json
from tqdm import tqdm



def answer_state(filepath):
    len_collection = []
    with open(filepath, 'r') as f:
        data = [json.loads(line.strip()) for line in f]

    for line in tqdm(data, desc='QAS '):
        len_collection.append(len(line['answer']))

    # 통계
    max_, min_, std_, mean_ = 0, 0, 0, 0
    max_ = max(len_collection)
    min_ = min(len_collection)
    mean_ = sum(len_collection) / len(len_collection)

    diff = [(num - mean_)**2 for num in len_collection]
    std_ = sum(diff) / len(len_collection)

    print(f"Max: {max_}")
    print(f"Min: {min_}")
    print(f"Mean: {mean_:.2f}")
    print(f"Std: {std_:.2f}")

    print(data[len_collection.index(max_)])
    print(data[len_collection.index(min_)])

def clip_answer(filepath, outfile, min_length=100, max_length=1000):
    total_data = []
    with open(filepath, 'r') as f:
        data = [json.loads(line.strip()) for line in f]

    for line in tqdm(data, desc='QAS '):
        length = len(line['answer'])
        if length >= min_length and length <= max_length:
            total_data.append(line)

    with open(outfile, 'w') as f:
        for line in total_data:
            json.dump(line, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"{len(total_data)} has extracted")

if __name__ == "__main__":
    filepath = '../collection/qas_hard.jsonl'
    outfile = '../collection/qas_hard_clipped.jsonl'
    clip_answer(filepath, outfile)