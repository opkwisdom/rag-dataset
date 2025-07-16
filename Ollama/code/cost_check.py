import json
import time
from tqdm import tqdm

rag_path = 'rag/gpt-4o-mini/qas_hard_rag_1000_true.json'
input_path = "../../IR_data/collection/qas_hard_for_rag.jsonl"

with open(rag_path, 'r') as f:
    rag = json.load(f)

with open(input_path, 'r') as f:
    data = [json.loads(d) for d in f]

def format_seconds_with_days(seconds):
    days = seconds // (24 * 3600)
    remainder = seconds % (24 * 3600)
    hours = remainder // 3600
    remainder %= 3600
    minutes = remainder // 60
    seconds = remainder % 60

    if days > 0:
        return f"{days}d {hours:02}:{minutes:02}:{seconds:02}"
    else:
        return f"{hours:02}:{minutes:02}:{seconds:02}"


def check_price(data, rag):
    rag_price = rag['metadata']['Total Cost($)']
    rag_length = rag['metadata']['Total Length']
    rag_count = rag['metadata']['Total Count']
    h, m, s = map(int, rag['metadata']['Total Time'].split(":"))
    rag_seconds = h * 3600 + m * 60 + s

    expected_prompt_length = 0
    for d in tqdm(data):
        expected_prompt_length += (len(d['question']) + len(d['context']) + len(d['answer']))
    expected_price = expected_prompt_length * rag_price / rag_length
    expected_seconds = int(expected_prompt_length * rag_seconds / rag_length)

    print(f"Current {rag_count} prompt length: {rag_length}")
    print(f"Expected {len(data)} prompt length: {expected_prompt_length}")
    print(f"Expected price: {expected_price:.2f}$")
    print(f"Expected time: {format_seconds_with_days(expected_seconds)}")

check_price(data, rag)