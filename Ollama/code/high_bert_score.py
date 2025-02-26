import json

# 평가 결과와 원본 데이터 파일 읽기
with open('eval_results.json', 'r') as f:
    eval_results = json.load(f)

with open('../../IR_data/collection/qas_hard_ollama.jsonl', 'r') as f:
    qa_pairs = [json.loads(line.strip()) for line in f]

# BERT F-score가 가장 높은 인덱스 찾기
max_f_score = float('-inf')
max_index = 0

for i, result in enumerate(eval_results):
    if result['BERT']['F'] > max_f_score:
        max_f_score = result['BERT']['F']
        max_index = i

# 가장 높은 점수를 받은 QA 쌍과 평가 결과 출력
best_qa = qa_pairs[max_index]
best_eval = eval_results[max_index]

print("Best performing QA pair:")
print("\nQuestion:", best_qa['question'])
print("\nReference Answer:", best_qa['answer'])
print("\nModel Answer:", best_qa['llm_answer'])
print("\nEvaluation Scores:")
print(json.dumps(best_eval, indent=2, ensure_ascii=False))