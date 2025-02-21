import json
from tqdm import tqdm

with open('evaluated_results.json', 'r') as f:
    results = json.load(f)

high_bert_r = 0
high_bert_p = 0
high_bert_f = 0
high_bleu = 0

for result in tqdm(results):
    bert_r = result['BERT']['R']
    bert_p = result['BERT']['P']
    bert_f = result['BERT']['F']
    bleu = result['BLEU']
    
    high_bert_r = max(high_bert_r, bert_r)
    high_bert_p = max(high_bert_p, bert_p)
    high_bert_f = max(high_bert_f, bert_f)
    high_bleu = max(high_bleu, bleu)

print(f"Highest BERT Recall: {high_bert_r}")
print(f"Highest BERT Precision: {high_bert_p}")
print(f"Highest BERT F1: {high_bert_f}")
print(f"Highest BLEU: {high_bleu}")