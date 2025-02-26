import json

bert_path = "../data/bert_results_roberta_large.json"
rouge_path = "../data/rouge_output.json"
bleu_path = "../data/evaluated_results.json"

with open(bert_path, 'r') as f:
    bert_data = json.load(f)

with open(rouge_path, 'r') as f:
    rouge_data = json.load(f)

with open(bleu_path, 'r') as f:
    bleu_data = json.load(f)

rouge_data = rouge_data["individual_scores"]
results = []
for bert, rouge, bleu in zip(bert_data, rouge_data, bleu_data):
    results.append({
        "question": bert["question"],
        "reference_answer": bert["reference_answer"],
        "model_answer": bert["model_answer"],
        "BERT": bert["BERT"],
        "ROUGE": rouge['rouge_scores'],
        "BLEU": bleu['BLEU']
    })

with open("../data/eval_results_roberta.json", 'w') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)