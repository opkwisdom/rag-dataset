import os
import json
import numpy as np

with open("../data/eval_results_roberta.json", 'r') as f:
    eval_data = json.load(f)

bert_results = []
rouge1_results = []
rouge2_results = []
rougeL_results = []
bleu_results = []

for data in eval_data:
    bert_results.append(data['BERT']['F'])
    rouge1_results.append(data['ROUGE']['rouge1'])
    rouge2_results.append(data['ROUGE']['rouge2'])
    rougeL_results.append(data['ROUGE']['rougeL'])
    bleu_results.append(data['BLEU'])

bert_arr = np.array(bert_results)
rouge1_arr = np.array(rouge1_results)
rouge2_arr = np.array(rouge2_results)
rougeL_arr = np.array(rougeL_results)
bleu_arr = np.array(bleu_results)

metrics_dict = {
    "BERT": bert_arr,
    "ROUGE-1": rouge1_arr,
    "ROUGE-2": rouge2_arr,
    "ROUGE-L": rougeL_arr,
    "BLEU": bleu_arr,
}

results = {
        "BERT": {
            "Avg": float(np.mean(bert_arr)),
            "25%": float(np.percentile(bert_arr, 25)),
            "Med": float(np.median(bert_arr)),
            "75%": float(np.percentile(bert_arr, 75)),
            "Max": float(np.max(bert_arr))
        },
        "ROUGE-1": {
            "Avg": float(np.mean(rouge1_arr)),
            "25%": float(np.percentile(rouge1_arr, 25)),
            "Med": float(np.median(rouge1_arr)),
            "75%": float(np.percentile(rouge1_arr, 75)),
            "Max": float(np.max(rouge1_arr))
        },
        "ROUGE-2": {
            "Avg": float(np.mean(rouge2_arr)),
            "25%": float(np.percentile(rouge2_arr, 25)),
            "Med": float(np.median(rouge2_arr)),
            "75%": float(np.percentile(rouge2_arr, 75)),
            "Max": float(np.max(rouge2_arr))
        },
        "ROUGE-L": {
            "Avg": float(np.mean(rougeL_arr)),
            "25%": float(np.percentile(rougeL_arr, 25)),
            "Med": float(np.median(rougeL_arr)),
            "75%": float(np.percentile(rougeL_arr, 75)),
            "Max": float(np.max(rougeL_arr))
        },
        "BLEU": {
            "Avg": float(np.mean(bleu_arr)),
            "25%": float(np.percentile(bleu_arr, 25)),
            "Med": float(np.median(bleu_arr)),
            "75%": float(np.percentile(bleu_arr, 75)),
            "Max": float(np.max(bleu_arr))
        },
    }

with open("../eval/roberta/eval_summary.json", 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

# Max results
bert_max_idx = int(np.argmax(bert_arr))
rouge1_max_idx = int(np.argmax(rouge1_arr))
rouge2_max_idx = int(np.argmax(rouge2_arr))
rougeL_max_idx = int(np.argmax(rougeL_arr))
bleu_max_idx = int(np.argmax(bleu_arr))

with open("../eval/roberta/best_performing_qa.json", 'w') as f:
    json.dump(
        {
            "BERT": eval_data[bert_max_idx],
            "ROUGE-1": eval_data[rouge1_max_idx],
            "ROUGE-2": eval_data[rouge2_max_idx],
            "ROUGE-L": eval_data[rougeL_max_idx],
            "BLEU": eval_data[bleu_max_idx],
        },
        f,
        ensure_ascii=False,
        indent=4
    )

def save_results(eval_data, metric_dict, percentile=[0, 5, 10, 20, 40, 60, 80, 100], n=10):
    for i in range(len(percentile) - 1):
        lower_pct = percentile[i] / 100.0
        upper_pct = percentile[i + 1] / 100.0

        quantile_results = {}

        # import pdb; pdb.set_trace()
        for metric_name, metric_arr in metric_dict.items():
            sorted_idx = np.argsort(metric_arr)[::-1]
            start_idx = int(len(metric_arr) * lower_pct)
            end_idx = int(len(metric_arr) * upper_pct)
            segment = sorted_idx[start_idx:end_idx]
            
            if len(segment) >= n:
                selected_idx = segment[:n]
            else:
                selected_idx = segment

            quantile_results[metric_name] = [eval_data[i] for i in selected_idx]
        
        file_name = f"../eval/roberta/upper{percentile[i]}.json"
        with open(file_name, 'w') as f:
            json.dump(quantile_results, f, ensure_ascii=False, indent=4)
        print(f"Saved {file_name}")

def retrieve_top_k_results(eval_data, metric_dict):
    percentile = [i for i in range(0, 101, 5)]

    if os.path.exists(f'../eval/roberta/n_topk.txt'):
        os.remove(f'../eval/roberta/n_topk.txt')

    for i in range(len(percentile)-1):
        lower_pct = percentile[i] / 100.0
        upper_pct = percentile[i + 1] / 100.0

        quantile_results = {}
        quantile_idx = []


        for metric_name, metric_arr in metric_dict.items():
            sorted_idx = np.argsort(metric_arr)[::-1]
            start_idx = int(len(metric_arr) * lower_pct)
            end_idx = int(len(metric_arr) * upper_pct)
            segment = set(sorted_idx[start_idx:end_idx])
            quantile_idx.append(segment)
        
        selected_idx = list(set.intersection(*quantile_idx))
        with open(f'../eval/roberta/n_topk.txt', 'a') as f:
            f.write(f"Top {percentile[i]}%: {len(selected_idx)}\n")
    

# save_results(eval_data, metrics_dict)
retrieve_top_k_results(eval_data, metrics_dict)