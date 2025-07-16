import json
import numpy as np
from collections import Counter, defaultdict
from argparse import ArgumentParser

from main_metrics import save_results

def base_stat(data):
    context_lengths = [len(item["context"]) for item in data]
    answer_lengths = [len(item["answer"]) for item in data]

    def length_summary(lengths):
        return {
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
            "mean": round(float(np.mean(lengths)), 2),
            "std": round(float(np.std(lengths)), 2),
        }

    return {
        "total_count": len(data),
        "context_length": length_summary(context_lengths),
        "answer_length": length_summary(answer_lengths),
    }


def keyphrase_stat(data):
    ngram_counter = Counter()

    for item in data:
        keyphrases = item.get("keyphrases", [])
        for phrase in keyphrases:
            n = len(phrase.split())
            bucket = str(n) if n < 5 else "5"
            ngram_counter[bucket] += 1

    total = sum(ngram_counter.values())
    stat = {
        bucket: round(100 * count / total, 2)
        for bucket, count in sorted(ngram_counter.items(), key=lambda x: int(x[0]))
    }

    return {"keyphrase_ngram_stat": stat}


if __name__ == "__main__":
    parser = ArgumentParser(description="Get data statistics")
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the input JSON file')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to save the output statistics JSON file')
    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stat = base_stat(data)
    stat.update(keyphrase_stat(data))

    save_results(args.output_file, stat)
    print(f"Saved stats to {args.output_file}")