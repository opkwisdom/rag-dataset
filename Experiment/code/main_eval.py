from argparse import ArgumentParser
from main_metrics import *

def main():
    parser = ArgumentParser(description="Evaluate metrics for KeyRAG")
    parser.add_argument('--model_name_or_path', type=str, required=True,
                        help='Path to the model or model name')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing the input JSON files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the output results')
    parser.add_argument('--task', type=str, required=True,
                        help='Task type (context_only, keyphrase_only, keysentence, query_only, context_keyphrase)')
    parser.add_argument('--infile_name', type=str, required=True,
                        help='Name of the input file to process')
    parser.add_argument('--outfile_name', type=str, required=True,
                        help='Name of the output file to save results')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing')
    parser.add_argument('--slice', type=int, default=None,
                        help='Optional slice of the dataset to process')
    args = parser.parse_args()
    print(f"Args: {args}")
    
    input_path = f"{args.input_dir}/{args.infile_name}.json"
    output_path = f"{args.output_dir}/{args.outfile_name}.json"
    
    metrics = [
        compute_bleu,
        compute_rouge,
        compute_bertscore,
        compute_conditional_ppl
    ]
    
    results: list[MetricResult] = [
        metric(
            filepath=input_path,
            model_id=args.model_name_or_path,
            batch_size=args.batch_size,
            slice=args.slice,
            task=args.task
        )
        for metric in metrics
    ]
    
    # Aggregate results
    json_results = {
        "task": args.task,
        "evaluator": args.model_name_or_path,
        "metrics": {
            "average_scores": {},
            "total_scores": []
        }
    }
    
    # Mean score
    for result in results:
        for key, value in result.mean_score.items():
            json_results["metrics"]["average_scores"][key] = value
    
    # Each score
    total_scores = []
    for result in results:
        for key, value in result.each_score.items():
            total_scores.append({key: value})
    
    metric_dict = {}
    for d in total_scores:
        metric_dict.update(d)

    final_scores = [
        {k: metric_dict[k][i] for k in metric_dict}
        for i in range(len(next(iter(metric_dict.values()))))
    ]
    json_results["metrics"]["total_scores"] = final_scores
    
    save_results(output_path, json_results)
    
if __name__ == "__main__":
    main()