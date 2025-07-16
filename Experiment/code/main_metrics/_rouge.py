from evaluate import load
from ._utils import *

def compute_rouge(
    filepath: str,
    model_id: str = None,
    batch_size: int = 16,
    slice: int = None,
    task: str = "context_only"
) -> MetricResult:
    '''
    Calculate the ROUGE score between predictions and references.
    
    Args:
        filepath: str, Path to the file containing predictions and references.
        model_id: str, Model ID for the evaluation (not used in this function).
        batch_size: int, Batch size for processing (not used in this function).
        slice: int, Optional slice of the dataset to process.
        task: str, Task type to determine how to process texts.
    Returns:
        MetricResult: Result containing ROUGE scores.
    '''
    metric_dataset: MetricDataset = prepare_dataset(filepath, slice)
    rouge = load('rouge')
    
    print('Calculating ROUGE...')
    each_score = rouge.compute(
        predictions=metric_dataset.predictions,
        references=metric_dataset.references,
        use_aggregator=False
    )
    mean_score = {k: sum(each_score[k]) / len(each_score[k]) for k in each_score.keys()}
    
    return MetricResult(
        metric_name='rouge',
        mean_score=mean_score,
        each_score=each_score
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute ROUGE score")
    parser.add_argument('--filepath', type=str, help='Path to the dataset file',
                        default='../../result/rag-eval/meta/context_only.json')
    # parser.add_argument('--model_id', type=str, default=None, help='Model ID for evaluation')
    # parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')
    # parser.add_argument('--slice', type=int, default=None, help='Slice of the dataset to process')
    args = parser.parse_args()

    compute_rouge(args.filepath, None, None, None)
    
    