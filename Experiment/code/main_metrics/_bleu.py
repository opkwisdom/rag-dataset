from evaluate import load
from ._utils import *

def compute_bleu(
    filepath: str,
    model_id: str = None,
    batch_size: int = 16,
    slice: int = None,
    task: str = "context_only"
) -> MetricResult:
    '''
    Calculate the BLEU score between predictions and references.
    
    Args:
        filepath: str, Path to the file containing predictions and references.
        model_id: str, Model ID for the evaluation (not used in this function).
        batch_size: int, Batch size for processing (not used in this function).
        slice: int, Optional slice of the dataset to process.
        task: str, Task type to determine how to process texts.
    Returns:
        MetricResult: Result containing BLEU scores.
    '''
    metric_dataset: MetricDataset = prepare_dataset(filepath, slice)
    bleu = load('bleu')
    print('Calculating BLEU...')
    
    each_score = [
        0.0 if not pred or not ref else bleu.compute(
            predictions=[pred],
            references=[ref],
            smooth=True
        )['bleu']
        for pred, ref in zip(metric_dataset.predictions, metric_dataset.references)
    ]
    
    return MetricResult(
        metric_name='bleu',
        mean_score={'bleu': sum(each_score) / len(each_score)},
        each_score={'bleu': each_score}
    )
