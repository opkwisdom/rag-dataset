from transformers import AutoModel, AutoTokenizer
from ._utils import *
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def _bertscore_worker(
    rank,
    world_size,
    master_port,
    model_id: str,
    predictions: list[str],
    references: list[str],
    batch_size: int,
    return_list
):
    setup_mp(rank, world_size, master_port)
    
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    model = AutoModel.from_pretrained('klue/bert-base')
    model.to(rank)
    model.eval()
    
    total_len = len(predictions)
    per_worker = total_len // world_size
    remainder = total_len % world_size
    
    start = rank * per_worker + min(rank, remainder)
    end = start + per_worker + (1 if rank < remainder else 0)
    predictions_shard = predictions[start:end]
    references_shard = references[start:end]
    scores = []
    
    iterator = range(0, len(predictions_shard), batch_size)
    if rank == 0:
        iterator = tqdm(iterator, desc=f"Rank {rank} processing BERTScore")
    
    for i in iterator:
        batch_predictions = predictions_shard[i:i+batch_size]
        batch_references = references_shard[i:i+batch_size]
        
        inputs_1 = tokenizer(batch_predictions, return_tensors='pt', padding=True, truncation=True).to(rank)
        inputs_2 = tokenizer(batch_references, return_tensors='pt', padding=True, truncation=True).to(rank)
        
        with torch.no_grad():
            last_hidden_1 = model(**inputs_1).last_hidden_state  # (B, L1, D)
            last_hidden_2 = model(**inputs_2).last_hidden_state  # (B, L2, D)
        
        emb_1_normalized = F.normalize(last_hidden_1, p=2, dim=-1)
        emb_2_normalized = F.normalize(last_hidden_2, p=2, dim=-1)
        
        cos_sim_matrix = torch.bmm(emb_1_normalized, emb_2_normalized.transpose(-2,-1))  # (B, L1, L2)
        
        precision = torch.mean(cos_sim_matrix.max(axis=1).values, axis=1)  # (B,)
        recall = torch.mean(cos_sim_matrix.max(axis=2).values, axis=1)     # (B,)

        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)  # Avoid division by zero
        
        for p, r, f1 in zip(precision, recall, f1_score):
            scores.append((p.item(), r.item(), f1.item()))
            
    return_list[rank] = scores
    dist.destroy_process_group()


def compute_bertscore(
    filepath: str,
    model_id: str = None,
    batch_size: int = 16,
    slice: int = None,
    task: str = 'context_only'
) -> MetricResult:
    '''
    Compute BERTScore for a given dataset.

    Args:
        filepath: str, Path to the JSON file.
        model_id: str, Model ID for the BERT model.
        batch_size: int, Batch size for processing.
        slice: int, Optional slice of the dataset to process.
        task: str, Task type to determine how to process texts.
    Returns:
        MetricResult: Result containing BERTScore.
    '''
    master_port = find_available_port([12355, 12356, 12357, 12358, 12359])
    
    metrics_dataset: MetricDataset = prepare_dataset(filepath, slice)
    manager = mp.Manager()
    world_size = torch.cuda.device_count()
    
    N = len(metrics_dataset.predictions)
    
    pred_results = manager.list([[] for _ in range(world_size)])
    
    print(f"Calculating BERTScore using klue/bert-base...")
    # torch multiprocessing
    mp.spawn(_bertscore_worker,
             args=(
                 world_size,
                 master_port,
                 model_id,
                 metrics_dataset.predictions,
                 metrics_dataset.references,
                 batch_size,
                 pred_results
             ),
            nprocs=world_size,
            join=True
    )
    
    all_scores = [score for sublist in pred_results for score in sublist]
    
    precisions = [s[0] for s in all_scores]
    recalls = [s[1] for s in all_scores]
    f1s = [s[2] for s in all_scores]
    
    return MetricResult(
        metric_name="bertscore",
        mean_score={
            "bert_p": sum(precisions) / len(precisions),
            "bert_r": sum(recalls) / len(recalls),
            "bert_f1": sum(f1s) / len(f1s),
        },
        each_score={
            "bert_p": precisions,
            "bert_r": recalls,
            "bert_f1": f1s,
        }
    )
# WARNING: Decompyle incomplete

