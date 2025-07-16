from transformers import AutoModelForCausalLM, AutoTokenizer
from ._utils import *
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.distributed import distributed as dist
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def _answer_relevancy_worker(
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
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    model.to(rank)
    model.eval()
    ddp_model = DDP(model, device_ids=[rank])
    
    predictions_shard = predictions[rank::world_size]
    references_shard = references[rank::world_size]
    
    scores = []
    iterator = range(0, len(predictions_shard), batch_size)
    if rank == 0:
        iterator = tqdm(iterator, f'''Rank {rank} processing Answer Relevancy''', **('desc',))


def compute_answer_relevancy(filepath = None, model_id = None, batch_size = None, slice = (None, 16, None, 'context_only'), task = ('filepath', str, 'model_id', str, 'batch_size', int, 'slice', int, 'task', str, 'return', MetricResult)):
    '''
    Calculate the answer relevancy score between predictions and references.
    
    Args:
        filepath: str, Path to the file containing predictions and references.
    Returns:
        dict: The answer relevancy score results.
    '''
    master_port = find_available_port([
        12355,
        12356,
        12357,
        12358,
        12359])
    metric_dataset = prepare_dataset(filepath, slice)
    manager = mp.Manager()
    world_size = torch.cuda.device_count()
# WARNING: Decompyle incomplete

