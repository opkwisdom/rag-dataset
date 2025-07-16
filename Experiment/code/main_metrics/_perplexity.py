from transformers import AutoModelForCausalLM, AutoTokenizer
from ._utils import *
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def _cppl_worker(
    rank,
    world_size,
    master_port,
    model_id: str,
    texts: list[str],
    batch_size: int,
    return_list
):
    setup_mp(rank, world_size, master_port)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    eos_token_id = tokenizer.encode("[EOS]")[0]
    
    model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": rank}
        )
    model.eval()
    
    total_len = len(texts)
    per_worker = total_len // world_size
    remainder = total_len % world_size
    
    start = rank * per_worker + min(rank, remainder)
    end = start + per_worker + (1 if rank < remainder else 0)
    shard = texts[start:end]
    scores = []
    
    iterator = range(0, len(shard), batch_size)
    if rank == 0:
        iterator = tqdm(iterator, desc=f"Rank {rank} processing PPL")

    for i in iterator:
        batch = shard[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(rank)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits   # logits per token, (B, L, V)

        shift_logits = logits[:, :-1, :].contiguous()  # Shift logits to the left
        shift_labels = inputs["input_ids"][:, 1:].contiguous()  # Shift
        
        for b in range(len(batch)):
            eos_pos = (inputs["input_ids"][b] == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                first_eos = eos_pos[0].item()
                shift_labels[b, :first_eos] = tokenizer.pad_token_id

        vocab_size = shift_logits.size(-1)
        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)
        loss_per_token = loss_fct(shift_logits.view(-1, vocab_size),
                                  shift_labels.view(-1)
                                  ).view(shift_labels.size())   # (B, L)
        
        token_count = (shift_labels != tokenizer.pad_token_id).sum(dim=1)
        sent_loss = loss_per_token.sum(dim=-1) / token_count.clamp(min=1)
        sent_ppl = torch.exp(sent_loss).detach().cpu().tolist()
        
        scores.extend(sent_ppl)
        
    return_list[rank] = scores
    dist.destroy_process_group()


def compute_conditional_ppl(
    filepath: str,
    model_id: str = None,
    batch_size: int = 16,
    slice: int = None,
    task: str = 'context_only'
) -> MetricResult:
    """
    Calculate the delta conditional perplexity between predictions and references.
    
    Args:
        filepath: str, Path to the file containing predictions and references.
        model_id: str, Identifier for the model used for evaluation.
        batch_size: int, Batch size for processing.
        slice: int, Optional slice of the dataset to process.
        task: str, Task type to determine how to process texts.
    Returns:
        float: The delta conditional perplexity score.
    """
    master_port = find_available_port([12355, 12356, 12357, 12358, 12359])

    metric_dataset: MetricDataset = prepare_ppl_dataset(filepath, slice, task)
    manager = mp.Manager()
    world_size = torch.cuda.device_count()

    N = len(metric_dataset.predictions)

    ref_results = manager.list([[] for _ in range(world_size)])
    
    print(f"Calculating Conditional Perplexity using {model_id}...")
    # torch multiprocessing
    mp.spawn(_cppl_worker,
             args=(
                world_size,
                master_port,
                model_id,
                metric_dataset.references,
                batch_size,
                ref_results
            ),
            nprocs=world_size,
            join=True
    )
    cppl = [r for sublist in ref_results for r in sublist]
    
    mean_cppl =  sum(cppl) / len(cppl)

    return MetricResult(
        metric_name="conditional_ppl",
        mean_score={"conditional_ppl": mean_cppl},
        each_score={"conditional_ppl": cppl}
    )