# Source Generated with Decompyle++
# File: _utils.cpython-310.pyc (Python 3.10)

from dataclasses import dataclass, field
import json
import os
import torch
import torch.distributed as dist
import socket

@dataclass
class MetricDataset:
    predictions: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

@dataclass
class MetricResult:
    metric_name: str
    mean_score: dict[str, float] = field(default_factory=dict)
    each_score: dict[str, list[float]] = field(default_factory=dict)
    

def load_json(path = None, slice = None):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if slice is not None:
        data = data[:slice]
    return data


def save_results(output_path, results):
    '''결과를 JSON 파일로 저장'''
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def prepare_dataset(filepath: str, slice: int = None, task: str = None) -> MetricDataset:
    '''
    Prepare dataset for evaluation.
    
    Args:
        filepath: str, Path to the file containing predictions and references.
    
    Returns:
        MetricDataset: A dataclass containing predictions and references.
    '''
    data = load_json(filepath, slice)
    predictions = [item['predicted_answer'] for item in data]
    references = [item['gold_answer'] for item in data]
    return MetricDataset(predictions=predictions, references=references)


def prepare_ppl_dataset(filepath: str, slice: int = None, task: str = None) -> MetricDataset:
    """
    Prepare dataset for conditional perplexity evaluation.
    
    Args:
        filepath: str, Path to the file containing predictions and references.
    
    Returns:
        MetricDataset: A dataclass containing predictions and references.
    """
    data = load_json(filepath, slice)
    if task == 'context_only':
        refer = ['context']
    elif task == 'keyphrase_only':
        refer = ['keyphrases']
    elif task == 'keysentence':
        refer = ['extracted_sentences']
    elif task == 'query_only':
        refer = None
    elif task == 'context_keyphrase':
        refer = ['keyphrases', 'context']
    else:
        raise ValueError(f"Unknown task type: {task}")
    if refer is not None:
        if task == 'context_keyphrase':
            predictions = [
                item["question"] + " " + item[refer[1]] + " " +
                " ".join(item[refer[0]]) + "[EOS]" +
                item["predicted_answer"]
                for item in data
            ]
            references = [
                item["question"] + " " + item[refer[1]] + " " +
                " ".join(item[refer[0]]) + "[EOS]" +
                item["gold_answer"]
                for item in data
            ]
        elif isinstance(data[0][refer[0]], list):
            predictions = [
                item['question'] + ' ' + ' '.join(item[refer[0]]) + '[EOS]' + item['predicted_answer']
                for item in data
            ]
            references = [
                item['question'] + ' ' + ' '.join(item[refer[0]]) + '[EOS]' + item['gold_answer']
                for item in data
            ]
        else:
            predictions = [
                item['question'] + ' ' + item[refer[0]] + '[EOS]' + item['predicted_answer']
                for item in data
            ]
            references = [
                item['question'] + ' ' + item[refer[0]] + '[EOS]' + item['gold_answer']
                for item in data
            ]
    else:
        predictions = [item['question'] + '[EOS]' + item['predicted_answer'] for item in data]
        references = [item['question'] + '[EOS]' + item['gold_answer'] for item in data]
    return MetricDataset(predictions=predictions, references=references)


def find_available_port(port_options):
    for port in port_options:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return str(port)
            except OSError:
                continue
    raise RuntimeError(f"No available port found in {port_options}")


def setup_mp(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    print(f'''[Rank {rank}] initializing process group at port {master_port}''')
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

