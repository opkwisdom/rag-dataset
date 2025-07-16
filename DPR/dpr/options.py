'''
Command line arguments utils
'''
import logging
import os
import random
import socket
import subprocess
from typing import Tuple
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
logger = logging.getLogger()

def set_cfg_params_from_state(state = None, cfg = None):
    '''
    Overrides some of the encoder config parameters from a give state object
    '''
    if not state:
        return None
    cfg.do_lower_case = None['do_lower_case']
    if 'encoder' in state:
        saved_encoder_params = state['encoder']
        OmegaConf.set_struct(cfg, False)
        cfg.encoder = saved_encoder_params
        OmegaConf.set_struct(cfg, True)
        return None


def get_encoder_params_state_from_cfg(cfg = None):
    '''
    Selects the param values to be saved in a checkpoint, so that a trained model can be used for downstream
    tasks without the need to specify these parameter again
    :return: Dict of params to memorize in a checkpoint
    '''
    return {
        'do_lower_case': cfg.do_lower_case,
        'encoder': cfg.encoder }


def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
        return None


def setup_cfg_gpu(cfg):
    '''
    Setup params for CUDA, GPU & distributed training
    '''
    logger.info("CFG's local_rank=%s", cfg.local_rank)
    ws = os.environ.get('WORLD_SIZE')
    cfg.distributed_world_size = int(ws) if ws else 1
    logger.info('Env WORLD_SIZE=%s', ws)
    import pdb
    pdb.set_trace()
    if cfg.distributed_port and cfg.distributed_port > 0:
        logger.info('distributed_port is specified, trying to init distributed mode from SLURM params ...')
        (init_method, local_rank, world_size, device) = _infer_slurm_init(cfg)
        logger.info('Inferred params from SLURM: init_method=%s | local_rank=%s | world_size=%s', init_method, local_rank, world_size)
        cfg.local_rank = local_rank
        cfg.distributed_world_size = world_size
        cfg.n_gpu = 1
        torch.cuda.set_device(device)
        device = str(torch.device('cuda', device))
        torch.distributed.init_process_group('nccl', init_method, world_size, local_rank, **('backend', 'init_method', 'world_size', 'rank'))
    elif cfg.local_rank == -1 or cfg.no_cuda:
        device = str(torch.device('cuda' if not torch.cuda.is_available() and cfg.no_cuda else 'cpu'))
        cfg.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(cfg.local_rank)
        device = str(torch.device('cuda', cfg.local_rank))
        torch.distributed.init_process_group('nccl', **('backend',))
        cfg.n_gpu = 1
    cfg.device = device
    logger.info('Initialized host %s as d.rank %d on device=%s, n_gpu=%d, world size=%d', socket.gethostname(), cfg.local_rank, cfg.device, cfg.n_gpu, cfg.distributed_world_size)
    logger.info('16-bits training: %s ', cfg.fp16)
    return cfg


def _infer_slurm_init(cfg = None):
    node_list = os.environ.get('SLURM_STEP_NODELIST')
    if node_list is None:
        node_list = os.environ.get('SLURM_JOB_NODELIST')
    logger.info('SLURM_JOB_NODELIST: %s', node_list)
    if node_list is None:
        raise RuntimeError("Can't find SLURM node_list from env parameters")
    local_rank = None
    world_size = None
    distributed_init_method = None
    device_id = None


def setup_logger(logger):
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_formatter = logging.Formatter('[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s')
    console = logging.StreamHandler()
    console.setFormatter(log_formatter)
    logger.addHandler(console)