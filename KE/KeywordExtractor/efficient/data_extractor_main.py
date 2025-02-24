import json
import numpy as np
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
import pickle
import torch

from utils import set_config, setup_logger
from prompt_ranker import CachedPromptRanker
from data import EfficientDataProcessor, custom_collate_fn


def parse_argument():
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input dataset.")
    parser.add_argument("--batch_size",
                        default=None,
                        type=int,
                        required=True,
                        help="Batch size for testing.")
    parser.add_argument("--log_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="Path for Logging file")
    parser.add_argument("--log_filename",
                        default=None,
                        type=str,
                        required=True,
                        help="Logging filename")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="Output directory")
    args = parser.parse_args()
    return args


def main():
    config = set_config()
    args = parse_argument()

    logger = setup_logger(args.log_dir, args.log_filename)
    logger.info("Start Extracting ...")

    model = AutoModelForSeq2SeqLM.from_pretrained(config['model_path'])
    model.to(config['device'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'])

    # Make dataset
    data_processor = EfficientDataProcessor(tokenizer, logger, config, args)
    kpe_dataset, doc_list, doc_id_list = data_processor.generate_dataset()
    
    logger.info(f"Dataset size: {len(kpe_dataset)}")
    logger.info(f"Save dataset to {args.output_dir}")
    torch.save((kpe_dataset, doc_list, doc_id_list), args.output_dir)

    kpe_dataset, doc_list, doc_id_list = torch.load(args.output_dir)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()