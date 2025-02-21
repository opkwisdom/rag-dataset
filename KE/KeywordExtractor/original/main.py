import json
import numpy as np
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader

from utils import set_config, setup_logger
from prompt_ranker import PromptRanker
from data import DataProcessor


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
    logger.info("Start Testing ...")

    model = AutoModelForSeq2SeqLM.from_pretrained(config['model_path'])
    model.to(config['device'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'])

    logger.info(f"Model: {model}")

    # Make dataset
    data_processor = DataProcessor(tokenizer, logger, config, args)
    dataset, doc_list, doc_id_list = data_processor.generate_dataset()

    dataloader = DataLoader(dataset, num_workers=4, batch_size=args.batch_size)
    prompt_ranker = PromptRanker(model, tokenizer, logger, config, args)
    keyphrases = prompt_ranker.extract_keyphrases(dataloader, doc_list, doc_id_list)

    with open(args.output_dir, "w") as f:
        json.dump(
            [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in item.items()} for item in keyphrases], 
            f, indent=4, ensure_ascii=False
        )


if __name__ == "__main__":
    main()