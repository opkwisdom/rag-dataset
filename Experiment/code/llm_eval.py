from argparse import ArgumentParser
import random
import json
from datasets import Dataset

from main_metrics import *

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    AnswerRelevancy,
)

def load_json(filepath, slice: int = None, permute: bool = False):
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    if permute:
        random.seed(42)
        random.shuffle(data)
    
    if slice is not None:
        data = data[:slice]
    return data

def build_dataset(filepath, slice: int = None, permute: bool = False, task: str = 'context_only'):
    raw_data = load_json(filepath, slice=slice, permute=permute)

    context_key_list = []
    if task == 'context_only':
        context_key_list.append("context")
    elif task == 'keyphrase_only':
        context_key_list.append("keyphrases")
    elif task == 'keysentence':
        context_key_list.append("extracted_sentences")
    elif task == 'query_only':
        context_key_list.append("context")
    elif task == 'context_keyphrase':
        context_key_list.append("keyphrases")
        context_key_list.append("context")
    
    dataset_list = []
    for item in raw_data:
        retrieved_contexts = ""
        for key in context_key_list:
            if key in item:
                if isinstance(item[key], list):
                    retrieved_contexts += " ".join(item[key]) + " "
                else:
                    retrieved_contexts += item[key] + " "
        
        dataset_list.append({
            "user_input": item["question"],
            "retrieved_contexts": [retrieved_contexts],
            "response": item["predicted_answer"],        
            "reference": item["gold_answer"]          
        })
    
    return Dataset.from_list(dataset_list)

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
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of new tokens to generate')
    args = parser.parse_args()
    print(f"Args: {args}")
    
    input_path = f"{args.input_dir}/{args.infile_name}.json"
    output_path = f"{args.output_dir}/{args.outfile_name}.json"
    
    # Prepare model and dataset
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model=args.model_name_or_path)
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings()
    )
    
    dataset = build_dataset(
        filepath=input_path,
        slice=args.slice,
        permute=True,
        task=args.task
    )
    
    # Evaluate metrics
    results = evaluate(
        dataset=dataset,
        metrics=[FactualCorrectness(), Faithfulness(), AnswerRelevancy()],
        llm=evaluator_llm,
        embeddings=embeddings
    )
    average_scores = {k: float(v) for k, v in results._repr_dict.items()}
    
    # Aggregate results
    json_results = {
        "task": args.task,
        "evaluator": args.model_name_or_path,
        "metrics": {
            "average_scores": average_scores,
            "total_scores": results.scores
        }
    }
    
    save_results(output_path, json_results)
    
if __name__ == "__main__":
    main()