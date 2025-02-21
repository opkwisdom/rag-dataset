from argparse import ArgumentParser
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, List, Tuple
import os
import json
from tqdm import tqdm
import torch
from pyserini.search.lucene import LuceneSearcher

JsonType = Dict[str, Any]

def read_file(filepath: str) -> List[JsonType]:
    _, ext = os.path.splitext(filepath)
    
    print(f'Reading from {filepath}...')
    if ext == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
    elif ext == '.jsonl':
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return data

def scoring(data, model, tokenizer, searcher, top_k):
    scored_data = []
    model.eval()
    
    pbar_outer = tqdm(data, desc="Scoring", unit='query')
    for d in pbar_outer:
        passage_scores: Tuple[str, float] = []
        
        # Question
        question = d["question"]
        question_inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
        question_embedding = model(**question_inputs).last_hidden_state[:, 0, :]
            
        # Passage
        positive_ids = d['positive_id']
        pbar_inner = tqdm(positive_ids, desc="Processing Passages", leave=False, unit="doc")
        
        for doc_id in pbar_inner:
            content = json.loads(searcher.doc(doc_id).raw())
            content_inputs = tokenizer(content['contents'], return_tensors="pt", padding=True, truncation=True)
            content_embedding = model(**content_inputs).last_hidden_state[:, 0, :]
            
            similarity = torch.matmul(question_embedding, content_embedding.T).squeeze().item()
            passage_scores.append((doc_id, similarity))
        
        # Top-k 정렬
        passage_scores.sort(key=lambda x: x[1], reverse=True)
        top_k_scores = passage_scores[:top_k]
        
        relevant_ids = [doc_id for doc_id, _ in top_k_scores]
        scores = [score for _, score in top_k_scores]
        
        d['relevant_ids'] = relevant_ids
        d['scores'] = scores
        d.pop('positive_id')
        
        scored_data.append(d)
        
    return scored_data

def save_to_jsonl(total_data, outfile_path):
    with open(outfile_path, 'w', encoding='utf-8') as outfile:
        for entry in total_data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')  # 줄바꿈 추가
    print(f"Data successfully saved to {outfile_path}")
    

def main():
    parser = ArgumentParser()
    parser.add_argument('--model_path', help='huggingface model path', required=True)
    parser.add_argument('--top_k', help='Maximum number of relevant passages', type=int, default=10)
    parser.add_argument('--infile_path', help='Input file path', required=True)
    parser.add_argument('--outfile_path', help='Output file path', required=True)
    parser.add_argument('--index_path', help='BM25 index path', default='/home/sangwon/nlplab_leaderboard2/bm25/index')
    args = parser.parse_args()
    
    # 모델 로드하기
    searcher = LuceneSearcher(args.index_path)
    
    model = AutoModel.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    data = read_file(args.infile_path)
    scored_data = scoring(data, model, tokenizer, searcher, args.top_k)
    
    save_to_jsonl(scored_data, args.outfile_path)
    

if __name__ == "__main__":
    main()