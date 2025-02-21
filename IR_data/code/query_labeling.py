from argparse import ArgumentParser
import jsonlines
import re
from tqdm import tqdm
from typing import List, Dict, Any
from pyserini.search.lucene import LuceneSearcher
import json

JsonType = Dict[str, Any]

def read_jsonl(filepath: str) -> List[JsonType]:
    data = []
    with jsonlines.open(filepath, 'r') as jsonl_reader:
        for json_item in tqdm(jsonl_reader):
            data.append(json_item)
    return data

def find_passages(queries: List[JsonType],
                  method: str = 'SM',
                  searcher: LuceneSearcher = None,
                  passages: List[JsonType] = None,
                  file_num: int = None,
                  k: int = None) -> List[JsonType]:
    total_data = []
    if method == 'SM':
        assert searcher is not None, "Searcher has not been set"
        assert file_num is not None, "file_num has not been set"
        assert k is not None, "Top-K is not been set"
        
        for query in tqdm(queries, desc=f"Finding {file_num}", position=file_num):
            contents = _bm25_search(query, searcher, k=k)
            
            query_filtered, answer_filtered = [], []
            for content in contents:
                content_text = content['contents'].lower()  # contents를 소문자로 변환
                content_id = content['id']
                query_keywords = query['query']
                answer = query['answer']
                
                # Query filter logic
                # 키워드가 순서대로 나타나는지 확인
                last_index = -1
                is_in_order = True
                n = len(content_text)
                
                while last_index < n:
                    for keyword in query_keywords:
                        current_index = content_text.find(keyword, last_index+1)  # 키워드 위치 찾기
                        if current_index == -1:
                            is_in_order = False
                            break
                        last_index = current_index

                    # 순서대로 나타나면 추가
                    if is_in_order:
                        query_filtered.append(content_id)
                        break
                    
                    if not is_in_order:
                        break
                
                # Answer filter logic
                # 이때는 텍스트 원본 그대로 적용
                if answer in content['contents']:
                    answer_filtered.append(content_id)
            
            # Positive by Query, Positive by Answer가 없음
            if not query_filtered or not answer_filtered:
                continue
            
            data = {
                "query_id": query['query_id'],
                'question': query['question'],
                'query': query['query'],
                'answer': query['answer'],
                'gold_id': query['gold_id'],
                'positive_id_by_query': query_filtered,
                'positive_id_by_answer': answer_filtered
            }
            total_data.append(data)
    return total_data

def _bm25_search(query: JsonType,
                searcher: LuceneSearcher = None,
                k: int = 5000) -> List[JsonType]:
    assert searcher is not None, "Searcher has not been set"
    hits = searcher.search(query['question'], k=k)
    contents = []
    for i in range(len(hits)):
        doc_id = hits[i].docid
        content = searcher.doc(doc_id).raw()
        contents.append(json.loads(content))
    return contents
    
def save_to_jsonl(total_data, outfile_path):
    with open(outfile_path, 'w', encoding='utf-8') as outfile:
        for entry in total_data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')  # 줄바꿈 추가
    print(f"Data successfully saved to {outfile_path}")
    

def main():
    parser = ArgumentParser()
    parser.add_argument('--passage_path', default='../../data/passages.jsonl')
    parser.add_argument('--infile', default='../data/query')
    parser.add_argument('--outfile', default='../data/ir_data')
    parser.add_argument('--method', default='SM')
    parser.add_argument('--index_path', default='/home/nlplab/etri/QA/bm25/index')
    parser.add_argument('--file_num', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    args = parser.parse_args()
    
    searcher = LuceneSearcher(args.index_path)
    
    infile_path = f"{args.infile}_{args.file_num}.jsonl"
    outfile_path = f"{args.outfile}_{args.method}_{args.file_num}.jsonl"
    
    # passages = read_jsonl(args.passage_path)
    queries = read_jsonl(infile_path)
    
    total_data = find_passages(queries, args.method, searcher, None, args.file_num, args.k)
    save_to_jsonl(total_data, outfile_path)
    

if __name__ == "__main__":
    main()