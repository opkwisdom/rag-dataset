from argparse import ArgumentParser
import jsonlines
import re
from tqdm import tqdm
from typing import List, Dict, Any
from pyserini.search.lucene import LuceneSearcher
import json

JsonType = Dict[str, Any]

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
            import pdb; pdb.set_trace();
            for content in contents:
                content_text = content['contents'].lower()  # contents를 소문자로 변환
                content_id = content['id']
                query_keywords = query['query']
                answer = query['answer']
                
                # Query filter logic
                # 키워드가 순서대로 나타나는지 확인
                last_index = -1
                is_in_order = True

                for keyword in query_keywords:
                    current_index = content_text.find(keyword)  # 키워드 위치 찾기
                    if current_index == -1 or current_index < last_index:
                        is_in_order = False
                        break
                    last_index = current_index

                # 순서대로 나타나면 추가
                if is_in_order:
                    query_filtered.append(content_id)
                
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

def read_jsonl(filepath: str) -> List[JsonType]:
    data = []
    with jsonlines.open(filepath, 'r') as jsonl_reader:
        for json_item in tqdm(jsonl_reader):
            data.append(json_item)
    return data


index_path = '/home/nlplab/etri/QA/bm25/index'
infile_path = '../data/query_sorted.jsonl'

query = {"query_id": "CFKO200403553211857-0", "question": "사회투자의 관점에서는 잊고 있는 다른 더 큰 특징은 무엇인가?", "answer": "연도별로 불규칙적이고 사회투자 부문간에도 불균등한 지출구조가 형성되고 있다는 점", "query": ["관점", "특징"], "gold_id": "CFKO200403553211857-15"}
contents = [{'id': "CFKO200421553662443-2", 'contents': "특징 관점 관점 특징"},
            {'id': "CFKO200421553662443-4", 'contents': "관점 특징 관점 특징"}]

total_data = []
answer_filtered, query_filtered = [], []
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
    
    import pdb; pdb.set_trace()
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
    
    # Answer filter logic
    # 이때는 텍스트 원본 그대로 적용
    if answer in content['contents']:
        answer_filtered.append(content_id)

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

print(total_data)