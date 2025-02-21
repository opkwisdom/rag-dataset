from keybert import KeyBERT
from flair.embeddings import TransformerDocumentEmbeddings
from konlpy.tag import Kkma
from typing import List, Tuple
from tqdm import tqdm
import jsonlines
import json
from dataclasses import dataclass
from argparse import ArgumentParser
import sys

@dataclass
class Pos:
    word: str
    tag: str
    start: int
    end: int


def process_pos_tags(keywords_list: List[str], pos_list: List[Tuple[str, str]]) -> List[List[Pos]]:
    '''
    형태소 분석 결과(Pos)를 기반으로 전성 어미(ET, ETD, ETN)를 제거하고,
    각 형태소에 대해 원문 문자열에서의 시작 인덱스(start)와 끝 인덱스(end)를 태깅하는 함수.

    Params:
        - keywords_list: 형태소 분석이 적용된 원문 문자열 리스트
        - pos_list: 각 문자열의 형태소와 품사 태그를 포함한 튜플 리스트
    
    Returns:
        - 각 문자열에 대해 단어, 품사 태그, 시작/끝 인덱스를 포함한 Pos 객체 리스트
    '''
    one_syllable = set(['ET', 'ETD', 'ETN'])
    total_pos_list = []
    
    for keywords, pos_tags in zip(keywords_list, pos_list):
        pos_with_indices = []
        current_idx = 0

        for word, tag in pos_tags:
            if tag not in one_syllable:
                start_idx = keywords.find(word, current_idx)
                end_idx = start_idx + len(word)
                pos_with_indices.append(
                    Pos(
                        word=word,
                        tag=tag,
                        start=start_idx,
                        end=end_idx
                    )
                )
                current_idx = end_idx

        total_pos_list.append(pos_with_indices)

    return total_pos_list

def split_into_keywords(keywords_list: List[str], pos_list: List[List[Pos]]) -> List[str]:
    '''
    Pos에 따라 명사를 추출하여 키워드 그룹을 생성하는 함수
    
    Params:
        - keywords_list: 형태소 분석이 적용된 원문 문자열 리스트
        - pos_list: 각 문자열에 대해 형태소와 품사 태그 정보를 포함한 Pos 객체 리스트
    
    Returns:
        - 명사 그룹으로 구성된 키워드 리스트
    '''
    # 명사 : NN, NNG, NNP, OL, OH
    assert len(keywords_list) == len(pos_list), f'# of keywords: {len(keywords_list)}, # of pos: {len(pos_list)}'
    total_key_set = set()
    noun_tags = set(['NN', 'NNG', 'NNP', 'OL', 'OH'])   # O(1) Search
    
    # import pdb; pdb.set_trace();
    for keywords, pos_tags in zip(keywords_list, pos_list):
        start, end = None, None
        
        for pos in pos_tags:
            if pos.tag in noun_tags:
                if start is None:
                    start = pos.start
                end = pos.end
            else:
                if start is not None and end is not None:
                    total_key_set.add(keywords[start:end])
                    start, end = None, None
    
        if start is not None and end is not None:
            total_key_set.add(keywords[start:end])
    
    return list(total_key_set)

def merge_keywords(keywords_list: List[str]) -> List[str]:
    # 리스트를 길이 기준으로 내림차순 정렬 (더 긴 키워드가 앞에 오도록)
    keywords_list = sorted(keywords_list, key=len, reverse=True)
    
    merged_keywords = []

    # 포함 여부를 확인하면서 키워드 병합
    for keyword in keywords_list:
        if not any(keyword in existing and existing != keyword for existing in merged_keywords):
            merged_keywords.append(keyword)
    
    return merged_keywords


def process_queries(json_data_list, model_file, split):
    '''Job function'''
    klue_bert = TransformerDocumentEmbeddings(model_file)
    kw_model = KeyBERT(model=klue_bert)
    kkma = Kkma()

    results = []
    for json_data in tqdm(json_data_list, desc=f"Processing Split {split}", position=split):
        question = json_data["question"]
        keywords_list = kw_model.extract_keywords(question, keyphrase_ngram_range=(1, 3), top_n=10)
        keywords_list = [k for k, s in keywords_list]  # score 제거
        pos_list = [kkma.pos(keywords) for keywords in keywords_list]  # Pos 태깅

        total_pos_list = process_pos_tags(keywords_list, pos_list)  # post-processing
        total_list = split_into_keywords(keywords_list, total_pos_list)
        keywords_set = merge_keywords(total_list)

        results.append({
            "query_id": json_data["query_id"],
            "question": json_data["query"],
            "query": keywords_set,
            "answer": json_data['answer'],
            "gold_id": json_data["gold_id"],
            "positive_id": json_data['positive_id']
        })
            
    return results

def main():
    parser = ArgumentParser()
    parser.add_argument('--infile_path', default="./collection/question")
    parser.add_argument('--outfile_path', default="./collection/query")
    parser.add_argument('--model_file', default="klue/bert-base")
    parser.add_argument('--file_postfix', type=int)
    args = parser.parse_args()
    
    split = args.file_postfix
    infile_path = f"{args.infile_path}_{split}.jsonl"
    with jsonlines.open(infile_path, "r") as jsonl_reader:
        json_data_list = [json_data for json_data in tqdm(jsonl_reader, desc="Reading Input")]
    
    results = process_queries(json_data_list, args.model_file, split)
    
    # 결과 저장
    outfile_path = f"{args.outfile_path}_{split}.jsonl"
    with jsonlines.open(outfile_path, mode="w") as jsonl_writer:
        jsonl_writer.write_all(results)



if __name__ == "__main__":
    main()