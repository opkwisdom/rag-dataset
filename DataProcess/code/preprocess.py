import os
import json
from tqdm import tqdm
import kss
from concurrent.futures import ProcessPoolExecutor
import sys

import re

def is_korean_percentage_below_threshold(text, threshold=0.10):
    """
    문서에서 한글 문자가 차지하는 비율이 지정된 비율(threshold) 이하인지 판단하는 함수.
    
    :param text: 문서 텍스트
    :param threshold: 비율 임계값 (기본값은 20%)
    :return: 한글 비율이 임계값(threshold) 이하인지 여부 (True/False)
    """
    # 한글 문자와 공백 외의 문자를 제외한 문자 수
    korean_characters = re.findall(r'[가-힣]', text)
    total_length = len(text)
    korean_length = len(korean_characters)
    
    # 비율 계산
    ratio = korean_length / total_length if total_length > 0 else 0
    
    # 비율이 임계값(threshold) 이하인지 확인
    return ratio <= threshold

def is_english(text, threshold=0.80):
    """
    문서에서 영어 문자가 차지하는 비율이 지정된 비율(threshold) 이상인지 판단하는 함수.
    
    :param text: 문서 텍스트
    :param threshold: 비율 임계값 (기본값은 80%)
    :return: 영어 비율이 임계값(threshold) 이상인지 여부 (True/False)
    """
    # 영어 알파벳과 공백 외의 문자를 제외한 문자 수
    english_characters = re.findall(r'[a-zA-Z]', text)
    total_length = len(text)
    english_length = len(english_characters)
    
    # 비율 계산
    ratio = english_length / total_length if total_length > 0 else 0
    
    # 비율이 임계값(threshold) 이상인지 확인
    return ratio >= threshold

def split_sentences(text):
    # 정규 표현식 패턴 정의
    # 다양한 구두점과 문장 경계 상황을 고려
    sentence_endings = re.compile(
        r'''
        (?<!\w\.\w\.)                # 약어 (예: U.S.)와 같은 패턴 제외
        (?<![A-Z][a-z]\.)            # 이니셜(예: "E.K.")과 같은 패턴 제외
        (?<![가-힣]\.[가-힣])         # 이니셜(예: "김.이")과 같은 패턴 제외
        (?<!\b[A-Z][a-z]\.\s[A-Z][a-z]\.)      # 영어 이름의 이니셜(예: "J. K.") 패턴 제외
        (?<=[.!?])                   # 문장 끝 구분자(예: ".", "?", "!") 뒤에
        (?=\s+|$)                    # 공백이 따르거나 문장의 끝인 경우
        |
        (?<=\.)                      # 마침표 다음에
        (?=\s*(?=[가-힣]))           # 공백이 있고 한글이 오는 경우
        |
        (?<=[가-힣][.!?])            # 한글 뒤에 마침표가 오고
        (?=\s*[가-힣])               # 그 다음에 한글이 있는 
        |
        (?<=\n)                      # 개행 문자 뒤에
        (?=\S)      
        ''', 
        re.VERBOSE
    )
    
    # 문장 분리
    sentences = sentence_endings.split(text)
    
    
    return sentences


prefix_path = "(분류) 국내 논문 QA 데이터셋"
# 총 276,804개의 논문(document) 존재 (27만개)
# 총 13,023,293개의 passage 존재 (1300만개)

def find_passage_for_answer(passages, answer_start,answer_text):
    # answer_start를 이용해 정답이 속한 passage를 찾음
    current_position = 0
    for passage in passages:
        passage_end = current_position + len(passage)
        if current_position <= answer_start < passage_end:
            # 정답이 passage 범위에 포함되나, 일부가 다음 passage로 넘어가는 경우를 처리
            answer_end = answer_start + len(answer_text)
            if answer_end > passage_end:
                return None  # 정답이 여러 passage에 걸쳐 있는 경우 예외 처리
            return passage
        current_position = passage_end + 1  # 공백을 고려해 위치 조정
    return None


def save_to_jsonl(file_path, data):
    with open(file_path, 'a', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')

def process_document_and_save(passages,qas,split):
    # context를 5문장 단위로 자르고, passage와 qa를 각각 jsonl 파일에 저장
    
    # passage들을 JSONL 파일로 저장
    save_to_jsonl(f"collection/passages_{split}.jsonl", passages)
    
    # 업데이트된 QAs를 JSONL 파일로 저장
    save_to_jsonl(f"collection/qas_{split}.jsonl", qas)

def combine_sentences(sentences, max_length=700, max_count=5):
    grouped_sentences = []
    current_group = ""
    sentence_count = 0

    for sentence in sentences:
        # 문장의 길이가 3 이하인 경우 바로 추가하고 카운트하지 않음
        if len(sentence) <= 3:
            current_group += sentence
        # 문장이 500글자를 넘으면 바로 그룹에 추가
        elif len(sentence) > max_length:
            if current_group:
                grouped_sentences.append(current_group)  # 현재 그룹 저장
            grouped_sentences.append(sentence)  # 500글자 넘는 문장은 그대로 추가
            current_group = ""
            sentence_count = 0  # 새 그룹 시작을 준비
        else:
            # 현재 그룹에 추가할 수 있으면 추가
            if len(current_group) + len(sentence) <= max_length and sentence_count < max_count:
                current_group += sentence
                sentence_count += 1
            else:
                # 그룹이 가득 찼으면 저장하고 새로운 그룹 시작
                grouped_sentences.append(current_group)
                current_group = sentence
                sentence_count = 1

    if current_group:  # 마지막 그룹이 남아있으면 추가
        grouped_sentences.append(current_group)
    
    return grouped_sentences

def read_save(files,split):
    cnt=0
    for file_path in tqdm(files):   
        with open(file_path, 'r', encoding='utf-8') as json_file: 
            try:
                json_data = json.load(json_file) # document opoen
                doc_id = json_data['doc_id']
                context = json_data['context']
                start = context.find("\n본문\n")# 테스트 예시
                if start==-1:
                    continue
                context = context[start+4:]
                sentences = split_sentences(context)
                passages = combine_sentences(sentences)
                passage_data = [{"id": f'{json_data["doc_id"]}-{i}', "contents": passage} for i, passage in enumerate(passages)]
                question_data = {}
                qa_temp = []
                for qa in json_data["qas"]:
                    answer_start = qa["answer"]["answer_start"]-start
                    if answer_start<0:
                        continue
                    answer_text = qa["answer"]["answer_text"]
                    related_passage = find_passage_for_answer([p["contents"] for p in passage_data], answer_start, answer_text)
                    if related_passage:
                        # 관련 passage ID를 추가
                        qa["gold_id"] = next(p["id"] for p in passage_data if p["contents"] == related_passage)
                    qa['answer'] = answer_text
                    qa['keyword'] = qa['keyword']['keyword_text']
                    qa_temp.append(qa)
                process_document_and_save(passage_data,qa_temp,split)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {file_path}: {e}")




def read_json_files_from_folder(folder_path,split):
    with open("path.json","r") as f:
        data = json.load(f)
    read_save(data[split],split)
    
if __name__ == '__main__':
    train_jsons = read_json_files_from_folder(prefix_path,sys.argv[1])