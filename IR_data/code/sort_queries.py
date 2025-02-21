import json
from tqdm import tqdm

def sort_queries_by_question(input_file, output_file):
    # 입력 파일 읽기
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = [json.loads(line) for line in infile]

    sorted_data = []

    for item in tqdm(data):
        question = item['question']
        query = item['query']
        
        # stopwords를 제외한 query 단어와 question 내 등장 위치 추출
        query_with_index = []
        for q in query:
            if q in stopwords:
                continue
            try:
                # question에서 q의 위치를 찾고 query_with_index에 추가
                query_with_index.append((q, question.index(q)))
            except ValueError:
                # question에 없는 단어는 무시
                continue
        
        # query_with_index를 question에서의 등장 순서로 정렬
        sorted_query = [q[0] for q in sorted(query_with_index, key=lambda x: x[1])]
        
        # query가 비어 있으면 건너뛰기
        if not sorted_query:
            continue
        
        # 정렬된 query 업데이트
        item['query'] = sorted_query
        sorted_data.append(item)

    # 정렬된 데이터를 출력 파일에 저장
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for item in sorted_data:
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'{len(sorted_data)} saved')

# 사용 예시
input_file = '../data/query_updated.jsonl'
output_file = '../data/query_sorted.jsonl'
stopwords = ["무엇", "언제", "어디서", "위", "관", "은"]

sort_queries_by_question(input_file, output_file)
