import re
import json
import random
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from argparse import ArgumentParser
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map

def bm25_search(qas, searcher):
    hits = searcher.search(qas,k=3000)
    retrieved_passages = []
    for i in range(len(hits)):
        doc_id = hits[i].docid
        score = hits[i].score
        doc_content = searcher.doc(doc_id).raw()  # 문서의 실제 내용을 가져옴
        doc_content = json.loads(doc_content)
        retrieved_passage = dict()
        retrieved_passage['passage_id'] = doc_id
        retrieved_passage['text'] = doc_content['contents']
        retrieved_passage['score'] = score
        retrieved_passages.append(retrieved_passage)

    return retrieved_passages

def normalize_sentence_spacing(text):
    # 종결 부호(.!?) 뒤에 오는 모든 공백 제거 후 단일 공백 삽입
    text = re.sub(r'([.!?])\s*', r'\1 ', text)
    return text.strip()
    
def get_negative_passage(answer, passages, cfg):
    negative_passages = []
    
    for _ in range(10000):
        random_index = random.randint(0, len(passages) - 1)
        if answer not in passages[random_index]['contents']:
            retrieved_passage = dict()
            retrieved_passage['passage_id'] = passages[random_index]['id']
            retrieved_passage['text'] = passages[random_index]['contents']
            retrieved_passage['score'] = 0
            negative_passages.append(retrieved_passage)
        if len(negative_passages) >= cfg.negative_passages:
            break
            
    return negative_passages

def get_positive_passage(passages, answer, cfg):
    positive_passages = []
    norm_answer = normalize_sentence_spacing(answer)

    # answer passage 가져오기
    for passage in passages:
        norm_text = normalize_sentence_spacing(passage['text']) 
        if norm_answer in norm_text : # 정답이 retrieved passage안에 있을 경우
            positive_passages.append(passage)  

    return positive_passages[:cfg.positive_passages]  # 최대 10개의 passage 반환

def get_infer_passage(passages, cfg):
    positive_passages = []
    # answer passage 가져오기
    for passage in passages: 
        positive_passages.append(passage)  

    return positive_passages[:cfg.positive_passages]  # 최대 10개의 passage 반환


def get_hard_negative_passage(passages, answer, cfg):
    hard_negative_passages = []
    norm_answer = normalize_sentence_spacing(answer)

    for passage in passages:
        norm_text = normalize_sentence_spacing(passage['text'])
        if norm_answer not in norm_text:
            hard_negative_passages.append(passage)

    return hard_negative_passages[:cfg.hard_negative_passages]


def main():
    parser = ArgumentParser()
    parser.add_argument("--qas_path", help="QAS path", default="/home/nlplab/etri/QA/data/qas_refine_3.jsonl")
    parser.add_argument("--passages_path", help="Passages path", default="/home/nlplab/etri/QA/data/passages.jsonl")
    parser.add_argument("--index_path", help="BM25 Index Path")
    parser.add_argument("--positive_passages", help="# of positive passages", type=int, default=10)
    parser.add_argument("--negative_passages", help="# of random negative passages", type=int, default=50)
    parser.add_argument("--hard_negative_passages", help="# of bm25 negative passages", type=int, default=50)
    parser.add_argument("--train_path", help="Train path", default="/home/nlplab/etri/QA/train_dataset/data/kisti_train_more.json")
    parser.add_argument("--dev_path", help="Dev path", default="/home/nlplab/etri/QA/train_dataset/data/kisti_dev_more.json")
    parser.add_argument("--test_path", help="Test path", default="/home/nlplab/etri/QA/train_dataset/data/kisti_test_more.json")
    # parser.add_argument("--num_processes", help="Multiprocessing", type=int, default=16)
    args = parser.parse_args()

    # JSONL 파일 경로 지정
    qas_file = args.qas_path
    passages_file = args.passages_path

    # 파일 열기 및 passages 로드
    with open(passages_file, 'r', encoding='utf-8') as file:
        total_lines = sum(1 for _ in file)
        file.seek(0)  # 파일 포인터를 다시 시작 부분으로 이동
        passages = [json.loads(line) for line in tqdm(file, total=total_lines, desc="Loading Passages")]

    # 파일 열기
    with open(qas_file, 'r', encoding='utf-8') as file:
        qas_lines = file.readlines()
    
    searcher = LuceneSearcher(args.index_path)
    
    dataset = []
    hard_negative_error_count = 0

    for idx, line in enumerate(tqdm(qas_lines, total=len(qas_lines), desc=f"Processing")):
        question = json.loads(line)
        qas = question['question']
        ans = question['answer']
        if question.get('gold_id', None) is None:
            continue
        
        retrieved_passage = bm25_search(qas, searcher)
            
        ans_passage_content = json.loads(searcher.doc(question['gold_id']).raw())['contents']  # 문서의 실제 내용을 가져옴
        ans_passage = dict()
        ans_passage['text'] = ans_passage_content
        ans_passage['passage_id'] = question['gold_id']
        ans_passage['score'] = 100
        
        # Positive, Negative, Hard Negative Passages 추출
        positive_passages = get_positive_passage(retrieved_passage, ans, args)
        # infer_passages = get_infer_passage(retrieved_passage, args)
        negative_passage = get_negative_passage(ans, passages, args)
        hard_negative_passages = get_hard_negative_passage(retrieved_passage, ans, args)
        
        if len(hard_negative_passages) < 10:
            print(f"error at index {idx}")
            print(hard_negative_passages)
            hard_negative_error_count+=1
        else:
            # 데이터 구조 생성
            data_entry = {
                "id": question['id'],
                "dataset": "KISTI",
                "question": qas,
                "answer": [ans],
                "gold_ctxs" : ans_passage,
                "positive_ctxs": positive_passages,
                "negative_ctxs": negative_passage,
                "hard_negative_ctxs": hard_negative_passages
                # "infer_ctxs" : infer_passages
            }
            dataset.append(data_entry)
    
    # num_processes = args.num_processes
    # chunk_size = len(qas_lines) // num_processes
    # chunks = [qas_lines[i:i + chunk_size] for i in range(0, len(qas_lines), chunk_size)]
    
    # results = process_map(
    #     process_chunk,
    #     [(chunk_id, chunk, passages, args) for chunk_id, chunk in enumerate(chunks)],
    #     max_workers=num_processes,
    #     chunksize=1
    # )
    # with Pool(processes=num_processes) as pool:
    #     results = pool.starmap(process_chunk, [(chunk_id, chunk, passages, args) for chunk_id, chunk in enumerate(chunks)])
        
    print("10개 보다 적은 hard negative를 가지고 있는 샘플의 개수: ", hard_negative_error_count)

    # 랜덤으로 셔플
    random.shuffle(dataset)

    train_dataset = dataset[:-10000]
    dev_dataset = dataset[-10000:-5000]
    test_dataset = dataset[-5000:]

    print("length of train_dataset: ",len(train_dataset))
    print("length of dev_dataset: ",len(dev_dataset))
    print("length of test_dataset: ",len(test_dataset))

    with open(args.train_path, 'w', encoding='utf-8') as f_out:
        json.dump(train_dataset, f_out, indent=4, ensure_ascii=False)

    with open(args.dev_path, 'w', encoding='utf-8') as f_out:
        json.dump(dev_dataset, f_out, indent=4, ensure_ascii=False)

    with open(args.test_path, 'w', encoding='utf-8') as f_out:
        json.dump(test_dataset, f_out, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()