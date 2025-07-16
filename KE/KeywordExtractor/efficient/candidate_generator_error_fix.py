from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from kiwipiepy import Kiwi
import json
import os
import re
from collections import OrderedDict
from argparse import ArgumentParser

class CandidateGenerator:
    def __init__(self):
        self.tokenizer = Kiwi(model_type="sbg")
        self.tag_pattern = ['NNG', 'NNP', 'NR', 'NP', 'XSN', # 명사, 명사 파생 접미사
                            'SO', 'SW', 'SL', 'SH', 'SN']     # 외국어, 부호, 특수문자
        self.ignore = [
            ('이', 'NP'),
            ('그', 'NP'),
            ('저', 'NP'),
            ('반면', 'NNG'),
            ('각각', 'NNG')
        ]

    def tokenize(self, text):
        '''Single processing tokenization'''
        tokens = self.tokenizer.tokenize(text)
        candidate_words = self.extract_candidate_words(text, tokens)
        return candidate_words

    def extract_candidate_words(self, text, tokens):
        unique_words = OrderedDict()  # 순서를 유지하면서 중복 제거

        start_idx, end_idx = -1, -1

        for token in tokens:
            form, tag = token.form, token.tag
            if tag in self.tag_pattern and (form, tag) not in self.ignore:
                if start_idx == -1:
                    start_idx = token.start     # 시작 인덱스
                end_idx = token.start + token.len
            else:
                if start_idx != -1:
                    word = text[start_idx:end_idx]
                    if word not in unique_words:
                        unique_words[word] = None
                    start_idx, end_idx = -1, -1

        if start_idx != -1:
            word = text[start_idx:end_idx]
            if word not in unique_words:
                unique_words[word] = None
        
        result = list(unique_words.keys())
        filtered_result = []

        for i, word in enumerate(result):
            if not any(word in other_word and word != other_word for other_word in result):
                filtered_result.append(word)

        return filtered_result

def find_start_indices(total_data, total_candidates):
    assert len(total_data) == len(total_candidates)

    new_data = []
    for i in tqdm(range(len(total_data)),
                  desc=f'Attach indices file',
                  total=len(total_data),
                  ncols=180):
        data = total_data[i]
        candidates = total_candidates[i]
        contents = data['context']

        candidate_with_indices = []
        for candidate in candidates:
            start_idx = contents.find(candidate)
            candidate_with_indices.append((candidate, start_idx))
        data['candidates'] = candidate_with_indices
        new_data.append(data)
    return new_data

def load_data(filepath):
    '''
    Load JSONL/JSON/TXT file
    '''
    _, ext = os.path.splitext(filepath)
    if ext == '.jsonl':
        with open(filepath, 'r') as f:
            data = [json.loads(d) for d in f]
    elif ext == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
    elif ext == '.txt':
        with open(filepath, 'r') as f:
            lines = f.readlines()
            data = [line.strip() for line in lines]
    return data

def post_regex_process(text):
    text = re.sub(r'\([^)]*?et al\.[^)]*?\)', '', text)
    text = re.sub(r'([.!?])(?=\S)', r'\1 ', text)
    return re.sub(r'\s+', ' ', text).strip()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--infile_path", type=str, default='../../IR_data/collection/small_passages.jsonl')
    parser.add_argument("--outfile_path", type=str, default='generator_test.jsonl')
    parser.add_argument("--target_ids_path", type=str, required=True)
    args = parser.parse_args()

    infile_path = args.infile_path + ".jsonl"
    outfile_path = args.outfile_path + ".jsonl"

    data = load_data(infile_path)
    target_ids = load_data(args.target_ids_path)

    # Filter data based on target_ids
    target_data = []
    for item in data:
        if item['id'] in target_ids:
            item['context'] = post_regex_process(item['context'])
            target_data.append(item)
    

    # Candidate Generator Test
    generator = CandidateGenerator()
    candidates = []
    # import pdb; pdb.set_trace()
    for i, item in enumerate(tqdm(target_data,
                                  desc=f'Tokenizing file',
                                  total=len(target_data),
                                  ncols=180)):
        try:
            candidate_words = generator.tokenize(item['context'])
            candidates.append(candidate_words)
        except UnicodeDecodeError:
            print(f"UnicodeDecodeError at {i}")
            continue
    # Attach index
    filtered_data = find_start_indices(target_data, candidates)

    with open(outfile_path, 'w') as f:
        for line in filtered_data:
            json_line = json.dumps(line, ensure_ascii=False)
            f.write(json_line + '\n')