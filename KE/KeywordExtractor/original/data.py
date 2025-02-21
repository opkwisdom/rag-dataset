import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


class KPE_Dataset(Dataset):
    def __init__(self, docs_pairs):
        self.docs_pairs = docs_pairs
    
    def __len__(self):
        return len(self.docs_pairs)

    def __getitem__(self, idx):
        doc_pair = self.docs_pairs[idx]

        en_input_ids = doc_pair[0][0]
        en_input_mask = doc_pair[1][0]
        de_input_ids = doc_pair[2][0]
        dic = doc_pair[3]
        
        return [en_input_ids, en_input_mask, de_input_ids, dic]


class DataProcessor:
    def __init__(self, tokenizer, logger, config, args):
        self.tokenizer = tokenizer
        self.en_temp = config['en_temp']
        self.de_temp = config['de_temp']
        self.logger = logger
        self.config = config
        self.load_data(args.dataset_dir)
        self.compute_prefix_len()

    def load_data(self, filepath):
        '''
        Load jsonl file
        '''
        with open(filepath, 'r') as f:
            self.data = [json.loads(d) for d in f]

    def generate_dataset(self):
        '''
        Core API in data.py which returns the dataset
        '''
        docs_pairs = []
        doc_list = []
        doc_id_list = []

        for idx, json_data in enumerate(tqdm(self.data, desc="Generate dataset ")):
            doc_id, candidates, doc = json_data['id'], json_data['candidates'], json_data['contents']
            
            # Generate docs_pairs for constructing dataset 
            doc = self.en_temp + "\"" + doc + "\""
            max_len = self.config['max_len']
            chunks = self.chunk_text(doc, max_len)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunked_doc = self.en_temp + "\"" + chunk + "\""
                doc_pairs = self.generate_doc_pairs(chunked_doc, candidates, idx)
                docs_pairs.extend(doc_pairs)
                
            doc_list.append(doc)
            doc_id_list.append(doc_id)

        dataset = KPE_Dataset(docs_pairs)
        self.logger.info(f"Original examples: {len(self.data)}")
        self.logger.info(f"Total examples: {len(dataset)}")

        return dataset, doc_list, doc_id_list
    
    def generate_doc_pairs(self, doc, candidates, idx):
        '''
        Generate doc pairs using T5Tokenizer
        '''
        doc_pairs = []
        
        en_input = self.tokenizer(doc, max_length=self.config['max_len'], padding="max_length", truncation=True, return_tensors="pt")
        en_input_ids = en_input["input_ids"]
        en_input_mask = en_input["attention_mask"]
        
        for candidate, pos in candidates:
            de_input = self.de_temp + candidate + " ."
            de_input_ids = self.tokenizer(de_input, max_length=30, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
            # 실제 토큰 길이: <EOS> 찾고, 필요 없는 토큰 제외
            de_token_len = (de_input_ids[0] == self.tokenizer.eos_token_id).nonzero()[0].item() - 1

            # en_input_ids에 따라 pos 찾기
            dic = {"de_token_len":de_token_len, "candidate":candidate, "idx":idx, "pos":pos % self.config['max_len']}
            
            doc_pairs.append([en_input_ids, en_input_mask, de_input_ids, dic])
        return doc_pairs
    
    def chunk_text(self, text, max_len):
        chunks = [text[i : i + max_len] for i in range(0, len(text), max_len)]
        return chunks
    
    def compute_prefix_len(self):
        en_input_ids = self.tokenizer(self.en_temp, return_tensors='pt')['input_ids']
        de_input_ids = self.tokenizer(self.de_temp, return_tensors='pt')['input_ids']
        self.en_temp_len = (en_input_ids[0] == self.tokenizer.eos_token_id).nonzero()[0].item()
        self.de_temp_len = (de_input_ids[0] == self.tokenizer.eos_token_id).nonzero()[0].item()
    