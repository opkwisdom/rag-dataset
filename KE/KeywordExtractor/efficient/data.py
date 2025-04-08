import os
import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

# Custom collate function
# Docs와 Cans 별도로 처리
def custom_collate_fn(batch):
    docs = [item[0] for item in batch]
    cans = [item[1] for item in batch]
    
    return docs, cans

# Combined Dataset
class KPE_Dataset(Dataset):
    def __init__(self, doc_dataset, can_dataset):
        self.doc_dataset = doc_dataset
        self.can_dataset = can_dataset
    
    def __len__(self):
        return len(self.doc_dataset)
    
    def __getitem__(self, idx):
        doc = self.doc_dataset[idx]
        can = self.can_dataset[idx]
        
        return [doc, can]
    
# Doc Pair Dataset
class DocPairDataset(Dataset):
    def __init__(self, docs_pairs):
        self.docs_pairs = docs_pairs
    
    def __len__(self):
        return len(self.docs_pairs)

    def __getitem__(self, idx):
        doc_pair = self.docs_pairs[idx]

        en_input_ids = doc_pair["en_input_ids"]
        en_input_mask = doc_pair["en_input_mask"]
        de_input_ids = doc_pair["de_input_ids"]
        
        return [en_input_ids, en_input_mask, de_input_ids]

# Candidate Pair Dataset
class CandidatePairDataset(Dataset):
    def __init__(self, cans_pairs):
        self.cans_pairs = cans_pairs
    
    def __len__(self):
        return len(self.cans_pairs)
    
    def __getitem__(self, idx):
        can_pair = self.cans_pairs[idx]
        
        candidate = can_pair["candidate"]
        ca_input_ids = can_pair["ca_input_ids"]
        token_len = can_pair["token_len"]
        pos = can_pair["pos"]
        
        return [candidate, ca_input_ids, token_len, pos]


class EfficientDataProcessor:
    def __init__(self, tokenizer, logger, config, args):
        self.tokenizer = tokenizer
        self.en_temp = config['en_temp']
        self.de_temp = config['de_temp']
        self.logger = logger
        self.config = config
        self.args = args
        self.load_data(args.dataset_dir)
        self.compute_prefix_len()

    def load_data(self, filepath):
        '''
        Load jsonl file
        '''
        self.logger.info(f'Reading file...')
        _, ext = os.path.splitext(filepath)
        if ext == '.jsonl':
            with open(filepath, 'r') as f:
                self.data = [json.loads(d) for d in f]
        elif ext == '.json':
            with open(filepath, 'r') as f:
                self.data = json.load(f)

    def generate_dataset(self):
        '''
        Core API in data.py which returns the dataset
        '''
        docs_pairs = []
        cans_pairs = []
        doc_list = []
        doc_id_list = []
        candidate_list = []

        save_mode = self.config.get("save_mode", False)
        chunk_size = 100000  # 저장할 예제 수 간격
        chunk_counter = 0   # 저장 chunk 번호

        for idx, json_data in enumerate(tqdm(self.data, desc="Generate dataset ")):
            doc_id, candidates, doc = json_data['id'], json_data['candidates'], json_data['context']
            
            # Generate docs_pairs for constructing dataset 
            doc = self.en_temp + "\"" + doc + "\""
            max_len = self.config['max_len']
            chunks = self.chunk_text(doc, max_len)
            
            # Max length 기준으로 chunking
            for _, chunk in enumerate(chunks):
                chunked_doc = self.en_temp + "\"" + chunk + "\""
                doc_pairs, can_pairs = self.generate_doc_pairs(chunked_doc, candidates, idx)
                docs_pairs.append(doc_pairs)
                cans_pairs.append(can_pairs)
                
                doc_list.append(doc)
                doc_id_list.append(doc_id)
                candis = [candi for candi, _ in candidates]
                candidate_list.append(candis)
        
        # Construct dataset
        # 남은 데이터가 있을 경우 처리
        if docs_pairs:
            doc_pair_dataset = DocPairDataset(docs_pairs)
            candidate_pair_dataset = CandidatePairDataset(cans_pairs)
            kpe_dataset = KPE_Dataset(doc_pair_dataset, candidate_pair_dataset)
            
            if save_mode:
                output_path = f"{self.args.output_dir}_{chunk_counter}.pt"
                torch.save((kpe_dataset, doc_list, doc_id_list), output_path)
                self.logger.info(f"Final chunk {chunk_counter} saved with {len(docs_pairs)} examples at {output_path}")
            else:
                self.logger.info(f"Original examples: {len(self.data)}")
                self.logger.info(f"Total examples: {len(doc_pair_dataset)}")
                return kpe_dataset, doc_list, doc_id_list, candidate_list
            
    
    def generate_doc_pairs(self, doc, candidates, idx):
        '''
        Generate doc pairs using T5Tokenizer more efficiently
        '''
        doc_pairs, can_pairs = [], []
        
        en_input = self.tokenizer(doc, max_length=self.config['max_len'], padding="max_length", truncation=True, return_tensors="pt")
        en_input_ids = en_input["input_ids"]
        en_input_mask = en_input["attention_mask"]
        
        de_input = self.tokenizer(self.de_temp, truncation=True, return_tensors="pt")
        de_input_ids = de_input["input_ids"][0]
        
        doc_pairs = {
            "en_input_ids": en_input_ids,
            "en_input_mask": en_input_mask,
            "de_input_ids": de_input_ids,
            "idx": idx
        }
        
        candis = []
        cans_input_ids = []
        cans_token_len = []
        cans_pos = []
        
        # 각 candidate에 대해 tokenizing
        for candidate, pos in candidates:
            can_input_ids = self.tokenizer(candidate, max_length=30, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
            # 실제 토큰 길이: <EOS> 찾고, 필요 없는 토큰 제외
            can_token_len = (can_input_ids[0] == self.tokenizer.eos_token_id).nonzero()[0].item()
            
            candis.append(candidate)
            cans_input_ids.append(can_input_ids)
            cans_token_len.append(can_token_len)
            cans_pos.append(pos)
            
        can_pairs = {
            "candidate": candis,
            "ca_input_ids": cans_input_ids,
            "token_len": cans_token_len,
            "pos": cans_pos
        }
        
        return doc_pairs, can_pairs
    
    def chunk_text(self, text, max_len):
        chunks = [text[i : i + max_len] for i in range(0, len(text), max_len)]
        return chunks
    
    def compute_prefix_len(self):
        en_input_ids = self.tokenizer(self.en_temp, return_tensors='pt')['input_ids']
        de_input_ids = self.tokenizer(self.de_temp, return_tensors='pt')['input_ids']
        self.en_temp_len = (en_input_ids[0] == self.tokenizer.eos_token_id).nonzero()[0].item()
        self.de_temp_len = (de_input_ids[0] == self.tokenizer.eos_token_id).nonzero()[0].item()
    