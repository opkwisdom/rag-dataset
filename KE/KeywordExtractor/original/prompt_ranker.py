from tqdm import tqdm
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class PromptRanker:
    def __init__(self, model, tokenizer, logger, config, args):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger
        self.en_temp = config['en_temp']
        self.de_temp = config['de_temp']
        self.config = config
        self.args = args
        self.compute_prefix_len()

    def extract_keyphrases(self, dataloader, doc_list, original_doc_id_list):
        self.model.eval()
        
        doc_with_candidates_list = []
        candidate_list = []
        cos_score_list = []
        doc_id_list = []
        pos_list = []
        
        for id, [en_input_ids,  en_input_mask, de_input_ids, dic] in enumerate(tqdm(dataloader,desc="Evaluating ")):
            en_input_ids = en_input_ids.to(self.config['device'])
            en_input_mask = en_input_mask.to(self.config['device'])
            de_input_ids = de_input_ids.to(self.config['device'])
            
            batch_size = en_input_ids.shape[0]
            de_input_len = de_input_ids.shape[1]
            score = np.zeros(batch_size)
            
            with torch.no_grad():
                output = self.model(input_ids=en_input_ids,
                                    attention_mask=en_input_mask,
                                    decoder_input_ids=de_input_ids)[0]  # (B, L, V)
                
                for i in range(self.de_temp_len, de_input_len):
                    logits = output[:, i, :]
                    logits = logits.softmax(dim=-1) # (B, V)
                    logits = logits.cpu().numpy()
                    
                    for j in range(batch_size):
                        if i < dic['de_token_len'][j]:
                            score += np.log(logits[j, int(de_input_ids[j][i+1])])
                        elif i == dic['de_token_len'][j]:
                            score[j] /= np.power(dic['de_token_len'][j] - self.de_temp_len, self.config['length_factor'])
                
                candidate_list.extend(dic['candidate'])
                doc_id_list.extend(dic['idx'])
                pos_list.extend(dic['pos'])
                cos_score_list.extend(score)
                
        candidate_list = np.array(candidate_list)
        doc_id_list = np.array(doc_id_list)
        pos_list = np.array(pos_list)
        cos_score_list = np.array(cos_score_list)
        
        for i, _ in enumerate(tqdm(doc_list, desc="Ranking ")):
            doc_len = len(doc_list[i]) - (len(self.en_temp) + 2)
            
            # 같은 doc id에 대한 mask
            mask = (doc_id_list == i)
            doc_results = pd.DataFrame(
                np.column_stack((candidate_list[mask], cos_score_list[mask], pos_list[mask])),
                columns=['candidate', 'score', 'pos'],
            ).astype({'score': float, 'pos': int})
            
            # Pos 페널티
            pos_scores = doc_results['pos'] / doc_len + self.config['position_factor'] / np.power(doc_len, 3)
            doc_results['scores'] = doc_results['score'] * pos_scores
            
            ranked = doc_results.sort_values(by='score', ascending=False)
            top_10 = ranked.reset_index(drop=True)
            top_10_can = top_10.loc[:10, 'candidate'].values.tolist()
            
            doc_with_top_10 = {"id": original_doc_id_list[i], "doc": doc_list[i], "keyphrases": top_10_can}
            doc_with_candidates_list.append(doc_with_top_10)
        
        return doc_with_candidates_list
            
    
    def compute_prefix_len(self):
        en_input_ids = self.tokenizer(self.en_temp, return_tensors='pt')['input_ids']
        de_input_ids = self.tokenizer(self.de_temp, return_tensors='pt')['input_ids']
        self.en_temp_len = (en_input_ids[0] == self.tokenizer.eos_token_id).nonzero()[0].item()
        self.de_temp_len = (de_input_ids[0] == self.tokenizer.eos_token_id).nonzero()[0].item()