from tqdm import tqdm
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.nn.utils.rnn import pad_sequence
import einops
from sklearn.feature_extraction.text import TfidfVectorizer

class IDFPromptRanker:
    def __init__(self, model, tokenizer, idfs, logger, config, args):
        self.model = model
        self.tokenizer = tokenizer
        self.idfs = idfs
        self.logger = logger
        self.en_temp = config['en_temp']
        self.de_temp = config['de_temp']
        self.config = config
        self.args = args
        self.compute_prefix_len()

    def extract_keyphrases(self, dataloader, doc_list, original_doc_id_list):
        self.model.eval()
        
        doc_with_candidates_list = []
        
        for i, (docs, cans) in enumerate(tqdm(dataloader, desc="Evaluating ", leave=False)):
            # Unpack data
            en_input_ids, en_input_mask, de_input_ids = zip(*docs)
            candidate, ca_input_ids, token_len, pos = zip(*cans)
            
            en_input_ids = torch.cat(en_input_ids)
            en_input_mask = torch.cat(en_input_mask)
            de_input_ids = torch.stack(de_input_ids)
            
            candidate = list(candidate) # List[List[str]]
            # IDF 추가
            candidate_idfs = []
            for candi in candidate:
                idf_values = [self.idfs[c] for c in candi]
                candidate_idfs.append(torch.tensor(idf_values, dtype=torch.float))   # candidate 별 IDF
            candidate_idfs = pad_sequence(candidate_idfs, batch_first=True, padding_value=0)  # (B, N)

            # 병렬 처리를 위해 텐서로 변환 (Pad)
            ca_input_ids = [torch.cat(input_ids) for input_ids in list(ca_input_ids)]
            ca_input_ids = pad_sequence(ca_input_ids, batch_first=True, padding_value=0)    # (B, N, dec_L)
            token_len = [torch.tensor(t_l) for t_l in token_len]
            token_len = pad_sequence(token_len, batch_first=True, padding_value=0)  # (B, N)
            pos = [torch.tensor(p) for p in pos]
            pos = pad_sequence(pos, batch_first=True, padding_value=0)  # (B, N)
            
            # Move to device
            en_input_ids = en_input_ids.to(self.config['device'])
            en_input_mask = en_input_mask.to(self.config['device'])
            de_input_ids = de_input_ids.to(self.config['device'])
            ca_input_ids = ca_input_ids.to(self.config['device'])
            candidate_idfs = candidate_idfs.to(self.config['device'])
            token_len = token_len.to(self.config['device'])
            pos = pos.to(self.config['device'])
            
            batch_size = ca_input_ids.shape[0]
            n_candidates = ca_input_ids.shape[1]
            # Candidate padding
            padded_candidates = [row + [''] * (n_candidates - len(row)) for row in candidate]
            candidate_np = np.array(padded_candidates)
            candidate_len = torch.tensor(
                np.char.str_len(candidate_np),
                device=self.config['device']
            )
            
            # Prefix 출력 계산
            with torch.no_grad():
                encoder_outputs = self.model.encoder(
                    input_ids=en_input_ids,
                    attention_mask=en_input_mask
                )[0]    # (B, enc_L, D)
            
            # Candidate 출력 계산
            with torch.no_grad():
                # 차원 맞추기
                # Memory size in case of batch 4
                # 1. flattend_ca_input_ids: 0.035MB (8bytes)
                # 2. expanded_encoder_outputs: 234MB (4bytes)
                # 3. expanded_past_key_values: 5.6GB (4bytes)
                # 한번에 가는 것이 아닌 배치별 seq_len만큼
                combined_ca_input_ids = torch.cat(
                    [
                        einops.repeat(
                            de_input_ids.unsqueeze(dim=1),
                            'b 1 dec_pre_L -> b N dec_pre_L',
                            N=n_candidates
                        ),  # (B, dec_pre_L) -> (B, 1, dec_pre_L)
                        ca_input_ids,    # (B, N, dec_L)
                    ],
                    dim=-1
                )   # (B, N, pre + dec_L)

                B, N, L = combined_ca_input_ids.shape

                # N개 독립적으로 처리
                batch_logits_list = []
                for j in range(N):
                    each_ca_input_ids = combined_ca_input_ids[:, j, :]
                    output = self.model(
                        encoder_outputs=(encoder_outputs,),  # Encoder 결과 재사용
                        decoder_input_ids=each_ca_input_ids,  # Candidate 입력
                    )[0]    # logits of Seq2SeqLMOutput (B, dec_L, V)
                    batch_logits_list.append(output)

                # (B, N, dec_L, V)
                batch_logits = torch.stack(batch_logits_list, dim=1)
                batch_logits = batch_logits[:, :, self.de_temp_len+1:, :]  # Prefix 제거
                batch_logits = batch_logits.softmax(dim=-1)
                
                # Candidate 별 점수 계산
                # 1. Indexing
                indexed_logits = torch.gather(
                    batch_logits,
                    dim=-1,
                    index=ca_input_ids.unsqueeze(-1)    # (B, N, dec_L, 1)
                ).squeeze(-1)   # (B, N, dec_L)
                
                # 2. Sum along the token_len masking (Compute p_c)
                _, _, T = indexed_logits.shape
                range_tensor = einops.rearrange(
                    torch.arange(T, device=indexed_logits.device),
                    'dec_L -> 1 1 dec_L'
                ) # (1, 1, dec_L)
                mask = range_tensor < token_len.unsqueeze(-1)
                log_logits = torch.log(indexed_logits)
                # Sum log probs with mask
                gen_scores = (log_logits * mask).sum(dim=-1)   # (B, N)
                length_penalty = torch.pow(candidate_len, self.config['length_factor']) # length penalty

                gen_scores = gen_scores / length_penalty
                gen_scores[torch.isnan(gen_scores)] = -float('inf')

                # 3. Compute pos penalty
                batch_doc_list = doc_list[B*i:B*(i+1)]
                batch_original_doc_id_list = original_doc_id_list[B*i:B*(i+1)]

                doc_len = torch.tensor(
                    [len(doc) - (self.en_temp_len + 2) for doc in batch_doc_list],
                    device=self.config['device']
                ).unsqueeze(-1) # (B, 1)
                pos_scores = pos / doc_len +  self.config['position_factor'] / torch.pow(doc_len, 3)

                batch_scores = gen_scores * pos_scores  # (B, N), negative log probs

                # 4. Compute IDF penalty
                batch_scores /= candidate_idfs  # (B, N)
                batch_scores[torch.isnan(batch_scores)] = -float('inf')
                
                # Find Top-k candidates
                k = min(10, batch_scores.size(-1))
                topk_scores, topk_indices = torch.topk(batch_scores, k=k, dim=-1)  # (B, top-k)
                topk_indices = topk_indices.cpu().numpy()

                matched_indices = np.arange(B)[:, None]
                topk_candidates = candidate_np[matched_indices, topk_indices]

                for j, doc in enumerate(batch_doc_list):
                    top_10_can = topk_candidates[j].tolist()
                    doc_with_top_10 = {"id": batch_original_doc_id_list[j], "doc": batch_doc_list[j], "keyphrases": top_10_can}
                    doc_with_candidates_list.append(doc_with_top_10)
        
        return doc_with_candidates_list
            
    def compute_prefix_len(self):
        en_input_ids = self.tokenizer(self.en_temp, return_tensors='pt')['input_ids']
        de_input_ids = self.tokenizer(self.de_temp, return_tensors='pt')['input_ids']
        self.en_temp_len = (en_input_ids[0] == self.tokenizer.eos_token_id).nonzero()[0].item()
        self.de_temp_len = (de_input_ids[0] == self.tokenizer.eos_token_id).nonzero()[0].item()


def idf_computation(passages, candidates):
    """
    Compute IDF for each candidate in the document list
    """
    if isinstance(candidates[0], list): # if candidates is a list of lists
        candidates = [c for sublist in candidates for c in sublist]
    candidates = list(set(candidates))

    vectorizer = TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 5),
        vocabulary=candidates,   # 주어진 candidates로 vocab 설정
        use_idf=True
    )

    vectorizer.fit(passages)

    feature_names = vectorizer.get_feature_names_out()
    idfs = vectorizer.idf_

    # 단어별 IDF 저장
    idf_dict = {term: idf_value for term, idf_value in zip(feature_names, idfs)}

    return idf_dict