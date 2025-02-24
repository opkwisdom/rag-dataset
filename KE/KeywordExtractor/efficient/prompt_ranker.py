from tqdm import tqdm
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.nn.utils.rnn import pad_sequence
import einops

class CachedPromptRanker:
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
        
        for i, (docs, cans) in enumerate(tqdm(dataloader, desc="Evaluating ")):
            # Unpack data
            en_input_ids, en_input_mask, de_input_ids = zip(*docs)
            candidate, ca_input_ids, token_len, pos = zip(*cans)
            
            en_input_ids = torch.cat(en_input_ids)
            en_input_mask = torch.cat(en_input_mask)
            de_input_ids = torch.stack(de_input_ids)
            
            candidate = list(candidate)
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
            token_len = token_len.to(self.config['device'])
            pos = pos.to(self.config['device'])
            
            batch_size = ca_input_ids.shape[0]
            n_candidates = ca_input_ids.shape[1]
            # Candidate padding
            padded_candidates = [row + [''] * (n_candidates - len(row)) for row in candidate]
            candidate_np = np.array(padded_candidates)
            
            # Prefix 출력 계산
            with torch.no_grad():
                prefix_outputs = self.model(
                    input_ids=en_input_ids,
                    attention_mask=en_input_mask,
                    decoder_input_ids=de_input_ids,
                    use_cache=True
                )
                past_key_values = prefix_outputs.past_key_values
                encoder_outputs = prefix_outputs.encoder_last_hidden_state   # (B, enc_L, D)
            
            # Candidate 출력 계산
            with torch.no_grad():
                # 차원 맞추기
                # Memory size in case of batch 4
                # 1. flattend_ca_input_ids: 0.035MB (8bytes)
                # 2. expanded_encoder_outputs: 234MB (4bytes)
                # 3. expanded_past_key_values: 5.6GB (4bytes)
                B, N, L = ca_input_ids.shape
                flattend_ca_input_ids = einops.rearrange(ca_input_ids, 'b n l -> (b n) l')
                expanded_encoder_outputs = einops.repeat(
                        encoder_outputs,  # (B, enc_L, D)
                        "b enc_l d -> 1 (b n) enc_l d", # (1, B*N, enc_L, D)
                        n=N  # Candidate 개수만큼 복제
                    )
                # T5 Dec_k, Dec_v, Enc_k, Enc_v 복제
                expanded_past_key_values = tuple(
                    (einops.repeat(dec_k, "b h dec_l d -> (b n) h dec_l d", n=N),
                    einops.repeat(dec_v, "b h dec_l d -> (b n) h dec_l d", n=N),
                    einops.repeat(enc_k, "b h enc_l d -> (b n) h enc_l d", n=N),
                    einops.repeat(enc_v, "b h enc_l d -> (b n) h enc_l d", n=N),)
                    for dec_k, dec_v, enc_k, enc_v in past_key_values
                ) # (# of layers, 4, B*N, H, L, D)

                output = self.model(
                            encoder_outputs=expanded_encoder_outputs,  # Encoder 결과 재사용
                            decoder_input_ids=flattend_ca_input_ids,  # Candidate 입력
                            past_key_values=expanded_past_key_values,  # 캐싱된 Key-Value 사용
                            use_cache=True
                        )[0]  # logits of Seq2SeqLMOutput
                batch_logits = einops.rearrange(
                    output,
                    '(b n) dec_l v -> b n dec_l v',
                    b=B, n=N
                ) # (B*N, dec_L, V) -> (B, N, dec_L, V)
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
                # Sum log probs
                gen_scores = (log_logits * mask).sum(dim=-1)   # (B, N)
                length_penalty = torch.pow(token_len, self.config['length_factor']) # length penalty

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

                batch_scores = gen_scores * pos_scores
                
                # Find Top-k candidates
                k = min(15, batch_scores.size(-1))
                topk_scores, topk_indices = torch.topk(batch_scores, k=15, dim=-1)  # (B, top-k)
                topk_indices = topk_indices.cpu().numpy()

                matched_indices = np.arange(B)[:, None]
                topk_candidates = candidate_np[matched_indices, topk_indices]

                for j, doc in enumerate(batch_doc_list):
                    top_15_can = topk_candidates[j].tolist()
                    doc_with_top_15 = {"id": batch_original_doc_id_list[j], "doc": batch_doc_list[j], "keyphrases": top_15_can}
                    doc_with_candidates_list.append(doc_with_top_15)
        
        return doc_with_candidates_list
            
    
    def compute_prefix_len(self):
        en_input_ids = self.tokenizer(self.en_temp, return_tensors='pt')['input_ids']
        de_input_ids = self.tokenizer(self.de_temp, return_tensors='pt')['input_ids']
        self.en_temp_len = (en_input_ids[0] == self.tokenizer.eos_token_id).nonzero()[0].item()
        self.de_temp_len = (de_input_ids[0] == self.tokenizer.eos_token_id).nonzero()[0].item()