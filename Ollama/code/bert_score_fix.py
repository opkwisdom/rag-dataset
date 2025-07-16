import torch
import json
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np

class BertScore:
    def __init__(self, model_name="klue/bert-base"):
        self.model = AutoModel.from_pretrained(model_name).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def get_embeddings(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
        attention_mask = inputs["attention_mask"]
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state
        
        # CLS 토큰 제외
        embeddings = embeddings[:, 1:, :]
        attention_mask = attention_mask[:, 1:]
        
        return embeddings, attention_mask

    def cosine_similarity_matrix(self, ref_emb, pred_emb, ref_mask, pred_mask):
        # L2 정규화
        ref_norm = torch.nn.functional.normalize(ref_emb, p=2, dim=-1)
        pred_norm = torch.nn.functional.normalize(pred_emb, p=2, dim=-1)
        
        # 코사인 유사도 계산
        similarity_matrix = torch.bmm(ref_norm, pred_norm.transpose(-1, -2))
        
        # 마스킹 적용
        mask = torch.bmm(ref_mask.unsqueeze(-1).float(), pred_mask.unsqueeze(-2).float()).to("cuda")
        similarity_matrix = similarity_matrix * mask
        
        return similarity_matrix

    def score(self, refs, preds):
        # 배치 처리
        batch_size = len(refs)
        ref_emb, ref_mask = self.get_embeddings(refs)
        pred_emb, pred_mask = self.get_embeddings(preds)
        
        similarity_matrix = self.cosine_similarity_matrix(ref_emb, pred_emb, ref_mask, pred_mask)
        
        # 각 토큰에 대한 최대 유사도 계산
        ref_scores = similarity_matrix.max(dim=-1)[0]  # recall
        pred_scores = similarity_matrix.max(dim=-2)[0]  # precision
        
        # 마스크를 고려한 평균 계산
        ref_mask = ref_mask.float()
        pred_mask = pred_mask.float()
        
        recall = (ref_scores * ref_mask).sum(dim=-1) / ref_mask.sum(dim=-1)
        precision = (pred_scores * pred_mask).sum(dim=-1) / pred_mask.sum(dim=-1)
        
        # F1 score 계산
        f_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # CPU로 이동하고 numpy 배열로 변환
        recall = recall.cpu().numpy()
        precision = precision.cpu().numpy()
        f_score = f_score.cpu().numpy()
        
        return {
            "BERT-R": recall.tolist(),
            "BERT-P": precision.tolist(),
            "BERT-F": f_score.tolist()
        }

class Evaluator:
    def __init__(self, model_name="klue/bert-base"):
        self.bert_evaluator = BertScore(model_name=model_name)
    
    def evaluate(self, refs, preds):
        """
        Evaluate a batch of reference and prediction pairs using only BERTScore
        """
        return self.bert_evaluator.score(refs, preds)

def main():
    input_path = "../../IR_data/collection/qas_hard_ollama.jsonl"
    
    # 데이터 로드
    with open(input_path, 'r') as f:
        data = [json.loads(line.strip()) for line in f]
    
    evaluator = Evaluator(model_name="klue/roberta-large")
    results = []
    # 배치 처리
    batch_size = 32  # GPU 메모리에 따라 조정
    with tqdm(range(0, len(data), batch_size), desc="Evaluating") as pbar:
        for i in pbar:
            batch = data[i:min(i + batch_size, len(data))]
            refs = [item['answer'] for item in batch]
            preds = [item['llm_answer'] for item in batch]
            
            batch_results = evaluator.evaluate(refs, preds)
            results.extend([{
                "question": item['question'],
                "reference_answer": item['answer'],
                "model_answer": item['llm_answer'],
                "BERT": {
                    "R": r,
                    "P": p,
                    "F": f
                }
            } for item, r, p, f in zip(
                batch,
                batch_results["BERT-R"],
                batch_results["BERT-P"],
                batch_results["BERT-F"]
            )])
            
            pbar.update(1)
    
    # 결과 저장
    with open('bert_results_roberta_large.json', 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()