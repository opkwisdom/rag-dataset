import torch
import evaluate
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
import numpy as np

class Evaluator:
    def __init__(self):
        self.bert_evaluator = BertScore()
        self._load_packages()
    
    def evaluate(self, refs, preds):
        """
        Evaluate a batch of reference and prediction pairs
        """
        bert_score = self.bert_score(refs, preds)
        rouge_score = self.rouge(refs, preds)
        bleu_score = self.bleu(refs, preds)
        return {
            "bert": bert_score,
            "rouge": rouge_score,
            "bleu": bleu_score
        }

    def bert_score(self, refs, preds):
        return self.bert_evaluator.score(refs, preds)
    
    def rouge(self, refs, preds):
        scores = {
            "rouge1": [],
            "rouge2": [],
            "rougeL": []
        }
        for ref, pred in zip(refs, preds):
            score = self._rouge.compute(predictions=[pred], references=[ref],
                                        rouge_types = ["rouge1", "rouge2", "rougeL"])
            scores["rouge1"].append(score["rouge1"])
            scores["rouge2"].append(score["rouge2"])
            scores["rougeL"].append(score["rougeL"])
        return scores
    
    def bleu(self, refs, preds):
        scores = {
            'bleu': []
        }
        for ref, pred in zip(refs, preds):
            score = self._bleu.compute(predictions=[pred], references=[ref])
            scores['bleu'].append(np.array(score['precisions']).mean())
        return scores
    
    def em(self, refs, preds):
        raise NotImplementedError
    
    def _load_packages(self):
        self._rouge = evaluate.load('rouge', keep_in_memory=True, use_gpu=True)
        self._bleu = evaluate.load("bleu", keep_in_memory=True, use_gpu=True)

class BertScore:
    def __init__(self, model_name="klue/bert-base"):
        self.model = AutoModel.from_pretrained(model_name).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def get_embeddings(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
        attention_mask = inputs["attention_mask"][:, 1:-1]    # Exclude CLS and SEP tokens
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0)

        return embeddings[:, 1:-1], attention_mask    # Exclude CLS and SEP tokens
    
    def cosine_similarity_matrix(self, ref_emb, pred_emb, ref_mask, pred_mask):
        ref_norm = ref_emb / ref_emb.norm(dim=-1).unsqueeze(-1)
        pred_norm = pred_emb / pred_emb.norm(dim=-1).unsqueeze(-1)
        
        # Cosine similarity: matrix multiplication of normalized embeddings
        similarity_matrix = torch.matmul(ref_norm, pred_norm.transpose(-1, -2))
        
        # Masking
        ref_mask = ref_mask.unsqueeze(-1).float().to("cuda")
        pred_mask = pred_mask.unsqueeze(-2).float().to("cuda")
        mask = ref_mask * pred_mask
        similarity_matrix = similarity_matrix * mask

        return similarity_matrix    # (batch, n_ref, n_pred)
    
    def score(self, ref, pred):
        ref_emb, ref_mask = self.get_embeddings(ref)
        pred_emb, pred_mask = self.get_embeddings(pred)
        emb_matrix = self.cosine_similarity_matrix(ref_emb, pred_emb, ref_mask, pred_mask)   
        bert_sum_score = emb_matrix.max(dim=1).values.sum(dim=-1)    # (batch, n_ref, n_pred) -> (batch,)
        n_ref, n_pred = ref_emb.shape[1], pred_emb.shape[1]

        recall_score = bert_sum_score / n_ref
        precision_score = bert_sum_score / n_pred
        f_score = 2 * recall_score * precision_score / (recall_score + precision_score)

        recall_score = list(recall_score.cpu().numpy())
        precision_score = list(precision_score.cpu().numpy())
        f_score = list(f_score.cpu().numpy())

        return {
            "BERT-R": recall_score,
            "BERT-P": precision_score,
            "BERT-F": f_score
        }

def align_results(results, align_direction="Sample"):
    def flatten(key, subkey=None):
        """Flatten nested lists from results based on key and optional subkey."""
        if subkey:
            return [val for item in results for val in item[key][subkey]]
        return [item[key][subkey] for item in results]
    
    # Flatten results
    bert_r = flatten('bert', 'BERT-R')
    bert_p = flatten('bert', 'BERT-P')
    bert_f = flatten('bert', 'BERT-F')

    rouge1 = flatten('rouge', 'rouge1')
    rouge2 = flatten('rouge', 'rouge2')
    rougeL = flatten('rouge', 'rougeL')

    bleu = flatten('bleu', 'bleu')
    
    # Align results
    if align_direction == "Metric":
        summary = {
            "BERT": {"R": bert_r, "P": bert_p, "F": bert_f},
            "ROUGE": {"ROUGE-1": rouge1, "ROUGE-2": rouge2, "ROUGE-L": rougeL},
            "BLEU": bleu
        }
    elif align_direction == "Sample":
        sample_count = len(bert_r)
        summary = []
        for i in range(sample_count):
            sample_result = {
                "BERT": {"R": bert_r[i], "P": bert_p[i], "F": bert_f[i]},
                "ROUGE": {"ROUGE-1": rouge1[i], "ROUGE-2": rouge2[i], "ROUGE-L": rougeL[i]},
                "BLEU": bleu[i]
            }
            summary.append(sample_result)
    else:
        raise ValueError("align_direction must be either 'Metric' or 'Sample'.")
    
    return summary

def default_converter(o):
    if isinstance(o, np.float32):
        return float(o)
    raise TypeError(f"Type {type(o)} not serializable")

def main():
    input_path = "../../IR_data/collection/qas_hard_ollama.jsonl"
     # 파일 로드
    with open(input_path, 'r') as f:
        data = [json.loads(line.strip()) for line in f]
    
    evaluator = Evaluator()
    results = []

    # Batch 설정
    batch_size = 64
    with tqdm(data, desc="Evaluate Answer pairs") as pbar:
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            refs = [item['answer'] for item in batch]
            preds = [item['llm_answer'] for item in batch]
            
            batch_results = evaluator.evaluate(refs, preds)
            results.append(batch_results)

            pbar.update(len(batch))

    # 결과 정렬
    aligned_results = align_results(results)
    
    with open('evaluated_results.json', 'w') as f:
        json.dump(aligned_results, f, indent=4, ensure_ascii=False, default=default_converter)


if __name__ == "__main__":
    main()