import glob
import json
import logging
import pickle
import time
import zlib
from typing import List, Tuple, Dict, Iterator
import hydra

from data_old.qa_validation import calculate_matches

from omegaconf import DictConfig, OmegaConf
from pyserini.search.lucene import LuceneSearcher
import json
import sys
import os
from bm25.options import setup_logger, setup_cfg_gpu, set_cfg_params_from_state

logger = logging.getLogger()
setup_logger(logger)

def validate(
    passages: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> List[List[bool]]:
    logger.info("validating passages. size=%d", len(passages))
    match_stats = calculate_matches(passages, answers, result_ctx_ids, workers_num, match_type)
    top_k_hits = match_stats.top_k_hits

    logger.info("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info("Validation results: top k documents hits accuracy %s", top_k_hits)
    
    # ---------- New metrics ----------
    # MRR@K
    mrr_k_hits = match_stats.mrr_k_hits
    logger.info("Validation results: MRR@k documents hits %s", mrr_k_hits)
    
    # Recall@K
    recall_k_hits = match_stats.recall_k_hits
    logger.info("Validation results: Recall@k documents hits %s", recall_k_hits)
    
    # NDCG@K
    ndcg_k_hits = match_stats.ndcg_k_hits
    logger.info("Validation results: NDCG@k documents hits %s", ndcg_k_hits)
    
    return match_stats.questions_doc_hits


def save_results(
    passages: Dict[object, Tuple[str, str]],
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str,
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)
        
        logger.info("%s", results_and_scores)

        # qa_validation.py > check_answer function과 동일한 이유
        # Log: /home/nlplab13/etri/ETRI/bm25/shell/outputs/2024-09-12/18-21-41:408L
        if len(results_and_scores[0]) == 0:
            results_item = {
                "question": q,
                "answers": q_answers,
                "ctxs": [
                    {
                        "id": None,
                        "title": None,
                        "text": None,
                        "score": None,
                        "has_answer": None,
                    }
                ],
            }
        else:
            results_item = {
                "question": q,
                "answers": q_answers,
                "ctxs": [
                    {
                        "id": results_and_scores[0][c],
                        "title": docs[c][1],
                        "text": docs[c][0],
                        "score": scores[c],
                        "has_answer": hits[c],
                    }
                    for c in range(ctxs_num)
                ],
            }

        merged_data.append(results_item)

    with open(out_file, "w", encoding="utf-8") as writer:  # 인코딩을 utf-8로 설정
        writer.write(json.dumps(merged_data, indent=4, ensure_ascii=False) + "\n")  # ensure_ascii=False 추가
    logger.info("Saved results * scores  to %s", out_file)
    
    
def get_all_passages(ctx_sources):
    all_passages = {}
    for ctx_src in ctx_sources:
        ctx_src.load_data_to(all_passages)
        logger.info("Loaded ctx data: %d", len(all_passages))

    if len(all_passages) == 0:
        raise RuntimeError("No passages data found. Please specify ctx_file param properly.")
    return all_passages


@hydra.main(config_path="conf", config_name="bm25")
def main(cfg: DictConfig):
    logger.info("CFG (after gpu  configuration):")
    logger.info("%s", OmegaConf.to_yaml(cfg))
    
    # Set-up LuceneSearcher
    if not cfg.index_path:
        logger.warning("Please specify index_path to use")
        return
    
    searcher = LuceneSearcher(cfg.index_path)
    
    # get questions & answers
    questions = []
    questions_text = []
    question_answers = []

    if not cfg.qa_dataset:
        logger.warning("Please specify qa_dataset to use")
        return

    ds_key = cfg.qa_dataset
    logger.info("qa_dataset: %s", ds_key)
    qa_src = hydra.utils.instantiate(cfg.datasets[ds_key])
    qa_src.load_data()

    total_queries = len(qa_src)
    for i in range(total_queries):
        qa_sample = qa_src[i]
        question, answers = qa_sample.query, qa_sample.answers
        questions.append(question)
        question_answers.append(answers)

    logger.info("questions len %d", len(questions))
    logger.info("question_answers len %d", len(question_answers))
    logger.info("questions_text len %d", len(questions_text))
    
    # send data for indexing
    id_prefixes = []
    ctx_sources = []
    for ctx_src in cfg.ctx_datatsets:
        ctx_src = hydra.utils.instantiate(cfg.ctx_sources[ctx_src])
        id_prefixes.append(ctx_src.id_prefix)
        ctx_sources.append(ctx_src)
        logger.info("ctx_sources: %s", type(ctx_src))

    logger.info("id_prefixes per dataset: %s", id_prefixes)

    all_passages = get_all_passages(ctx_sources)
    
    # get top k results
    top_results_and_scores = []
    for question in questions:
        hits_per_question = searcher.search(question, k=cfg.n_docs)
        result_docids = [ctx_src.id_prefix + hit.docid for hit in hits_per_question]
        result_scores = [hit.score for hit in hits_per_question]
        top_results_and_scores.append([result_docids, result_scores])
    
    questions_doc_hits = validate(
        all_passages,
        question_answers,
        top_results_and_scores,
        cfg.validation_workers,
        cfg.match,
    )

    if cfg.out_file:
        save_results(
            all_passages,
            questions_text if questions_text else questions,
            question_answers,
            top_results_and_scores,
            questions_doc_hits,
            cfg.out_file,
        )


if __name__ == "__main__":
    main()