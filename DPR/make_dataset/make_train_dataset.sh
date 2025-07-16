python make_train_dataset.py \
    --qas_path="/home/junho/KeyRAG/data/KISTI/hard_qas/qas_hard_rag.jsonl" \
    --passages_path="/home/junho/KeyRAG/data/KISTI/collection/passages.jsonl" \
    --index_path="/home/junho/KeyRAG/bm25/index" \
    --positive_passages=10 \
    --negative_passages=0 \
    --hard_negative_passages=50 \
    --train_path="/home/junho/KeyRAG/data/KISTI/dpr_dataset/data/kisti_train.json" \
    --dev_path="/home/junho/KeyRAG/data/KISTI/dpr_dataset/data/kisti_dev.json" \
    --test_path="/home/junho/KeyRAG/data/KISTI/dpr_dataset/data/kisti_test.json"
