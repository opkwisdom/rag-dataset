MODEL_PATH="klue/roberta-base"
TOP_K=10
INFILE_PATH="../data/ir_data_positive_filter_3.jsonl"
OUTFILE_PATH="../data/ir_rag_data_"$TOP_K".jsonl"

python3 ../code/scoring.py \
    --model_path=$MODEL_PATH \
    --top_k=$TOP_K \
    --infile_path=$INFILE_PATH \
    --outfile_path=$OUTFILE_PATH
