IN_PATH="../data/ir_data_positive.jsonl"
MIN_RELEVANT_PASSAGES=3
MAX_RELEVANT_PASSAGES=200
OUT_PATH="../data/ir_data_positive_filter_"$MIN_RELEVANT_PASSAGES".jsonl"

python3 ../code/filter.py \
    --in_path=$IN_PATH \
    --out_path=$OUT_PATH \
    --min_relevant_passages=$MIN_RELEVANT_PASSAGES \
    --max_relevant_passages=$MAX_RELEVANT_PASSAGES