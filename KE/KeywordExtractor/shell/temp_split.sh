INFILE_PATH="collection/passages_10.jsonl"
OUTFILE_PATH="collection/passages_2"
NUM_THREADS=16

python split_data.py \
    --infile_path $INFILE_PATH \
    --outfile_path $OUTFILE_PATH \
    --num_files $NUM_THREADS