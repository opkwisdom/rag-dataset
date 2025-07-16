INTER_PATH="collection/passages"
OUTFILE_PATH="collection/passages_candidate"
NUM_THREADS=0_0
i=0_0

python candidate_generator.py \
    --infile_path $INTER_PATH \
    --outfile_path $OUTFILE_PATH \
    --file_postfix $i  # & 기호를 사용해 백그라운드에서 병렬 실행