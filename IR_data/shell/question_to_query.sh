#!/bin/bash

INFILE_PATH="../data/ir_data_positive_filter.jsonl"
INTER_PATH="../collection/question"
OUTFILE_PATH="../collection/query"
MODEL_FILE="klue/bert-base"
NUM_THREADS=16

# 0부터 NUM_THREADS까지 수를 매개변수로 하여 파이썬 파일을 동시에 실행
b=1

CURRENT_DATE=$(date +"%m-%d")

mkdir "../collection"

python ../code/split_data.py \
    --infile_path $INFILE_PATH \
    --outfile_path $INTER_PATH \
    --num_files $NUM_THREADS

NUM=$(($NUM_THREADS - $b))
# 입력 폴더와 출력 파일 설정

mkdir ../"$CURRENT_DATE"

for i in $(seq 0 $NUM)
do
    python ../code/question_to_query.py \
        --infile_path $INTER_PATH \
        --outfile_path $OUTFILE_PATH \
        --model_file $MODEL_FILE \
        --file_postfix $i &  # & 기호를 사용해 백그라운드에서 병렬 실행
done

# 모든 백그라운드 작업이 끝날 때까지 대기
wait
echo "All scripts have finished executing."



# 입력 폴더와 출력 파일 설정
input_folder="../collection"  # 예: './data'
output_file=../"$CURRENT_DATE/query_positive.jsonl"

# 출력 파일이 이미 존재하는 경우 삭제
if [ -f "$output_file" ]; then
    rm "$output_file"
fi

# 폴더 내에 있는 모든 query로 시작하는 jsonl 파일을 찾습니다.
for jsonl_file in "$input_folder"/query*.jsonl
do
    if [ -f "$jsonl_file" ]; then
        cat "$jsonl_file" >> "$output_file"
    fi
done


echo "Merged JSON Lines files into $output_file"

rm -rf "../collection"