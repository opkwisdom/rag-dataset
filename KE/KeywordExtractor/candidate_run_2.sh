#!/bin/bash

INFILE_PATH="../../IR_data/collection/passages_candidate.jsonl"
INTER_PATH="collection/passages_inter"
OUTFILE_PATH="collection/passages_candidate"
NUM_THREADS=8

# 0부터 NUM_THREADS까지 수를 매개변수로 하여 파이썬 파일을 동시에 실행
b=1

CURRENT_DATE=$(date +"%m-%d")
mkdir -p "collection"
mkdir -p "$CURRENT_DATE"


# 첫 번째 루프: 0 ~ 7
echo "Running candidate generator for files 0 to 7..."
for i in $(seq 0 7)
do
    python original/candidate_generator.py \
        --infile_path $INTER_PATH \
        --outfile_path $OUTFILE_PATH \
        --file_postfix $i &  # 백그라운드 실행
done

# 모든 백그라운드 작업이 끝날 때까지 기다림
wait

# 두 번째 루프: 8 ~ 15
echo "Running candidate generator for files 8 to 15..."
for i in $(seq 8 15)
do
    python candidate_generator.py \
        --infile_path $INTER_PATH \
        --outfile_path $OUTFILE_PATH \
        --file_postfix $i &  # 백그라운드 실행
done

# 모든 백그라운드 작업이 끝날 때까지 기다림
wait

echo "All tasks completed!"


# 입력 폴더와 출력 파일 설정
input_folder="collection"  # 예: './data'
output_file_prefix="$CURRENT_DATE/passages_candidate"

# 첫 번째 집합: 0 ~ 7
output_file1="${output_file_prefix}_1.jsonl"
echo "Merging files 0 to 7 into $output_file1..."
> "$output_file1"  # 기존 내용을 비우기

for i in {0..7}
do
    jsonl_file="$input_folder/passages_candidate_${i}.jsonl"
    if [ -f "$jsonl_file" ]; then
        cat "$jsonl_file" >> "$output_file1"
    fi
done

echo "Merged JSON Lines files into $output_file1"

# 두 번째 집합: 8 ~ 15
output_file2="${output_file_prefix}_2.jsonl"
echo "Merging files 8 to 15 into $output_file2..."
> "$output_file2"  # 기존 내용을 비우기

for i in {8..15}
do
    jsonl_file="$input_folder/passages_candidate_${i}.jsonl"
    if [ -f "$jsonl_file" ]; then
        cat "$jsonl_file" >> "$output_file2"
    fi
done

echo "Merged JSON Lines files into $output_file2"