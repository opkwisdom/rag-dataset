FILE_PATH="../data/ir_data_SM_raw.jsonl 
../data/ir_data_SM_filter.jsonl 
../data/ir_data_SM_final.jsonl 
../data/ir_data_SM_final_2.jsonl 
../data/ir_data_SM_final_3.json"
FILE_TYPE="Raw Filter Processed Version2 Version3"

python3 ../code/ir_data_stat.py \
    --file_path $FILE_PATH \
    --file_type $FILE_TYPE