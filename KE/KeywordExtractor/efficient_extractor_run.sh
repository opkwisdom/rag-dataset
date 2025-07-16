# Dataset name: Inspec, SemEval2010, SemEval2017, DUC2001, nus, krapivin

dataset_name=error_fix.jsonl
batch_size=1
log_dir=log
log_filename=error_fix_idf_efficient_rag
output_dir=efficient/output/error_fix.json

CUDA_VISIBLE_DEVICES=0 python3 efficient/main.py \
    --dataset_dir ./data/$dataset_name \
    --batch_size $batch_size \
    --log_dir $log_dir \
    --log_filename $log_filename \
    --output_dir $output_dir

echo Run PromptRank finish