# Dataset name: Inspec, SemEval2010, SemEval2017, DUC2001, nus, krapivin

dataset_name=tiny_candidate.jsonl
batch_size=8
log_dir=log
log_filename=efficient_tiny
output_dir=efficient/output/tiny_keyphrases.json

python3 efficient/main.py \
    --dataset_dir ./data/$dataset_name \
    --batch_size $batch_size \
    --log_dir $log_dir \
    --log_filename $log_filename \
    --output_dir $output_dir

echo Run PromptRank finish