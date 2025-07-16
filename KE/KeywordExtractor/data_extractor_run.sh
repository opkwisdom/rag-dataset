# Dataset name: Inspec, SemEval2010, SemEval2017, DUC2001, nus, krapivin

dataset_name=passages_candidate_2.jsonl
batch_size=16
log_dir=log
log_filename=efficient_p2_data
output_dir=data/kpe_dataset_2

python3 efficient/data_extractor_main.py \
    --dataset_dir ./data/$dataset_name \
    --batch_size $batch_size \
    --log_dir $log_dir \
    --log_filename $log_filename \
    --output_dir $output_dir

echo Run PromptRank finish