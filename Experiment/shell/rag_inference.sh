#!/bin/bash

python code/rag_inference.py \
    --prompt_type "keyphrase_only" \
    --num_samples -1 \
    --use_quantization \
    --batch_size 64