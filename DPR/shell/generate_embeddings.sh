#!/bin/bash

BASE_DIR=/home/junho/KeyRAG/DPR
# Set the paths to your model, data, and output directory
MODEL_FILE="$BASE_DIR/trained_encoder_model/dpr_biencoder.1"    # Best model file
CTX_SRC="dpr_wiki"
NUM_SHARDS=10
OUT_FILE="$BASE_DIR/trained_encoder_model/embeddings"

# Run the generate_dense_embeddings.py script with the specified parameters
for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICE=0,1 python ../generate_dense_embeddings.py \
        model_file=${MODEL_FILE} \
        ctx_src=${CTX_SRC} \
        shard_id=${SHARD_ID} \
        num_shards=${NUM_SHARDS} \
        out_file=${OUT_FILE}
done

wait
echo "All shards completed."

