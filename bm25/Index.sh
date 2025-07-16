# !/bin/bash

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /home/junho/KeyRAG/bm25/collection\
  --index /home/junho/KeyRAG/bm25/index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 64 \
  --storePositions --storeDocvectors --storeRaw