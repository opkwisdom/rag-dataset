 # !/bin/bash

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /home/sangwon/nlplab_leaderboard2/bm25/collection\
  --index /home/sangwon/nlplab_leaderboard2/bm25/index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 64 \
  --storePositions --storeDocvectors --storeRaw