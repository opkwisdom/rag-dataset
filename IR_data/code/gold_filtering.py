from tqdm import tqdm
import jsonlines
from pyserini.search.lucene import LuceneSearcher
import json

filepath = "../../data/qas.jsonl"
index_path = "../../bm25/index_key"
out_path = "../data/ir_data_v1.jsonl"
searcher = LuceneSearcher(index_path)

all_data = []
with jsonlines.open(filepath, "r") as json_file:
    for json_data in tqdm(json_file.iter(), desc="Processing "):
        doc_id = json_data.get('gold_id')
        if doc_id is not None:
            raw_document = searcher.doc(doc_id).raw()
            document_json = json.loads(raw_document)
            data = {
                "query_id": json_data["id"],
                "query": json_data["question"],
                "doc_id": json_data["gold_id"],
                "document": document_json["contents"],
                "rel": 1
            }
            all_data.append(data)

with open(out_path, encoding="utf-8", mode="w") as writer:
    for i in all_data:
        writer.write(json.dumps(i, ensure_ascii=False) + '\n')

print(f"Data successfully saved to {out_path}: length={len(all_data)}")