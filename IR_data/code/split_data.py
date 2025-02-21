from tqdm import tqdm
import jsonlines
import sys
from argparse import ArgumentParser

def split_list(data, num_chunks):
    """데이터 리스트를 num_chunks만큼 균등하게 분할"""
    avg_len = len(data) // num_chunks
    return [data[i * avg_len:(i + 1) * avg_len]
            for i in range(num_chunks - 1)] + [data[(num_chunks - 1) * avg_len:]]

def main():
    parser = ArgumentParser()
    parser.add_argument('--infile_path', default="./data/ir_data_v1.jsonl")
    parser.add_argument('--outfile_path', default="./collection/question")
    parser.add_argument('--num_files', default=32, type=int)
    args = parser.parse_args()
    
    with jsonlines.open(args.infile_path, "r") as jsonl_reader:
        json_data_list = [json_data for json_data in tqdm(jsonl_reader, desc="Reading Input")]
    
    chunks = split_list(json_data_list, args.num_files)
    
    # Write each chunk to a separate file
    for i, chunk in enumerate(chunks):
        output_file = f"{args.outfile_path}_{i}.jsonl"
        with jsonlines.open(output_file, mode="w") as jsonl_writer:
            jsonl_writer.write_all(chunk)
    print(f"Saved chunks")

if __name__ == "__main__":
    main()
    
    
