# import
import re
import ast
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
import argparse
from tqdm import tqdm
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--prompt_type', type=str, default='context_keyphrase', 
                   choices=['context_only', 'keyphrase_only', 'context_keyphrase', 'keysentence', 'query_only'],
                   help='Type of prompt to use: context_only, keyphrase_only, or context_keyphrase')
parser.add_argument('--save_dir', type=str, default='RAG')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
parser.add_argument('--checkpoint_interval', type=int, default=10000, help='Save checkpoint every N samples')
parser.add_argument('--device_map', type=str, default='balanced', help='Device mapping strategy')
parser.add_argument('--use_quantization', action='store_true', help='Use quantization for model loading')
parser.add_argument('--use_flash_attn', action='store_true', help='Use Flash Attention if available')
parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for prompt creation')
parser.add_argument('--use_compile', action='store_true', help='Use torch.compile for model acceleration')
parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to process (default: 10)')

args = parser.parse_args()

# Create directory
os.makedirs(args.save_dir, exist_ok=True)

# Load JSON data
print("Loading JSON dataset...")
with open('/home/junho/KeyRAG/data/KISTI/idf_cand/qas_idf_candidate.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)


# Convert to list format and take only first N samples
if args.num_samples == -1:
    print("Processing ALL samples from dataset...")
    if isinstance(json_data, dict):
        data_list = []
        for key, value in json_data.items():  # 전체 데이터
            if isinstance(value, dict):
                value['id'] = key
                data_list.append(value)
            else:
                data_list.append({'id': key, 'data': value})
    elif isinstance(json_data, list):
        data_list = json_data  # 전체 리스트
    else:
        raise ValueError("Unexpected JSON data format")
else:
    print(f"Taking first {args.num_samples} samples from dataset...")
    if isinstance(json_data, dict):
        data_list = []
        for key, value in list(json_data.items())[:args.num_samples]:
            if isinstance(value, dict):
                value['id'] = key
                data_list.append(value)
            else:
                data_list.append({'id': key, 'data': value})
    elif isinstance(json_data, list):
        data_list = json_data[:args.num_samples]
    else:
        raise ValueError("Unexpected JSON data format")

# Convert to DataFrame
data = pd.DataFrame(data_list)

print(f"Loaded {len(data)} samples")
print("Data columns:", data.columns.tolist())
print("\nSample data:")
print(data.head(2))

def make_prompt_keyphrase_only(row):
    """Create prompt using only keyphrases and question"""
    question = row.get("question", "")
    keyphrases = row.get("keyphrases", [])
    
    # Handle case where there might be no keyphrases
    if not keyphrases:
        keyphrases_text = "주어진 핵심어가 없습니다."
    else:
        # Join keyphrases with comma
        keyphrases_text = " | ".join(keyphrases)
    
    return f"""당신은 주어진 핵심어들을 활용하여 질문에 대한 완전하고 상세한 답변을 제공하는 전문가입니다.

            <지침>
            1. 주어진 핵심어들을 모두 활용하여 질문에 답하세요.
            2. 핵심어들 간의 관계와 연결점을 파악하여 논리적으로 설명하세요.
            3. 단순한 정의뿐만 아니라 특징, 구성요소, 목적, 기능 등을 포함한 포괄적인 설명을 제공하세요.
            4. 2-3문장으로 구성된 완전한 답변을 작성하세요.
            5. 정중하고 공손한 어조로 "~습니다", "~됩니다" 형태의 존댓말을 사용하세요.
            6. 핵심어에 없는 정보는 추가하지 마세요.

            핵심어: {keyphrases_text}
            질문: {question}

            답변:"""

def make_prompt_key_sentence(row):
    """Create prompt using only keyphrases and question"""
    question = row.get("question", "")
    keysentence = row.get("extracted_sentences", [])
    
    # Handle case where there might be no keyphrases
    if not keysentence:
        keysentence_text = "주어진 핵심문장이 없습니다."
    else:
        # Join keyphrases with comma
        keysentence_text = " | ".join(keysentence)
    
    return f"""당신은 주어진 핵심문장을 활용하여 질문에 대한 완전하고 상세한 답변을 제공하는 전문가입니다.

            <지침>
            1. 주어진 핵심문장을 모두 활용하여 질문에 답하세요.
            2. 핵심문장 간의 관계와 연결점을 파악하여 논리적으로 설명하세요.
            3. 단순한 정의뿐만 아니라 특징, 구성요소, 목적, 기능 등을 포함한 포괄적인 설명을 제공하세요.
            4. 2-3문장으로 구성된 완전한 답변을 작성하세요.
            5. 정중하고 공손한 어조로 "~습니다", "~됩니다" 형태의 존댓말을 사용하세요.
            6. 핵심문장에 없는 정보는 추가하지 마세요.

            질문: {question}
            핵심문장: {keysentence_text}
            
            답변:"""

def make_prompt_context_keyphrase(row):
    """Create prompt using context, keyphrases and question"""
    context = row.get("context", "")
    question = row.get("question", "")
    keyphrases = row.get("keyphrases", [])
    
    # Handle missing data
    if not context:
        context = "주어진 문서가 없습니다."
    if not keyphrases:
        keyphrases_text = "주어진 핵심어가 없습니다."
    else:
        keyphrases_text = ", ".join(keyphrases)
    
    return f"""당신은 문서와 핵심어를 종합적으로 분석하여 질문에 대한 완전하고 상세한 답변을 제공하는 전문가입니다.

            <지침>
            1. 질문과 관련된 모든 정보를 문서에서 찾아 통합하세요.
            2. 단순한 정의뿐만 아니라 특징, 구성요소, 목적, 기능 등을 포함한 포괄적인 설명을 제공하세요.
            3. 문서의 맥락과 세부사항을 활용하여 풍부하고 완전한 답변을 작성하세요.
            4. 질문의 핵심에 답하되, 관련된 중요한 정보도 함께 제공하세요.
            5. 문서를 참고했다는 표현은 절대 사용하지 마세요.

            문서: {context}
            질문: {question}
            
            핵심어는 문서의 중요한 부분이 담겨있습니다. 아래 핵심어를 통해 답변을 강화하세요.
            핵심어: {keyphrases_text}

            정중하고 공손한 어조로 "~습니다", "~됩니다" 형태의 존댓말을 사용하세요.
            2-3문장으로 구성된 완전한 답변을 작성하세요.
            상세한 답변:"""

def make_prompt(row):
    """Create prompt for QAS task using context and question"""
    context = row.get("context", "")
    question = row.get("question", "")
    
    # Handle case where there might be no context
    if not context:
        context = "주어진 정보가 없습니다."

    # 쿼리 + context
    return f"""당신은 문서를 종합적으로 분석하여 질문에 대한 완전하고 상세한 답변을 제공하는 전문가입니다.

                <지침>
                1. 질문과 관련된 모든 정보를 문서에서 찾아 통합하세요.
                2. 단순한 정의뿐만 아니라 특징, 구성요소, 목적, 기능 등을 포함한 포괄적인 설명을 제공하세요.
                3. 문서의 맥락과 세부사항을 활용하여 풍부하고 완전한 답변을 작성하세요.
                4. 질문의 핵심에 답하되, 관련된 중요한 정보도 함께 제공하세요.
                5. 문서를 참고했다는 표현은 절대 사용하지 마세요.

                문서: {context}
                질문: {question}

                정중하고 공손한 어조로 "~습니다", "~됩니다" 형태의 존댓말을 사용하세요.
                2-3문장으로 구성된 완전한 답변을 작성하세요.
                상세한 답변:"""

def make_prompt_query_only(row):
    """Create prompt for QAS task using context and question"""
    question = row.get("question", "")
    

    # 쿼리 + context
    return f"""당신은 질문에 대한 완전하고 상세한 답변을 제공하는 전문가입니다.

                질문: {question}

                정중하고 공손한 어조로 "~습니다", "~됩니다" 형태의 존댓말을 사용하세요.
                2-3문장으로 구성된 완전한 답변을 작성하세요.
                상세한 답변:"""

print("Preprocessing prompts...")
# Parallel prompt creation
print(f"Using prompt type: {args.prompt_type}")

# Parallel prompt creation with specified prompt type
with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
    if args.prompt_type == "keyphrase_only":
        prompt_futures = {executor.submit(make_prompt_keyphrase_only, row): i for i, row in data.iterrows()}
    if args.prompt_type == "context_keyphrase":
        prompt_futures = {executor.submit(make_prompt_context_keyphrase, row): i for i, row in data.iterrows()}
    if args.prompt_type == "keysentence":
        prompt_futures = {executor.submit(make_prompt_key_sentence, row): i for i, row in data.iterrows()}
    if args.prompt_type == "context_only":
        prompt_futures = {executor.submit(make_prompt, row): i for i, row in data.iterrows()}
    if args.prompt_type == "query_only":
        prompt_futures = {executor.submit(make_prompt_query_only, row): i for i, row in data.iterrows()}
    
    
    
    all_prompts = []
    for i in tqdm(range(len(data)), desc="Creating prompts"):
        all_prompts.append(None)  # Placeholder
    
    for future in as_completed(prompt_futures):
        idx = prompt_futures[future]
        try:
            prompt = future.result()
            all_prompts[idx] = prompt
        except Exception as e:
            print(f"Error creating prompt for row {idx}: {e}")
            all_prompts[idx] = "Error in prompt creation"


# Add prompts to dataframe for easy access
data['prompt'] = all_prompts

# Model configuration
print("Loading model and tokenizer...")
model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Additional model loading arguments
model_kwargs = {
    "device_map": args.device_map,
    "use_auth_token": True,
}

if args.use_quantization:
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"  # Use nf4 for better accuracy/performance trade-off
    )
    # Set torch to run with tensor cores for faster matrix multiplication
    torch.set_float32_matmul_precision('high')

    model_kwargs["quantization_config"] = bnb_config
    model_kwargs["torch_dtype"] = torch.float16  # Use half precision throughout

# Add Flash Attention if specified
if args.use_flash_attn:
    model_kwargs["use_flash_attention_2"] = True

# Load tokenizer with left padding
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Fix padding direction - THIS IS THE KEY CHANGE
tokenizer.padding_side = "left"
# Fix for padding token missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

# Use torch.compile if available and requested (requires PyTorch 2.0+)
if args.use_compile and hasattr(torch, 'compile'):
    try:
        print("Compiling model for better performance...")
        model = torch.compile(model)
        print("Model compiled successfully!")
    except Exception as e:
        print(f"Model compilation failed: {e}")
        print("Continuing with standard model.")

# Function to extract answer from model output
def extract_answer(text):
    """Extract answer from model output, stopping at the last period or first line break"""
    # Look for answer after "답변:" marker
    if "답변:" in text:
        raw_answer = text.split("답변:")[-1].strip()
    else:
        raw_answer = text.strip()

    raw_answer = raw_answer.replace('|', '')
    
    # Find the first line break
    first_newline_idx = raw_answer.find('\n')
    if first_newline_idx == -1:
        first_newline_idx = raw_answer.find('\r')
    
    # If there's a line break, extract only the first line
    if first_newline_idx != -1:
        first_line = raw_answer[:first_newline_idx].strip()
    else:
        first_line = raw_answer.strip()
    
    # Now find the last period in the first line
    last_period_idx = first_line.rfind('.')
    
    if last_period_idx != -1:
        # Extract up to and including the last period in the first line
        answer = first_line[:last_period_idx + 1].strip()
    else:
        # If no period found in first line, use the whole first line
        answer = first_line
    
    return raw_answer, answer

# Optimize tokenizer for batch processing
@torch.no_grad()
def batch_tokenize(batch_texts):
    """Tokenize a batch of texts efficiently with warning for overly long inputs"""
    max_length = 3000  # Increase for longer contexts
    
    # Pre-tokenize with only encoding to check lengths
    encodings = tokenizer(batch_texts, add_special_tokens=True, truncation=False)
    
    for i, input_ids in enumerate(encodings["input_ids"]):
            length = len(input_ids)
            if length > max_length:
                overflow = length - max_length
                print(f"⚠️ Warning: Text {i} is {overflow} tokens too long ({length} > {max_length}). It will be truncated.")
    
    # Final tokenization with truncation and padding
    return tokenizer(
        batch_texts,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(model.device)

@torch.no_grad()
def process_batch(batch_prompts):
    """Process a batch of prompts with optimized inference"""
    # Tokenize batch
    inputs = batch_tokenize(batch_prompts)
    
    # Generate with optimized settings for longer, more comprehensive answers
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,  # Increase for longer, more detailed answers
        do_sample=True,
        temperature=0.2,     # Lower for more focused answers
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,  # Enable KV caching
    )
    
    # Process results
    results = []
    for output in outputs:
        result_text = tokenizer.decode(output, skip_special_tokens=True)
        raw_answer, answer = extract_answer(result_text)
        results.append((raw_answer, answer))
    
    return results

def predict_with_dynamic_batching(df):
    """Process data with dynamic batch sizing and performance monitoring"""
    all_results = []
    total = len(df)
    
    # Start with batch size from args
    current_batch_size = args.batch_size
    
    # Process in dynamically-sized batches
    start_idx = 0
    
    with tqdm(total=total, desc="Processing") as pbar:
        while start_idx < total:
            end_idx = min(start_idx + current_batch_size, total)
            batch = df.iloc[start_idx:end_idx]
            batch_size = end_idx - start_idx
            
            # Get pre-computed prompts
            batch_prompts = batch['prompt'].tolist()
            
            try:
                # Measure batch processing time
                start_time = time.time()
                
                # Process batch
                batch_results = process_batch(batch_prompts)
                
                # Calculate time per sample
                end_time = time.time()
                time_per_sample = (end_time - start_time) / batch_size
                
                # Adjust batch size dynamically based on performance
                if time_per_sample < 0.5 and current_batch_size < 32:  # Increase if fast
                    current_batch_size = min(current_batch_size + 2, 32)
                elif time_per_sample > 2.0 and current_batch_size > 2:  # Decrease if slow
                    current_batch_size = max(current_batch_size - 2, 16)
                
                # Store results
                for i, (raw_output, answer) in enumerate(batch_results):
                    row_idx = start_idx + i
                    row_data = df.iloc[row_idx]
                    all_results.append({
                        "id": row_data.get("id", f"sample_{row_idx}"),
                        "question": row_data.get("question", ""),
                        "context": row_data.get("context", ""),
                        "gold_answer": row_data.get("answer", ""),
                        "predicted_answer": answer,
                        "raw_input": batch_prompts[i],
                        "raw_output": raw_output
                    })
                
                # Update progress
                pbar.update(batch_size)
                pbar.set_postfix(batch_size=current_batch_size, time_per_sample=f"{time_per_sample:.2f}s")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Reduce batch size on OOM error
                    print(f"\nOOM error with batch size {current_batch_size}. Reducing batch size.")
                    torch.cuda.empty_cache()
                    gc.collect()
                    current_batch_size = max(current_batch_size // 2, 1)
                    continue  # Retry with smaller batch
                else:
                    print(f"\nError processing batch: {e}")
                    # Fall back to processing individual rows
                    for i, row in batch.iterrows():
                        prompt = row['prompt']
                        try:
                            # Process single prompt
                            single_result = process_batch([prompt])[0]
                            all_results.append({
                                "id": row.get("id", f"sample_{i}"),
                                "question": row.get("question", ""),
                                "context": row.get("context", ""),
                                "gold_answer": row.get("answer", ""),
                                "predicted_answer": single_result[1],
                                "raw_input": prompt,
                                "raw_output": single_result[0]
                            })
                        except Exception as inner_e:
                            print(f"Error processing row {i}: {inner_e}")
                            all_results.append({
                                "id": row.get("id", f"sample_{i}"),
                                "question": row.get("question", ""),
                                "context": row.get("context", ""),
                                "gold_answer": row.get("answer", ""),
                                "predicted_answer": None,
                                "raw_input": prompt,
                                "raw_output": f"Error: {str(inner_e)}"
                            })
                        pbar.update(1)
            
            # Save checkpoint if needed
            if (start_idx // args.checkpoint_interval) != (end_idx // args.checkpoint_interval):
                checkpoint_idx = (end_idx // args.checkpoint_interval) * args.checkpoint_interval
                print(f"\n✅ Checkpoint {checkpoint_idx}/{total} — 중간 저장 중...")
                temp_df = pd.DataFrame(all_results)
                temp_df.to_csv(
                    f"{args.save_dir}/checkpoint_{checkpoint_idx}.csv",
                    index=False,
                    encoding="utf-8-sig"
                )
            
            # Move to next batch
            start_idx = end_idx
            
            # Cleanup after every few batches
            if start_idx % (args.checkpoint_interval // 2) == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    return pd.DataFrame(all_results)

# Optimize CUDA performance
torch.backends.cudnn.benchmark = True  # Use cudnn auto-tuner
if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
    torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 precision if available

# 출력 결과 일관성을 위한 시드 설정
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Run inference with optimized settings
print("Starting inference with optimized settings...")
start_time = time.time()

try:
    result_df = predict_with_dynamic_batching(data)
    
    # Save final result as CSV
    result_df.to_csv(
        f"{args.save_dir}/qas_results.csv",
        index=False,
        encoding="utf-8-sig"
    )
    
    # Also save as JSON for easier reading
    result_dict = result_df.to_dict('records')
    with open(f"{args.save_dir}/qas_results_query_only.json", 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    
    end_time = time.time()
    total_time = end_time - start_time
    samples_per_second = len(data) / total_time
    
    print(f"✅ Processing complete!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average speed: {samples_per_second:.2f} samples/second")
    print(f"Results saved to: {args.save_dir}/qas_results.csv and {args.save_dir}/qas_results.json")
    
    # Print sample results
    print("\n📋 Sample Results:")
    for i, row in result_df.head(3).iterrows():
        print(f"\n--- Sample {i+1} ---")
        print(f"Question: {row['question'][:100]}...")
        print(f"Gold Answer: {row['gold_answer'][:100]}...")
        print(f"Predicted: {row['predicted_answer'][:100]}...")
    
except Exception as e:
    print(f"Error during inference: {e}")
    import traceback
    traceback.print_exc()

# Clean up resources
print("Cleaning up resources...")
del model, tokenizer
torch.cuda.empty_cache()
gc.collect()