import os
import time
import json

from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.cache import SQLiteCache   # Cache 설정
from langchain.globals import set_llm_cache         # Cache 설정
from langchain_community.callbacks import get_openai_callback   # Metadata 저장

def time_str_to_seconds(time_str):
    """'HH:MM:SS' 형식의 문자열을 초로 변환"""
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s

def seconds_to_time_str(seconds):
    """초를 'HH:MM:SS' 형식의 문자열로 변환"""
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def load_data(input_path):
    """JSONL 파일에서 데이터를 로드"""
    data = []
    with open(input_path, 'r') as f:
        data = [json.loads(line.strip()) for line in f]
    return data

def load_existing_results(output_path):
    """기존 결과를 로드"""
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            results = json.load(f)
        return results
    else:
        return {
            "Model": None,
            "Datetime": "",
            "metadata": {
                "Total Prompt": 0,
                "Total Completion": 0,
                "Total Tokens": 0,
                "Total Cost($)": 0,
                "Total Length": 0,
                "Total Count": 0,
                "Total Time": "00:00:00"
            },
            "data": []
        }

def save_results(output_path, results):
    """결과를 JSON 파일로 저장"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

def merge_metadata(existing, new):
    merged = {}
    merged["Total Prompt"] = existing.get("Total Prompt", 0) + new.get("Total Prompt", 0)
    merged["Total Completion"] = existing.get("Total Completion", 0) + new.get("Total Completion", 0)
    merged["Total Tokens"] = existing.get("Total Tokens", 0) + new.get("Total Tokens", 0)
    merged["Total Cost($)"] = existing.get("Total Cost($)", 0) + new.get("Total Cost($)", 0)
    merged["Total Length"] = existing.get("Total Length", 0) + new.get("Total Length", 0)
    merged["Total Count"] = existing.get("Total Count", 0) + new.get("Total Count", 0)

    existing_time = time_str_to_seconds(existing.get("Total Time", "00:00:00"))
    new_time = time_str_to_seconds(new.get("Total Time", "00:00:00"))
    merged["Total Time"] = seconds_to_time_str(existing_time + new_time)
    return merged

def setup_model_and_chain(model_name="gpt-4o"):
    """모델, 프롬프트 템플릿, 파서, 캐시를 설정하고 체인을 생성"""
    # 캐시 설정
    os.makedirs("cache", exist_ok=True)
    cache = SQLiteCache(database_path="cache/my_llm_cache.db")
    set_llm_cache(cache)

    # 모델 로드
    model = ChatOpenAI(
        model=model_name,
        temperature=0,
        # openai_api_key=API
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # ChatPromptTemplate 생성
    chat_prompt = ChatPromptTemplate([
        ("system", "아래 <대화>는 여러 개의 턴으로 구성되어 있습니다. 각 턴의 question, query, answer, 그리고 reference를 영어로 번역해주세요. 단, 출력 형식은 입력 형식과 동일하게 맞춰주세요."),
        ("user", "<대화> {conversation}")
    ])

    # Parser 지정
    parser = StrOutputParser()

    # 체인 생성: 프롬프트 템플릿 → 모델 → 파서
    chain = chat_prompt | model | parser

    return chain

def generate_answer(new_data, chain, output_path, existing_results, model_name='gpt-4o'):
    """데이터를 처리하여 모델 응답과 메타데이터를 생성"""
    new_results = []
    new_metadata = {
        "Total Prompt": 0,
        "Total Completion": 0,
        "Total Tokens": 0,
        "Total Cost($)": 0,
        "Total Length": 0,
        "Total Count": 0,
        "Total Time": "00:00:00",
    }

    batch_size = 50    # 100개마다 체크포인트 저장
    start = time.time()

    for i, item in enumerate(tqdm(new_data, desc="Generating answers")):
        # question = item.get('question', '')
        # context = item.get('context', '')
        # answer = item.get('answer', '')
        
        input_text = ""
        for tid in item:
            input_text += f"{tid}:\nquestion: {item[tid]['question']}\nquery: {item[tid]['query']}\nanswer: {item[tid]['answer']}\n"
            
            if len(item[tid]['reference']) > 0:
                input_text += f"reference: {item[tid]['reference'][0]}"
            
        with get_openai_callback() as cb:
            result = chain.invoke({"conversation": input_text})
            print(result)
            # item[f'{model_name}_answer'] = response

            temp = defaultdict(dict)
            
            cur_tid = ""
            for t in result.split("\n"):
                if t == "": continue
                
                if t.startswith("BK"):
                    cur_tid = t
                    temp[cur_tid] = {
                        'question': "",
                        'query': "",
                        'answer': "",
                        'reference': ""
                    }
                
                if t.startswith("question"):
                    temp[cur_tid]['question'] = t
                elif t.startswith("query"):
                    temp[cur_tid]['query'] = t
                elif t.startswith("answer"):
                    temp[cur_tid]['answer'] = t
                elif t.startswith("reference"):
                    temp[cur_tid]['reference'] = t

            # # 결과 저장
            # new_results.append(item)

            # 메타데이터 업데이트
            prompt_length = len(input_text)
            new_metadata["Total Prompt"] += cb.prompt_tokens
            new_metadata["Total Completion"] += cb.completion_tokens
            new_metadata["Total Tokens"] += cb.total_tokens
            new_metadata["Total Cost($)"] += cb.total_cost
            new_metadata["Total Length"] += prompt_length
            new_metadata["Total Count"] += 1

            with open(os.path.join(output_path), 'w+') as fout:
                fout.write(f'{json.dumps(temp, ensure_ascii=False)}\n')

        # if (i + 1) % batch_size == 0:
        #     elapsed = time.time() - start
        #     new_metadata["Total Time"] = seconds_to_time_str(elapsed)
        #     combined_data = existing_results.get("data", []) + new_results
        #     combined_metadata = merge_metadata(existing_results.get("metadata", {}), new_metadata)
        #     checkpoint = {
        #         "Model": model_name,
        #         "Datetime": datetime.now().strftime("%H:%M:%S"),
        #         "metadata": combined_metadata,
        #         "data": combined_data
        #     }
        #     save_results(output_path, checkpoint)
        #     print(f"Checkpoint saved at {output_path}")

    elapsed = time.time() - start
    new_metadata["Total Time"] = seconds_to_time_str(elapsed)
    combined_data = existing_results.get("data", []) + new_results
    combined_metadata = merge_metadata(existing_results.get("metadata", {}), new_metadata)
    final_results = {
        "Model": model_name,
        "Datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": combined_metadata,
        "data": combined_data
    }
    save_results(output_path, final_results)
    print(f"Final results saved at {output_path}")

    return new_results, new_metadata



def main():
    # input_path = "../../IR_data/collection/qas_hard_for_rag.jsonl"
    input_path = 'rac_data.jsonl'
    data = load_data(input_path)
    # data = data[6000:39000]   # 일부 데이터만 사용
    # data = data[42000:]

    # 모델 및 경로 설정
    model_name = "gpt-4o-mini"
    final_output_path = f"rac_data_en.json"
    
    print("data_loaded")

    # 기존 결과 로드
    existing_results = load_existing_results(final_output_path)
    existing_data = existing_results.get("data", [])

    start_idx = len(existing_data)
    if start_idx >= len(data):
        print("All data has been processed.")
        return
    new_data = data[start_idx:]
    
    print("existing data loaded")

    # 체인 설정
    chain = setup_model_and_chain(model_name)

    # 답변 생성
    new_results, new_metadata = generate_answer(new_data, chain, final_output_path, existing_results, model_name)

    print("Done!")

if __name__ == "__main__":
    main()