import os
import time
import json
from tqdm import tqdm
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
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
    _, ext = os.path.splitext(input_path)
    data = []
    if ext == ".json":
        with open(input_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
    elif ext == ".jsonl":
        with open(input_path, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data

def load_existing_results(output_path):
    """기존 결과를 로드"""
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding="utf-8") as f:
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

def setup_model_and_chain(model_name="gpt-4o", task="full-rag"):
    """모델, 프롬프트 템플릿, 파서, 캐시를 설정하고 체인을 생성"""
    # 캐시 설정
    os.makedirs("cache", exist_ok=True)
    cache = SQLiteCache(database_path=f"cache/my_llm_cache.db")
    set_llm_cache(cache)

    # 모델 로드
    if "gpt" in model_name:
        model = ChatOpenAI(
            model=model_name,
            max_tokens=1024,
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif 'llama' in model_name:
        model = ChatOllama(
            model=model_name,
            num_ctx=1024,
            temperature=0,
        )

    # ChatPromptTemplate 생성
    if task == "full-rag":
        chat_prompt = ChatPromptTemplate([
            ("system", "다음은 주어진 질문과 관련 문서, 정답 예시입니다. 관련된 <관련 문서>와 <정답 예시>를 보고 재구성하여 <질문>에 대해 정확하게 200자 이내로 답변해주세요."),
            ("user", "<질문> {question}"),
            ("assistant", "<관련 문서> {context}"),
            ("assistant", "<정답 예시> {answer}"),
        ])
    elif task == "compressed-rag":
        chat_prompt = ChatPromptTemplate([
            ("system", "다음은 주어진 질문과 관련 핵심 문구입니다. <질문>과 <핵심 문구>를 보고 답을 추론하여 정확하게 200자 이내로 답변해주세요."),
            ("user", "<질문> {question}"),
            ("assistant", "<핵심 문구> {context}")
        ])
    else:
        chat_prompt = ChatPromptTemplate([
            ("system", "다음은 모델이 생성한 정답입니다. <생성된 정답>을 각각의 독립적인 사실들로 나눠주세요."),
            ("user", "<질문> {question}"),
            ("assistant", "<핵심 문구> {context}")
        ])

    # Parser 지정
    parser = StrOutputParser()

    # 체인 생성: 프롬프트 템플릿 → 모델 → 파서
    chain = chat_prompt | model | parser

    return chain

def generate_answer(new_data, chain, output_path, existing_results, model_name='gpt-4o', task="full-rag"):
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

    batch_size = 100    # 100개마다 체크포인트 저장
    start = time.time()

    for i, item in enumerate(tqdm(new_data, desc="Generating answers")):
        if task == "full-rag":
            question = item.get('question', '')
            context = item.get('context', '')
            answer = item.get('answer', '')
        elif task == "compressed-rag":
            question = item.get('question', '')
            context = item.get('keyphrases', '')

        with get_openai_callback() as cb:
            if task == "full-rag":
                response = chain.invoke({"question": question, "context": context, "answer": answer})
                item[f'{model_name}_answer'] = response
            elif task == "compressed-rag":
                wo_response = chain.invoke({"question": question, "context": ""})
                c_response = chain.invoke({"question": question, "context": context})

                item[f'answer_wo'] = wo_response
                item[f'answer_c'] = c_response
            

            # 결과 저장
            new_results.append(item)

            # 메타데이터 업데이트
            prompt_length = len(question) + len(context)
            new_metadata["Total Prompt"] += cb.prompt_tokens
            new_metadata["Total Completion"] += cb.completion_tokens
            new_metadata["Total Tokens"] += cb.total_tokens
            new_metadata["Total Cost($)"] += cb.total_cost
            new_metadata["Total Length"] += prompt_length
            new_metadata["Total Count"] += 1

        if (i + 1) % batch_size == 0:
            elapsed = time.time() - start
            new_metadata["Total Time"] = seconds_to_time_str(elapsed)
            combined_data = existing_results.get("data", []) + new_results
            combined_metadata = merge_metadata(existing_results.get("metadata", {}), new_metadata)
            checkpoint = {
                "Model": model_name,
                "Datetime": datetime.now().strftime("%Y-%M-%d %H:%M:%S"),
                "metadata": combined_metadata,
                "data": combined_data
            }
            save_results(output_path, checkpoint)
            print(f"Checkpoint saved at {output_path}")

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
    input_path = "../data/qas_rag_gpt-4o-mini_final.json"
    data = load_data(input_path)
    data = data[:100]   # 일부 데이터만 사용

    # 모델 및 경로 설정
    model_name = "gpt-4o-mini"
    final_output_path = f"rag/{model_name}/compressed/qas_rag_100.json"

    # task 설정
    task = "compressed-rag"

    # 기존 결과 로드
    existing_results = load_existing_results(final_output_path)
    existing_data = existing_results.get("data", [])

    start_idx = len(existing_data)
    if start_idx >= len(data):
        print("All data has been processed.")
        return
    new_data = data[start_idx:]

    # 체인 설정
    chain = setup_model_and_chain(model_name, task)

    # 답변 생성
    new_results, new_metadata = generate_answer(new_data, chain, final_output_path, existing_results, model_name, task)

    print("Done!")

if __name__ == "__main__":
    main()