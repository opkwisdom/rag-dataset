import os
import time
import json
from tqdm import tqdm
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.cache import SQLiteCache   # Cache 설정
from langchain.globals import set_llm_cache         # Cache 설정
from langchain_community.callbacks import get_openai_callback   # Metadata 저장

def load_data(input_path):
    """JSONL 파일에서 데이터를 로드합니다."""
    data = []
    with open(input_path, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_results(output_path, results):
    """결과를 JSON 파일로 저장합니다."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

def setup_model_and_chain():
    """모델, 프롬프트 템플릿, 파서, 캐시를 설정하고 체인을 생성합니다."""
    # 캐시 설정
    os.makedirs("cache", exist_ok=True)
    cache = SQLiteCache(database_path="cache/my_llm_cache.db")
    set_llm_cache(cache)

    # 모델 로드
    model = ChatOpenAI(
        model="gpt-4o",
        max_tokens=1024,
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # ChatPromptTemplate 생성
    chat_prompt = ChatPromptTemplate([
        ("system", "다음은 주어진 질문과 관련된 정보입니다. 관련된 정보를 바탕으로 주어진 질문에 대해 정확하게 답변해주세요."),
        ("user", "{question}"),
        ("assistant", "<정보> {context}"),
    ])

    # Parser 지정
    parser = StrOutputParser()

    # 체인 생성: 프롬프트 템플릿 → 모델 → 파서
    chain = chat_prompt | model | parser

    return chain

def generate_answer(data, chain):
    """데이터를 처리하여 모델 응답과 메타데이터를 생성합니다."""
    rag_results = []
    metadata = {
        "Total Prompt": 0,
        "Total Completion": 0,
        "Total Tokens": 0,
        "Total Cost": 0,
        "Total Length": 0,
        "Total Count": 0,
        "Total Time": 0,
    }

    start = time.time()
    for item in tqdm(data, desc="Generating answers"):
        question = item.get('question', '')
        context = item.get('context', '')

        with get_openai_callback() as cb:
            response = chain.invoke({"question": question, "context": context})
            item['rag_answer'] = response

            # 결과 저장
            rag_results.append(item)

            # 메타데이터 업데이트
            prompt_length = len(question) + len(context)
            metadata["Total Prompt"] += cb.prompt_tokens
            metadata["Total Completion"] += cb.completion_tokens
            metadata["Total Tokens"] += cb.total_tokens
            metadata["Total Cost"] += cb.total_cost
            metadata["Total Length"] += prompt_length
            metadata["Total Count"] += 1
    end = time.time()
    elapsed = end - start
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed))    # 시:분:초 형식으로 변환
    metadata["Total Time"] = formatted_time

    return rag_results, metadata



def main():
    input_path = "../../IR_data/collection/qas_hard_clipped_context.jsonl"
    data = load_data(input_path)
    data = data[:10]    # 일부 데이터만 사용

    # 모델 및 체인 설정
    chain = setup_model_and_chain()

    # 답변 생성
    rag_results, metadata = generate_answer(data, chain)

    # 결과 저장
    results = {"Datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "metadata": metadata, "data": rag_results}
    output_path = "rag/qas_hard_rag.json"
    save_results(output_path, results)

    print("Done!")

if __name__ == "__main__":
    main()