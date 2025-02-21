import json
import torch
import time
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from llama_cpp import Llama

def make_prompt_with_examples(instruction):
    question, answer = instruction['question'], instruction['answer']
    min_length = int(len(answer)*0.8)
    max_length = int(len(answer)*1.2)

    examples = [
        ("통신내역을 사용자 컴퓨터에 저장하는 인터넷 메신저로는 어떤 것들이 있는가?",
         "일부 메신저 서비스는 통신내역을 서버에만 저장하거나 혹은 양쪽 모두에 남겨놓기도 한다. 야후!, Mi3 메신저가 전자에 해당하며 네이트온, 버디버디가 후자에 해당한다."),
        ("RVA 방법의 장단점은 무엇인가?",
         "효율성 측면에서는 매우 뛰어나지만 Miller 루프의 연산 과정에서 랜덤값이 상쇄되어 중간 결과값이 노출되는 취약성이 있다."),
        ("최근접 이웃 보간법의 장단점은 무엇인가?",
         "간단하기 때문에 처리속도 및 하드웨어 구현이 쉽지만 가장 가까운 픽셀을 할당함으로써 원래의 화상이 크게 변하는 결과를 가져올 수 있다."),
    ]

    # Chat Completion 형식으로 변환
    messages = [{"role": "system", "content": f"당신은 유능한 한국어 AI assistant입니다. 사용자의 질문에 대해 아래 형식에 맞게 답변해 주세요. 답변 길이는 {min_length}~{max_length}자로 제한됩니다."}]
    
    for ex_question, ex_answer in examples:
        messages.append({"role": "user", "content": f"Q: {ex_question}"})
        messages.append({"role": "assistant", "content": f"A: {ex_answer}"})
    
    # 마지막 질문 추가
    messages.append({"role": "user", "content": f"Q: {question}\nA:"})

    return messages



def main():
    input_path = "../../IR_data/collection/qas_hard_clipped.jsonl"
    output_path = "../../IR_data/collection/qas_hard_ollama.jsonl"

    # 파일 로드
    with open(input_path, 'r') as f:
        data = [json.loads(line.strip()) for line in f]

    buffer_data = []
    chunk_size = 2000

    # 중간부터 시작하는 것 고려
    start_idx = 0
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            saved_data = f.readlines()
        start_idx = len(saved_data)

    # Langhcian 구성 (llama 모델 로드)
    llm = Llama(
        model_path="model/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf",
        n_gpu_layers=32,
        use_mmap=True,
    )

    pbar = tqdm(data[start_idx:], desc="Generate answers", initial=start_idx, total=len(data))

    # Chatbot을 이용하여 답변 생성
    for i, instruction in enumerate(pbar, start=start_idx+1):
        final_prompt = make_prompt_with_examples(instruction)

        response = llm.create_chat_completion(messages=final_prompt)
        import pdb; pdb.set_trace()

        instruction['llm_answer'] = response['choices'][0]['message']['content']
        

        buffer_data.append(instruction)        

        # Chunk 단위로 파일에 저장
        if i % chunk_size == 0:
            with open(output_path, 'a', encoding='utf-8') as f:
                for item in buffer_data:
                    line = json.dumps(item, ensure_ascii=False)
                    f.write(line + '\n')
            buffer_data = []  # 버퍼 비우기
    
    # 마지막 버퍼에 남은 데이터 저장
    if buffer_data:
        with open(output_path, 'a', encoding='utf-8') as f:
            for item in buffer_data:
                line = json.dumps(item, ensure_ascii=False)
                f.write(line + '\n')

    print("Done!")

if __name__ == "__main__":
    main()