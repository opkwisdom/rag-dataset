import json
import transformers
import torch
# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from tqdm import tqdm
import os
import gc
import time
import sys


class LLMManager:
    def __init__(self):
        self.llm = self.load_llm()
        self.last_reload_time = time.time()
    
    def load_llm(self):
        print("Loading Ollama model...")
        return ChatOllama(model="kimjk/llama3.2-korean", temperature=0.0, timeout=10, keep_alive=1)
    
    def reload_model(self):
        print("Reloading the LLM model now...")
        self.llm = None
        # GC 호출
        gc.collect()
        torch.cuda.empty_cache()
        self.llm = self.load_llm()
        self.last_reload_time = time.time()
    
    def get_llm(self):
        current_time = time.time()
        # 5분 = 300초
        if current_time - self.last_reload_time > 300:
            self.reload_model()
        return self.llm
    

# def make_prompt(instruction):
#     # Prompt 구성
#     question, answer = instruction['question'], instruction['answer']
#     PROMPT = f'당신은 유능한 한국어 AI 어시스턴트 입니다. 사용자의 질문에 대해 아래의 형식에 맞게 답을 해주세요. 답변의 길이는 {int(len(answer)*0.8)} ~ {int(len(answer)*1.2)}로 제한됩니다.\n\n'
#     FEW_SHOT = [
#         {
#             'user': '통신내역을 사용자 컴퓨터에 저장하는 인터넷 메신저로는 어떤 것들이 있는가?',
#             'assistant': '일부 메신저 서비스는 통신내역을 서버에만 저장하거나 혹은 양쪽 모두에 남겨놓기도 한다. 야후!, Mi3 메신저가 전자에 해당하며 네이트온, 버디버디가 후자에 해당한다.'
#         },
#         {
#             'user': 'RVA 방법의 장단점은 무엇인가?',
#             'assistant': '효율성 측면에서는 매우 뛰어나지만 Miller 루프의 연산 과정에서 랜덤값이 상쇄되어 중간 결과값이 노출되는 취약성이 있다.'
#         },
#         {
#             'user': '최근접 이웃 보간법의 장단점은 무엇인가?',
#             'assistant': '간단하기 때문에 처리속도 및 하드웨어 구현이 쉽지만 가장 가까운 픽셀을 할당함으로써 원래의 화상이 크게 변하는 결과를 가져올 수 있다.'
#         },
#     ]
#     messages = [SystemMessage(content=PROMPT)]

#     for fewshot in FEW_SHOT:
#         messages.append(HumanMessage(content=fewshot['user']))
#         messages.append(AIMessage(content=fewshot['assistant']))
    
#     messages.append(HumanMessage(content=question))

#     return messages

def make_prompt_with_examples(instruction):
    question, answer = instruction['question'], instruction['answer']
    min_length = int(len(answer)*0.8)
    max_length = int(len(answer)*1.2)
    
    PROMPT = (
        "당신은 유능한 한국어 AI 어시스턴트입니다. 사용자의 질문에 아래 형식에 맞게 답변해 주세요. "
        f"답변의 길이는 {min_length}~{max_length}자로 제한됩니다.\n\n"
        "예시:\n"
        "Q: 통신내역을 사용자 컴퓨터에 저장하는 인터넷 메신저로는 어떤 것들이 있는가?\n"
        "A: 일부 메신저 서비스는 통신내역을 서버에만 저장하거나 혹은 양쪽 모두에 남겨놓기도 한다. "
        "야후!, Mi3 메신저가 전자에 해당하며 네이트온, 버디버디가 후자에 해당한다.\n\n"
        "Q: RVA 방법의 장단점은 무엇인가?\n"
        "A: 효율성 측면에서는 매우 뛰어나지만 Miller 루프의 연산 과정에서 랜덤값이 상쇄되어 중간 결과값이 노출되는 취약성이 있다.\n\n"
        "Q: 최근접 이웃 보간법의 장단점은 무엇인가?\n"
        "A: 간단하기 때문에 처리속도 및 하드웨어 구현이 쉽지만 가장 가까운 픽셀을 할당함으로써 원래의 화상이 크게 변하는 결과를 가져올 수 있다.\n\n"
        "이제 사용자 질문에 대해 답변해 주세요. Q는 포함하지 말고 A만 포함해서 답해주세요:\n\n"
    )
    
    final_prompt = f"{PROMPT}Q: {question}\nA:"
    return [{"role": "system", "content": final_prompt}]

def safe_invoke(llm, messages, retries=3):
    for _ in range(retries):
        try:
            return llm.invoke(messages)  # 타임아웃 추가
        except Exception as e:
            print(f"Error: {e}, Retrying...")
            time.sleep(5)  # 재시도 전 5초 대기
    return None  # 실패 시 None 반환


def main():
    input_path = "../../IR_data/collection/qas_hard_clipped.jsonl"
    output_path = "../../IR_data/collection/qas_hard_ollama.jsonl"

    # 파일 로드
    with open(input_path, 'r') as f:
        data = [json.loads(line.strip()) for line in f]

    buffer_data = []
    chunk_size = 100

    # 중간부터 시작하는 것 고려
    start_idx = 0
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            saved_data = f.readlines()
        start_idx = len(saved_data)

    # Ollama 모델 로드
    llm_manager = LLMManager()  # 임시 방편

    pbar = tqdm(data[start_idx:], desc="Generate answers", initial=start_idx, total=len(data))
    # Chatbot을 이용하여 답변 생성
    for i, instruction in enumerate(pbar, start=start_idx+1):
        llm = llm_manager.get_llm()
        messages = make_prompt_with_examples(instruction)
        response = safe_invoke(llm, messages)

        if response is None:
            instruction['llm_answer'] = 'None'
        else:
            instruction['llm_answer'] = response

        buffer_data.append(instruction)

        # Chunk 단위로 파일에 저장
        if i % chunk_size == 0:
            with open(output_path, 'a', encoding='utf-8') as f:
                for item in buffer_data:
                    line = json.dumps(item, ensure_ascii=False)
                    f.write(line + '\n')
            buffer_data = []  # 버퍼 비우기

    pbar.close()
    
    # 마지막 버퍼에 남은 데이터 저장
    if buffer_data:
        with open(output_path, 'a', encoding='utf-8') as f:
            for item in buffer_data:
                line = json.dumps(item, ensure_ascii=False)
                f.write(line + '\n')

    print("Done!")

if __name__ == "__main__":
    main()