from datetime import datetime
from openai import OpenAI
from copy import deepcopy
import json
import time
import os

def time_str_to_seconds(time_str):
    """'HH:MM:SS' 형식의 문자열을 초로 변환"""
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s

def seconds_to_time_str(seconds):
    """초를 'HH:MM:SS' 형식의 문자열로 변환"""
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def load_data(path):
    _, ext = os.path.splitext(path)
    if ext == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
    elif ext == '.jsonl':
        with open(path, 'r') as f:
            data = [json.loads(d) for d in f]
    return data

def make_batches(original_data):
    init_template = {
        "custom_id": None,  # custom_id는 batch내에서 유일한 값을 가지도록 설정해야합니다.
        "method": "POST", 
        "url": "/v1/chat/completions",
        "body": {"model": "gpt-4o-mini",
                "messages":
                    None,
                "max_tokens": 1024
                }
        }

    batch_size = 3000

    batches = []
    for i, data in enumerate(original_data):
        temp = deepcopy(init_template)
        temp['custom_id'] = f'{i}'

        question = data.get('question', '')
        context = data.get('context', '')
        answer = data.get('answer', '')

        messages = [
            {'role': 'system', 'content': "다음은 주어진 질문과 관련 문서, 정답 예시입니다. 관련된 <관련 문서>와 <정답 예시>를 바탕으로 주어진 <질문>에 대해 정확하게 200자 이내로 답변해주세요."},
            {'role': 'user', 'content': f'<질문> {question}'},
            {'role': 'assistant', 'content': f'<관련 문서> {context}'},
            {'role': 'assistant', 'content': f'<정답 예시> {answer}'},
        ]
        temp['body']['messages'] = messages
        batches.append(temp)

        if (i + 1) % batch_size == 0:
            batch_path = f"rag/gpt-4o-mini/batch/batches_{i+1}.jsonl"
            with open(batch_path, 'w') as file:
                for item in batches:
                    json_string = json.dumps(item, ensure_ascii=False)
                    file.write(json_string + '\n')
            batches = []

    if batches:
        batch_path = f"rag/gpt-4o-mini/batch/batches_{i+1}.jsonl"
        with open(batch_path, 'w') as file:
            for item in batches:
                json_string = json.dumps(item, ensure_ascii=False)
                file.write(json_string + '\n')
    
    print("Make batches done!")

def upload_batches(batch_path):
    # 배치 파일 만들기
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # available only in version after openai==1.2.0
    batch_input_file = client.files.create(
        file=open(batch_path, "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    print(f"Batch file uploaded: {batch_input_file_id}")
    batch_request = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h", #응답 요청 시간(240502기준 24h만 가능)
        metadata={
        "description": "10000 prompts for gpt-4o-mini"
        }
    )
    batch_id = batch_request.id

    # 배치 처리가 끝날 때까지 대기
    while True:
        current_batch = client.batches.retrieve(batch_id)
        status = current_batch.status

        if status == "completed":
            print("Batch processing completed!")
            break
        else:
            print(f"Current status: {status}")
            time.sleep(30)

    return batch_id

def parse_request_and_save(batch_file_id, output_path, start_idx, end_idx):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    batch_request = client.batches.retrieve(batch_file_id)
    file_id = batch_request.output_file_id

    elapsed = batch_request.finalizing_at - batch_request.created_at
    elapsed_time = seconds_to_time_str(elapsed)

    contents = client.files.content(file_id).text.split('\n')[:-1]
    reference_data = original_data[start_idx:end_idx]
    assert len(contents) == len(reference_data), f'Length mismatch: {len(contents)} vs {len(reference_data)}'

    metadata = {
        "Total Prompt": 0,
        "Total Completion": 0,
        "Total Tokens": 0,
        "Total Length": 0,
        "Total Count": 0,
        "Total Time": ""
    }

    for content, ref in zip(contents, reference_data):
        data = json.loads(content)
        body = data['response']['body']

        # Meta data update
        prompt_tokens = body['usage']['prompt_tokens']
        completion_tokens = body['usage']['completion_tokens']
        total_tokens = body['usage']['total_tokens']
        prompt_length = len(ref['question']) + len(ref['context'])

        metadata["Total Prompt"] += prompt_tokens
        metadata["Total Completion"] += completion_tokens
        metadata["Total Tokens"] += total_tokens
        metadata["Total Length"] += prompt_length
        metadata["Total Count"] += 1

        # Data append
        ref['gpt-4o-mini_answer'] = body['choices'][0]['message']['content']

    metadata["Total Time"] = elapsed_time
    parsed_results = {
        "Model": "gpt-4o-mini",
        "Datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": metadata,
        "data": reference_data
    }

    with open(output_path, 'w') as f:
        json.dump(parsed_results, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    input_path = "../../IR_data/collection/qas_hard_for_rag.jsonl"
    batch_dir = "rag/gpt-4o-mini/batch"
    output_dir = "rag/gpt-4o-mini/output"

    original_data = load_data(input_path)
    if not os.listdir('rag/gpt-4o-mini/batch'):
        make_batches(original_data)
    
    batch_files = os.listdir(batch_dir)
    batch_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    for i, batch_file in enumerate(batch_files):
        filename, ext = os.path.splitext(batch_file)
        if os.path.exists(os.path.join(output_dir, f"{filename}.json")):
            continue

        batch_name, ext = os.path.splitext(batch_file)
        batch_path = os.path.join(batch_dir, batch_file)
        output_path = os.path.join(output_dir, f'{batch_name}.json')

        batch_file_id = upload_batches(batch_path)

        start_idx = i * 3000
        end_idx = (i+1) * 3000
        parse_request_and_save(batch_file_id, output_path, start_idx, end_idx)

        print(f"Done for {start_idx} to {end_idx}!")
        time.sleep(5)

    # batch_file_id = "batch_67c050d5d2ac8190ad422a7d8f8762b4"
    # parse_request_and_save(batch_file_id, "rag/gpt-4o-mini/output/batches_42000.json", 39000, 42000)