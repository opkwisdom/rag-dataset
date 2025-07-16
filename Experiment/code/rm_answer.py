import json

def load_json(file_path):
    """
    JSON 파일을 로드하는 함수
    :param file_path: JSON 파일 경로
    :return: JSON 데이터
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    """
    JSON 데이터를 파일에 저장하는 함수
    :param data: 저장할 JSON 데이터
    :param file_path: 저장할 파일 경로
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_file = "test.json"  # 입력 파일 경로
    output_file = "test_modified.json"  # 출력 파일 경로

    # JSON 데이터 로드
    data = load_json(input_file)

    filtered_data = []
    for item in data:
        item.pop('answer')
        item.pop('gold_passage_id')
        filtered_data.append(item)

    # 필터링된 데이터 저장
    save_json(filtered_data, output_file)

    print("답변이 있는 항목만 필터링하여 저장 완료!")