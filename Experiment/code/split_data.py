import json
import random

def load_json(file_path):
    """
    JSON 파일을 로드하는 함수
    :param file_path: JSON 파일 경로
    :return: JSON 데이터
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def split_dataset(dataset, split_ratio=0.8):
    """
    데이터셋을 주어진 비율로 분할하는 함수
    :param dataset: 전체 데이터셋
    :param split_ratio: 훈련 데이터 비율 (0~1 사이)
    :return: 훈련 데이터와 검증 데이터
    """
    random.seed(42)  # 재현성을 위해 시드 설정
    random.shuffle(dataset)  # 데이터셋을 무작위로 섞음

    split_index = int(len(dataset) * split_ratio)
    train_data = dataset[:split_index]
    val_test_data = dataset[split_index:]
    val_data = val_test_data[:len(val_test_data) // 2]
    test_data = val_test_data[len(val_test_data) // 2:]
    return train_data, val_data, test_data

def save_json(data, file_path):
    """
    JSON 데이터를 파일에 저장하는 함수
    :param data: 저장할 JSON 데이터
    :param file_path: 저장할 파일 경로
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    input_file = "../final/final_dataset.json"  # 입력 파일 경로
    output_train_file = "train.json"
    output_val_file = "valid.json"
    output_test_file = "test.json"

    # JSON 데이터 로드
    dataset = load_json(input_file)

    # 데이터셋 분할
    train_data, val_data, test_data = split_dataset(dataset)

    # 분할된 데이터 저장
    save_json(train_data, output_train_file)
    save_json(val_data, output_val_file)
    save_json(test_data, output_test_file)

    print("데이터셋 분할 완료!")