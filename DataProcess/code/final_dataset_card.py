import json

def modify_dataset(input_file, output_file):
    """
    JSON 데이터셋을 수정하는 함수
    1. id를 0부터 순서대로 변경
    2. context 필드 제거
    3. top-3, top-4, top-5 배열을 한 줄로 출력
    """
    
    # JSON 파일 읽기
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    print(f"총 {len(data)}개 항목을 수정합니다...")
    
    # 데이터 수정
    for i, item in enumerate(data):
        # 1. id를 순서대로 변경 (0부터 시작)
        item['id'] = i
        
        # 2. context 필드 제거
        if 'context' in item:
            del item['context']
    
    # 3. JSON 저장 (배열을 한 줄로 출력)
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2, separators=(',', ': '))
    
    print(f"수정 완료! {output_file}에 저장되었습니다.")
    print(f"- id: 0부터 {len(data)-1}까지 순서대로 변경")
    print(f"- context 필드 제거")
    print(f"- top-3, top-4, top-5 배열을 한 줄로 출력")

# 사용 예시
if __name__ == "__main__":
    input_file = "output_complete_final_with_indices.json"  # 입력 파일명
    output_file = "final_dataset.json"  # 출력 파일명
    
    try:
        modify_dataset(input_file, output_file)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {input_file}")
    except json.JSONDecodeError:
        print("JSON 파일 형식이 올바르지 않습니다.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")