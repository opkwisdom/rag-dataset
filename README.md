# K-RAG: Korean paper-based Retrieval-Augmented Generation Dataset

K-RAG는 **전문 도메인**에 특화된 **한국어 RAG(Retrieval-Augmented Generation)** 데이터셋으로,  
질문에 대한 응답 생성을 위해 핵심 문서 정보(`keyphrase`, `key sentence`)를 활용합니다.

---


## 데이터 구성

각 샘플은 다음 정보를 포함합니다:

- `question`: 사용자 질문
- `context`: 전체 문서 또는 단락
- `keyphrases`: 문서에서 추출된 핵심 구문 목록
- `key_sentences`: 문서 내 주요 문장
- `answer`: 모델이 생성해야 할 응답

---


## 데이터셋 통계

| Split  | Samples |
|--------|---------|
| Train  | 54,224  |
| Valid  | 6,778   |
| Test   | 6,779   |

총 **67,781개**의 전문 QA 샘플이 포함되어 있으며,  
분야별 도메인 전문성과 정보 압축성을 갖춘 RAG 학습에 적합합니다.

---
![alt text](image.png)

