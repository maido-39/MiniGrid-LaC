# 유사도 계산 모듈 (Similarity Calculator)

Word2Vec과 SBERT를 사용하여 단어 및 문장 간의 의미적 유사도와 거리를 계산하는 모듈입니다.

## 📁 파일 구조

```
similarity_calculator/
├── __init__.py            # 패키지 초기화 파일
├── word_similarity.py     # Word2Vec을 사용한 단어 간 유사도 계산
├── sentence_similarity.py # SBERT를 사용한 문장 간 유사도 계산
├── example.py             # 통합 사용 예제
└── README.md              # 이 파일
```

## 🚀 설치 방법

필요한 패키지를 설치합니다:

```bash
pip install gensim sentence-transformers numpy
```

또는 프로젝트의 `requirements.txt`에 다음을 추가하세요:

```
gensim>=4.0.0
sentence-transformers>=2.0.0
numpy>=1.20.0
```

## 📖 사용 방법

### 빠른 시작

```python
# 패키지에서 직접 import
from similarity_calculator import Word2VecSimilarity, SBERTSimilarity

# 또는 개별 모듈에서 import
from similarity_calculator.word_similarity import Word2VecSimilarity
from similarity_calculator.sentence_similarity import SBERTSimilarity
```

### 1. Word2Vec - 단어 간 유사도 계산

```python
from similarity_calculator import Word2VecSimilarity

# Word2VecSimilarity 객체 생성 (예제 데이터로 학습)
similarity_calculator = Word2VecSimilarity()

# 코사인 유사도 계산
similarity = similarity_calculator.cosine_similarity('사과', '바나나')
print(f"사과와 바나나의 유사도: {similarity}")

# 유클리드 거리 계산
distance = similarity_calculator.euclidean_distance('사과', '바나나')
print(f"사과와 바나나의 거리: {distance}")

# 가장 유사한 단어 찾기
similar_words = similarity_calculator.find_most_similar('사과', topn=5)
print(f"사과와 유사한 단어들: {similar_words}")
```

### 2. SBERT - 문장 간 유사도 계산

```python
from similarity_calculator import SBERTSimilarity

# SBERTSimilarity 객체 생성
similarity_calculator = SBERTSimilarity()

# 코사인 유사도 계산
similarity = similarity_calculator.cosine_similarity(
    '오늘 날씨가 좋다',
    '오늘 날씨가 맑다'
)
print(f"문장 간 유사도: {similarity}")

# 유클리드 거리 계산
distance = similarity_calculator.euclidean_distance(
    '오늘 날씨가 좋다',
    '오늘 날씨가 맑다'
)
print(f"문장 간 거리: {distance}")

# 가장 유사한 문장 찾기
query = "나는 프로그래밍을 좋아한다"
candidates = [
    "코딩은 재미있다",
    "나는 음악을 좋아한다",
    "프로그래밍은 어렵다"
]
similar_sentences = similarity_calculator.find_most_similar(
    query, candidates, topn=3
)
print(f"유사한 문장들: {similar_sentences}")
```

## 🧪 실행 예제

각 모듈을 직접 실행하여 예제를 확인할 수 있습니다:

```bash
# Word2Vec 예제 실행
python similarity_calculator/word_similarity.py

# SBERT 예제 실행
python similarity_calculator/sentence_similarity.py

# 통합 예제 실행
python similarity_calculator/example.py
```

## 📊 유사도 측정 방법

### 코사인 유사도 (Cosine Similarity)
- **범위**: -1 ~ 1
- **의미**:
  - 1에 가까울수록: 매우 유사
  - 0에 가까울수록: 관련 없음
  - -1에 가까울수록: 반대 의미
- **공식**: cos(θ) = (A · B) / (||A|| * ||B||)

### 유클리드 거리 (Euclidean Distance)
- **범위**: 0 이상
- **의미**: 거리가 작을수록 유사
- **공식**: distance = sqrt(sum((A[i] - B[i])²))

## 💡 고급 사용법

### 사전 학습된 Word2Vec 모델 사용

```python
# Google News Word2Vec 모델 사용 (영어)
# 다운로드: https://code.google.com/archive/p/word2vec/
similarity_calculator = Word2VecSimilarity(
    model_path='GoogleNews-vectors-negative300.bin'
)
```

### 다른 SBERT 모델 사용

```python
# 더 정확한 다국어 모델 사용
similarity_calculator = SBERTSimilarity(
    model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
)

# 영어 전용 모델 사용 (더 빠름)
similarity_calculator = SBERTSimilarity(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)
```

### 커스텀 데이터로 Word2Vec 학습

```python
# 나만의 문장 데이터로 학습
sentences = [
    ['안녕', '하세요'],
    ['좋은', '아침'],
    ['안녕', '하세요', '반갑습니다']
]

similarity_calculator = Word2VecSimilarity(sentences=sentences)
```

## ⚠️ 주의사항

1. **SBERT 모델 다운로드**: SBERT 모델은 처음 실행 시 자동으로 다운로드됩니다. 인터넷 연결이 필요합니다.

2. **메모리 사용량**: SBERT 모델은 상당한 메모리를 사용합니다. 메모리가 부족한 경우 더 작은 모델을 사용하세요.

3. **언어 지원**: 
   - Word2Vec: 학습 데이터에 따라 다름
   - SBERT: `paraphrase-multilingual-*` 모델은 한국어를 지원합니다

4. **처리 속도**: 
   - Word2Vec: 매우 빠름
   - SBERT: 상대적으로 느리지만 더 정확함

## 📚 문서 및 참고 자료

### 프로젝트 문서
- [상세 API 문서](../docs/similarity-calculator-api.md) - 모든 메서드의 상세한 사용법과 예제

### 외부 참고 자료
- [Word2Vec 논문](https://arxiv.org/abs/1301.3781)
- [SBERT 논문](https://arxiv.org/abs/1908.10084)
- [Sentence Transformers 문서](https://www.sbert.net/)

