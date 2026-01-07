# Similarity Calculator API 문서

이 문서는 `Word2VecSimilarity`와 `SBERTSimilarity` 클래스의 API를 설명합니다. 이 모듈들은 단어 및 문장 간의 의미적 유사도와 거리를 계산할 수 있게 해줍니다.

## 목차

- [Word2VecSimilarity - 단어 간 유사도 계산](#word2vecsimilarity---단어-간-유사도-계산)
- [SBERTSimilarity - 문장 간 유사도 계산](#sbertsimilarity---문장-간-유사도-계산)
- [사용 예제](#사용-예제)

---

## Word2VecSimilarity - 단어 간 유사도 계산

Word2Vec 모델을 사용하여 단어 간의 의미적 유사도를 계산하는 클래스입니다.

### 클래스 초기화

```python
from similarity_calculator import Word2VecSimilarity

# 방법 1: 예제 데이터로 자동 학습 (기본값)
word_sim = Word2VecSimilarity()

# 방법 2: 사전 학습된 모델 로드
word_sim = Word2VecSimilarity(model_path='word2vec.model')
# 또는
word_sim = Word2VecSimilarity(model_path='GoogleNews-vectors-negative300.bin')

# 방법 3: 커스텀 데이터로 학습
sentences = [
    ['안녕', '하세요'],
    ['좋은', '아침'],
    ['안녕', '하세요', '반갑습니다']
]
word_sim = Word2VecSimilarity(sentences=sentences)
```

#### 파라미터

- `model_path` (str, optional): 사전 학습된 Word2Vec 모델 파일 경로
  - `.bin` 파일: Google News 등 사전 학습된 모델
  - `.model` 파일: Gensim으로 저장한 모델
- `sentences` (list, optional): 모델을 학습시킬 문장 리스트
  - 형식: `[['단어1', '단어2'], ['단어3', '단어4']]`
  - 각 내부 리스트는 단어로 분리된 하나의 문장

### 주요 메서드

#### get_word_vector(word)

특정 단어의 벡터 표현을 가져옵니다.

```python
vector = word_sim.get_word_vector('사과')
print(vector)  # numpy.ndarray (100차원)
```

**Parameters:**
- `word` (str): 벡터를 가져올 단어

**Returns:**
- `numpy.ndarray`: 단어의 벡터 표현 (100차원 배열)
- `None`: 단어가 모델에 없는 경우

**예제:**
```python
word_sim = Word2VecSimilarity()
vector = word_sim.get_word_vector('사과')
if vector is not None:
    print(f"벡터 차원: {vector.shape}")  # (100,)
    print(f"벡터 일부: {vector[:5]}")     # 처음 5개 값
```

---

#### cosine_similarity(word1, word2)

두 단어 간의 코사인 유사도를 계산합니다.

```python
similarity = word_sim.cosine_similarity('사과', '바나나')
print(similarity)  # 0.5234
```

**Parameters:**
- `word1` (str): 첫 번째 단어
- `word2` (str): 두 번째 단어

**Returns:**
- `float`: 코사인 유사도 값 (-1 ~ 1)
  - **1에 가까울수록**: 두 단어가 매우 유사함
  - **0에 가까울수록**: 두 단어가 관련이 없음
  - **-1에 가까울수록**: 두 단어가 반대 의미
- `None`: 단어가 모델에 없는 경우

**예제:**
```python
word_sim = Word2VecSimilarity()

# 유사한 단어들 (과일)
similarity1 = word_sim.cosine_similarity('사과', '바나나')
print(f"사과-바나나: {similarity1:.4f}")  # 약 0.5 이상

# 관련 없는 단어들
similarity2 = word_sim.cosine_similarity('사과', '자동차')
print(f"사과-자동차: {similarity2:.4f}")  # 약 0.0~0.2

# 유사한 단어들 (동물)
similarity3 = word_sim.cosine_similarity('고양이', '강아지')
print(f"고양이-강아지: {similarity3:.4f}")  # 약 0.5 이상
```

---

#### euclidean_distance(word1, word2)

두 단어 간의 유클리드 거리를 계산합니다.

```python
distance = word_sim.euclidean_distance('사과', '바나나')
print(distance)  # 8.234
```

**Parameters:**
- `word1` (str): 첫 번째 단어
- `word2` (str): 두 번째 단어

**Returns:**
- `float`: 유클리드 거리 (0 이상의 값)
  - **거리가 작을수록**: 두 단어가 더 유사함
  - **거리가 클수록**: 두 단어가 더 다름
- `None`: 단어가 모델에 없는 경우

**예제:**
```python
word_sim = Word2VecSimilarity()

# 유사한 단어들
distance1 = word_sim.euclidean_distance('사과', '바나나')
print(f"사과-바나나 거리: {distance1:.4f}")  # 작은 값 (약 5~10)

# 관련 없는 단어들
distance2 = word_sim.euclidean_distance('사과', '자동차')
print(f"사과-자동차 거리: {distance2:.4f}")  # 큰 값 (약 10 이상)
```

---

#### find_most_similar(word, topn=5)

주어진 단어와 가장 유사한 단어들을 찾습니다.

```python
similar_words = word_sim.find_most_similar('사과', topn=5)
for word, similarity in similar_words:
    print(f"{word}: {similarity:.4f}")
```

**Parameters:**
- `word` (str): 기준 단어
- `topn` (int): 반환할 유사 단어의 개수 (기본값: 5)

**Returns:**
- `list`: `(단어, 유사도)` 튜플의 리스트, 유사도 순으로 정렬됨
- 빈 리스트: 단어가 모델에 없는 경우

**예제:**
```python
word_sim = Word2VecSimilarity()

# '사과'와 가장 유사한 단어 3개 찾기
similar_words = word_sim.find_most_similar('사과', topn=3)
print("'사과'와 유사한 단어들:")
for word, similarity in similar_words:
    print(f"  - {word}: {similarity:.4f}")
```

---

## SBERTSimilarity - 문장 간 유사도 계산

SBERT (Sentence-BERT) 모델을 사용하여 문장 간의 의미적 유사도를 계산하는 클래스입니다.

### 클래스 초기화

```python
from similarity_calculator import SBERTSimilarity

# 기본 모델 사용 (다국어 지원, 한국어 포함)
sentence_sim = SBERTSimilarity()

# 다른 모델 사용
sentence_sim = SBERTSimilarity(
    model_name='sentence-transformers/all-MiniLM-L6-v2'  # 영어 전용, 더 빠름
)
```

#### 파라미터

- `model_name` (str): 사용할 SBERT 모델 이름 (기본값: `'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'`)
  - `'paraphrase-multilingual-MiniLM-L12-v2'`: 다국어 지원 모델 (한국어 포함) - **권장**
  - `'sentence-transformers/all-MiniLM-L6-v2'`: 영어 전용 모델 (더 빠름)
  - `'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'`: 더 정확한 다국어 모델 (더 느림)

**참고:** 처음 실행 시 모델을 자동으로 다운로드하므로 인터넷 연결이 필요합니다.

### 주요 메서드

#### get_sentence_vector(sentence)

특정 문장의 벡터 표현을 가져옵니다.

```python
vector = sentence_sim.get_sentence_vector('오늘 날씨가 좋다')
print(vector.shape)  # (384,) 또는 모델에 따라 다름
```

**Parameters:**
- `sentence` (str): 벡터로 변환할 문장

**Returns:**
- `numpy.ndarray`: 문장의 벡터 표현 (384차원 또는 모델에 따라 다름)

**예제:**
```python
sentence_sim = SBERTSimilarity()
vector = sentence_sim.get_sentence_vector('오늘 날씨가 좋다')
print(f"벡터 차원: {vector.shape}")
print(f"벡터 일부: {vector[:5]}")
```

---

#### get_sentences_vectors(sentences)

여러 문장의 벡터를 한 번에 가져옵니다 (배치 처리로 더 효율적).

```python
sentences = ['문장1', '문장2', '문장3']
vectors = sentence_sim.get_sentences_vectors(sentences)
print(vectors.shape)  # (3, 384)
```

**Parameters:**
- `sentences` (list): 벡터로 변환할 문장들의 리스트

**Returns:**
- `numpy.ndarray`: 문장 벡터들의 배열 (문장 수 x 벡터 차원)

**예제:**
```python
sentence_sim = SBERTSimilarity()
sentences = [
    '오늘 날씨가 좋다',
    '오늘 날씨가 맑다',
    '나는 사과를 좋아한다'
]
vectors = sentence_sim.get_sentences_vectors(sentences)
print(f"벡터 배열 형태: {vectors.shape}")  # (3, 384)
```

---

#### cosine_similarity(sentence1, sentence2)

두 문장 간의 코사인 유사도를 계산합니다.

```python
similarity = sentence_sim.cosine_similarity(
    '오늘 날씨가 좋다',
    '오늘 날씨가 맑다'
)
print(similarity)  # 0.8234
```

**Parameters:**
- `sentence1` (str): 첫 번째 문장
- `sentence2` (str): 두 번째 문장

**Returns:**
- `float`: 코사인 유사도 값 (-1 ~ 1)
  - **1에 가까울수록**: 두 문장이 매우 유사함 (의미가 같음)
  - **0에 가까울수록**: 두 문장이 관련이 없음
  - **-1에 가까울수록**: 두 문장이 반대 의미

**예제:**
```python
sentence_sim = SBERTSimilarity()

# 거의 같은 의미
similarity1 = sentence_sim.cosine_similarity(
    '오늘 날씨가 좋다',
    '오늘 날씨가 맑다'
)
print(f"유사도: {similarity1:.4f}")  # 약 0.7 이상

# 유사한 구조, 다른 내용
similarity2 = sentence_sim.cosine_similarity(
    '나는 사과를 좋아한다',
    '나는 바나나를 좋아한다'
)
print(f"유사도: {similarity2:.4f}")  # 약 0.5~0.7

# 관련 없음
similarity3 = sentence_sim.cosine_similarity(
    '고양이는 귀여운 동물이다',
    '자동차는 빠르다'
)
print(f"유사도: {similarity3:.4f}")  # 약 0.3 이하
```

---

#### euclidean_distance(sentence1, sentence2)

두 문장 간의 유클리드 거리를 계산합니다.

```python
distance = sentence_sim.euclidean_distance(
    '오늘 날씨가 좋다',
    '오늘 날씨가 맑다'
)
print(distance)  # 4.567
```

**Parameters:**
- `sentence1` (str): 첫 번째 문장
- `sentence2` (str): 두 번째 문장

**Returns:**
- `float`: 유클리드 거리 (0 이상의 값)
  - **거리가 작을수록**: 두 문장이 더 유사함
  - **거리가 클수록**: 두 문장이 더 다름

**예제:**
```python
sentence_sim = SBERTSimilarity()

# 유사한 문장들
distance1 = sentence_sim.euclidean_distance(
    '오늘 날씨가 좋다',
    '오늘 날씨가 맑다'
)
print(f"거리: {distance1:.4f}")  # 작은 값 (약 5 이하)

# 관련 없는 문장들
distance2 = sentence_sim.euclidean_distance(
    '고양이는 귀여운 동물이다',
    '자동차는 빠르다'
)
print(f"거리: {distance2:.4f}")  # 큰 값 (약 20 이상)
```

---

#### batch_cosine_similarity(sentences)

여러 문장들 간의 코사인 유사도를 한 번에 계산합니다 (유사도 행렬).

```python
sentences = ['문장1', '문장2', '문장3']
similarity_matrix = sentence_sim.batch_cosine_similarity(sentences)
print(similarity_matrix)
# [[1.000, 0.823, 0.456],
#  [0.823, 1.000, 0.512],
#  [0.456, 0.512, 1.000]]
```

**Parameters:**
- `sentences` (list): 유사도를 계산할 문장들의 리스트

**Returns:**
- `numpy.ndarray`: 유사도 행렬 (N x N, N은 문장 개수)
  - `similarity_matrix[i][j]`는 `sentences[i]`와 `sentences[j]`의 유사도
  - 대각선은 항상 1.0 (자기 자신과의 유사도)

**예제:**
```python
sentence_sim = SBERTSimilarity()
sentences = [
    '나는 사과를 좋아한다',
    '나는 바나나를 좋아한다',
    '고양이는 귀여운 동물이다',
    '강아지는 친절한 동물이다',
    '자동차는 빠르다'
]

similarity_matrix = sentence_sim.batch_cosine_similarity(sentences)

# 각 문장 쌍의 유사도 확인
for i, sent1 in enumerate(sentences):
    for j, sent2 in enumerate(sentences):
        if i < j:  # 중복 제거
            print(f"{sent1[:15]}... ↔ {sent2[:15]}...: {similarity_matrix[i][j]:.4f}")
```

---

#### find_most_similar(query_sentence, candidate_sentences, topn=3)

주어진 문장과 가장 유사한 문장들을 찾습니다.

```python
query = "나는 프로그래밍을 좋아한다"
candidates = [
    "코딩은 재미있다",
    "나는 음악을 좋아한다",
    "프로그래밍은 어렵다"
]
similar_sentences = sentence_sim.find_most_similar(query, candidates, topn=3)
```

**Parameters:**
- `query_sentence` (str): 기준 문장
- `candidate_sentences` (list): 후보 문장들의 리스트
- `topn` (int): 반환할 유사 문장의 개수 (기본값: 3)

**Returns:**
- `list`: `(문장, 유사도)` 튜플의 리스트, 유사도 순으로 정렬됨

**예제:**
```python
sentence_sim = SBERTSimilarity()

query = "나는 프로그래밍을 좋아한다"
candidates = [
    "코딩은 재미있다",
    "나는 음악을 좋아한다",
    "프로그래밍은 어렵다",
    "나는 개발자가 되고 싶다",
    "오늘 날씨가 좋다"
]

similar_sentences = sentence_sim.find_most_similar(query, candidates, topn=3)

print(f"기준 문장: '{query}'")
print("\n가장 유사한 문장 Top 3:")
for i, (sentence, similarity) in enumerate(similar_sentences, 1):
    print(f"  {i}. {sentence} (유사도: {similarity:.4f})")
```

---

## 사용 예제

### 예제 1: 단어 유사도 계산

```python
from similarity_calculator import Word2VecSimilarity

# Word2VecSimilarity 객체 생성
word_sim = Word2VecSimilarity()

# 단어 쌍들의 유사도 계산
word_pairs = [
    ('사과', '바나나'),
    ('고양이', '강아지'),
    ('자동차', '버스'),
]

print("단어 간 코사인 유사도:")
for word1, word2 in word_pairs:
    similarity = word_sim.cosine_similarity(word1, word2)
    distance = word_sim.euclidean_distance(word1, word2)
    print(f"  {word1} ↔ {word2}: 유사도={similarity:.4f}, 거리={distance:.4f}")

# 가장 유사한 단어 찾기
similar_words = word_sim.find_most_similar('사과', topn=3)
print("\n'사과'와 유사한 단어들:")
for word, similarity in similar_words:
    print(f"  - {word}: {similarity:.4f}")
```

### 예제 2: 문장 유사도 계산

```python
from similarity_calculator import SBERTSimilarity

# SBERTSimilarity 객체 생성
sentence_sim = SBERTSimilarity()

# 문장 쌍들의 유사도 계산
sentence_pairs = [
    ('오늘 날씨가 좋다', '오늘 날씨가 맑다'),
    ('나는 사과를 좋아한다', '나는 바나나를 좋아한다'),
    ('고양이는 귀여운 동물이다', '자동차는 빠르다'),
]

print("문장 간 코사인 유사도:")
for sent1, sent2 in sentence_pairs:
    similarity = sentence_sim.cosine_similarity(sent1, sent2)
    distance = sentence_sim.euclidean_distance(sent1, sent2)
    print(f"  '{sent1}' ↔ '{sent2}'")
    print(f"    유사도: {similarity:.4f}, 거리: {distance:.4f}")

# 가장 유사한 문장 찾기
query = "나는 프로그래밍을 좋아한다"
candidates = [
    "코딩은 재미있다",
    "나는 음악을 좋아한다",
    "프로그래밍은 어렵다",
    "나는 개발자가 되고 싶다"
]

similar_sentences = sentence_sim.find_most_similar(query, candidates, topn=3)
print(f"\n'{query}'와 가장 유사한 문장들:")
for i, (sentence, similarity) in enumerate(similar_sentences, 1):
    print(f"  {i}. {sentence} (유사도: {similarity:.4f})")
```

### 예제 3: 배치 유사도 계산

```python
from similarity_calculator import SBERTSimilarity

sentence_sim = SBERTSimilarity()

sentences = [
    '나는 사과를 좋아한다',
    '나는 바나나를 좋아한다',
    '고양이는 귀여운 동물이다',
    '강아지는 친절한 동물이다',
    '자동차는 빠르다'
]

# 모든 문장 쌍의 유사도 행렬 계산
similarity_matrix = sentence_sim.batch_cosine_similarity(sentences)

print("문장들 간의 유사도 행렬:")
print(f"{'':<25}", end="")
for i in range(len(sentences)):
    print(f"{i+1:<8}", end="")
print()

for i, sent in enumerate(sentences):
    sent_display = sent[:23] + ".." if len(sent) > 25 else sent
    print(f"{sent_display:<25}", end="")
    for j in range(len(sentences)):
        print(f"{similarity_matrix[i][j]:>7.3f} ", end="")
    print()
```

### 예제 4: 통합 사용

```python
from similarity_calculator import Word2VecSimilarity, SBERTSimilarity

# 단어 유사도
word_sim = Word2VecSimilarity()
word_similarity = word_sim.cosine_similarity('사과', '바나나')
print(f"단어 유사도: {word_similarity:.4f}")

# 문장 유사도
sentence_sim = SBERTSimilarity()
sentence_similarity = sentence_sim.cosine_similarity(
    '나는 사과를 좋아한다',
    '나는 바나나를 좋아한다'
)
print(f"문장 유사도: {sentence_similarity:.4f}")
```

---

## 유사도 값 해석 가이드

### 코사인 유사도 (Cosine Similarity)

| 값 범위 | 의미 | 예시 |
|---------|------|------|
| 0.8 ~ 1.0 | 매우 유사 | '오늘 날씨가 좋다' ↔ '오늘 날씨가 맑다' |
| 0.5 ~ 0.8 | 유사 | '나는 사과를 좋아한다' ↔ '나는 바나나를 좋아한다' |
| 0.3 ~ 0.5 | 약간 유사 | '고양이는 동물이다' ↔ '강아지는 동물이다' |
| 0.0 ~ 0.3 | 관련 없음 | '고양이는 귀여운 동물이다' ↔ '자동차는 빠르다' |
| -1.0 ~ 0.0 | 반대 의미 | (일반적으로 드뭄) |

### 유클리드 거리 (Euclidean Distance)

| 거리 범위 | 의미 | 예시 |
|-----------|------|------|
| 0 ~ 5 | 매우 유사 | 유사한 의미의 문장/단어 |
| 5 ~ 10 | 유사 | 관련 있는 문장/단어 |
| 10 ~ 20 | 약간 유사 | 약간 관련 있는 문장/단어 |
| 20 이상 | 관련 없음 | 관련 없는 문장/단어 |

**참고:** 유클리드 거리는 모델과 벡터 차원에 따라 절대값이 달라질 수 있습니다. 상대적인 비교에 사용하는 것이 좋습니다.

---

## 주의사항

1. **SBERT 모델 다운로드**: SBERT 모델은 처음 실행 시 자동으로 다운로드됩니다. 인터넷 연결이 필요합니다.

2. **메모리 사용량**: SBERT 모델은 상당한 메모리를 사용합니다 (약 500MB~1GB). 메모리가 부족한 경우 더 작은 모델을 사용하세요.

3. **처리 속도**: 
   - Word2Vec: 매우 빠름 (수 밀리초)
   - SBERT: 상대적으로 느림 (수백 밀리초~수 초), 하지만 더 정확함

4. **언어 지원**: 
   - Word2Vec: 학습 데이터에 따라 다름
   - SBERT: `paraphrase-multilingual-*` 모델은 한국어를 지원합니다

5. **단어/문장이 모델에 없는 경우**: 
   - Word2Vec: `None` 또는 빈 리스트 반환
   - SBERT: 항상 벡터를 생성하지만, 의미가 없는 문장의 경우 정확도가 낮을 수 있습니다

