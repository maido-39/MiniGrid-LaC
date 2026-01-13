"""
SBERT (Sentence-BERT)를 사용한 문장 간 유사도 및 거리 계산 모듈

이 모듈은 SBERT 모델을 사용하여 문장 간의 의미적 유사도를 계산합니다.
코사인 유사도와 유클리드 거리를 계산할 수 있습니다.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')


class SBERTSimilarity:
    """
    SBERT를 사용한 문장 간 유사도 계산 클래스
    
    SBERT는 문장 전체를 하나의 벡터로 변환하여
    문장 간의 의미적 유사도를 계산할 수 있게 해줍니다.
    """
    
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        """
        SBERTSimilarity 클래스 초기화
        
        Args:
            model_name (str): 사용할 SBERT 모델 이름
                - 'paraphrase-multilingual-MiniLM-L12-v2': 다국어 지원 모델 (한국어 포함)
                - 'sentence-transformers/all-MiniLM-L6-v2': 영어 전용 모델 (더 빠름)
                - 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2': 더 정확한 다국어 모델
        """
        print(f"SBERT 모델 로드 중: {model_name}")
        print("(처음 실행 시 모델을 다운로드하므로 시간이 걸릴 수 있습니다...)")
        
        # SentenceTransformer 모델 로드
        # 이 모델은 문장을 벡터로 변환해줍니다
        self.model = SentenceTransformer(model_name)
        
        print("모델 로드 완료!")
    
    def get_sentence_vector(self, sentence):
        """
        특정 문장의 벡터를 가져오는 메서드
        
        Args:
            sentence (str): 벡터로 변환할 문장
            
        Returns:
            numpy.ndarray: 문장의 벡터 표현 (384차원 또는 모델에 따라 다름)
        """
        # 모델을 사용하여 문장을 벡터로 인코딩
        # encode 메서드는 문장을 받아서 숫자 벡터로 변환합니다
        vector = self.model.encode(sentence, convert_to_numpy=True)
        return vector
    
    def get_sentences_vectors(self, sentences):
        """
        여러 문장의 벡터를 한 번에 가져오는 메서드 (더 효율적)
        
        Args:
            sentences (list): 벡터로 변환할 문장들의 리스트
                예: ['문장1', '문장2', '문장3']
            
        Returns:
            numpy.ndarray: 문장 벡터들의 배열 (문장 수 x 벡터 차원)
        """
        # 여러 문장을 한 번에 인코딩 (배치 처리로 더 빠름)
        vectors = self.model.encode(sentences, convert_to_numpy=True)
        return vectors
    
    def cosine_similarity(self, sentence1, sentence2):
        """
        두 문장 간의 코사인 유사도를 계산하는 메서드
        
        코사인 유사도는 -1부터 1까지의 값을 가지며,
        - 1에 가까울수록: 두 문장이 매우 유사함 (의미가 같음)
        - 0에 가까울수록: 두 문장이 관련이 없음
        - -1에 가까울수록: 두 문장이 반대 의미
        
        Args:
            sentence1 (str): 첫 번째 문장
            sentence2 (str): 두 번째 문장
            
        Returns:
            float: 코사인 유사도 값 (-1 ~ 1)
        """
        # 각 문장의 벡터 가져오기
        vec1 = self.get_sentence_vector(sentence1)
        vec2 = self.get_sentence_vector(sentence2)
        
        # 코사인 유사도 계산 공식:
        # cos(θ) = (A · B) / (||A|| * ||B||)
        # 여기서 A · B는 내적(dot product), ||A||는 벡터의 크기(norm)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # 0으로 나누는 것을 방지
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def euclidean_distance(self, sentence1, sentence2):
        """
        두 문장 간의 유클리드 거리를 계산하는 메서드
        
        유클리드 거리는 두 벡터 사이의 직선 거리입니다.
        거리가 작을수록 두 문장이 더 유사합니다.
        
        Args:
            sentence1 (str): 첫 번째 문장
            sentence2 (str): 두 번째 문장
            
        Returns:
            float: 유클리드 거리 (0 이상의 값, 작을수록 유사)
        """
        # 각 문장의 벡터 가져오기
        vec1 = self.get_sentence_vector(sentence1)
        vec2 = self.get_sentence_vector(sentence2)
        
        # 유클리드 거리 계산 공식:
        # distance = sqrt(sum((vec1[i] - vec2[i])^2))
        # numpy의 linalg.norm을 사용하면 간단하게 계산 가능
        distance = np.linalg.norm(vec1 - vec2)
        return float(distance)
    
    def batch_cosine_similarity(self, sentences):
        """
        여러 문장들 간의 코사인 유사도를 한 번에 계산하는 메서드
        
        Args:
            sentences (list): 유사도를 계산할 문장들의 리스트
                예: ['문장1', '문장2', '문장3']
            
        Returns:
            numpy.ndarray: 유사도 행렬 (N x N, N은 문장 개수)
                similarity_matrix[i][j]는 sentences[i]와 sentences[j]의 유사도
        """
        # 모든 문장을 한 번에 벡터로 변환 (효율적)
        vectors = self.get_sentences_vectors(sentences)
        
        # 벡터 정규화 (크기를 1로 만듦)
        # 정규화하면 내적만으로 코사인 유사도를 계산할 수 있음
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized_vectors = vectors / norms
        
        # 모든 문장 쌍에 대한 코사인 유사도 계산
        # 행렬 곱셈을 사용하여 한 번에 계산 (매우 효율적)
        similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
        
        return similarity_matrix
    
    def find_most_similar(self, query_sentence, candidate_sentences, topn=3):
        """
        주어진 문장과 가장 유사한 문장들을 찾는 메서드
        
        Args:
            query_sentence (str): 기준 문장
            candidate_sentences (list): 후보 문장들의 리스트
            topn (int): 반환할 유사 문장의 개수 (기본값: 3)
            
        Returns:
            list: (문장, 유사도) 튜플의 리스트, 유사도 순으로 정렬됨
        """
        # 기준 문장의 벡터
        query_vector = self.get_sentence_vector(query_sentence)
        
        # 후보 문장들의 벡터 (한 번에 계산)
        candidate_vectors = self.get_sentences_vectors(candidate_sentences)
        
        # 각 후보 문장과의 코사인 유사도 계산
        similarities = []
        query_norm = np.linalg.norm(query_vector)
        
        for i, candidate_vector in enumerate(candidate_vectors):
            dot_product = np.dot(query_vector, candidate_vector)
            candidate_norm = np.linalg.norm(candidate_vector)
            
            if query_norm == 0 or candidate_norm == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (query_norm * candidate_norm)
            
            similarities.append((candidate_sentences[i], float(similarity)))
        
        # 유사도 순으로 정렬 (내림차순)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 topn개만 반환
        return similarities[:topn]


def main():
    """
    SBERT 문장 유사도 계산 예제 실행 함수
    """
    print("=" * 60)
    print("SBERT 문장 간 유사도 계산 예제")
    print("=" * 60)
    print()
    
    # SBERTSimilarity 객체 생성
    similarity_calculator = SBERTSimilarity()
    
    # 테스트할 문장 쌍들
    test_pairs = [
        ('오늘 날씨가 좋다', '오늘 날씨가 맑다'),           # 거의 같은 의미
        ('나는 사과를 좋아한다', '나는 바나나를 좋아한다'),   # 유사한 구조, 다른 내용
        ('고양이는 귀여운 동물이다', '자동차는 빠르다'),      # 관련 없음
        ('파이썬은 프로그래밍 언어다', 'Python is a programming language'),  # 다른 언어, 같은 의미
    ]
    
    print("\n[코사인 유사도 계산]")
    print("-" * 60)
    print(f"{'문장1':<30} {'문장2':<30} {'유사도':<10} {'의미'}")
    print("-" * 60)
    
    for sent1, sent2 in test_pairs:
        similarity = similarity_calculator.cosine_similarity(sent1, sent2)
        
        # 유사도에 따른 의미 해석
        if similarity > 0.7:
            meaning = "매우 유사"
        elif similarity > 0.5:
            meaning = "유사"
        elif similarity > 0.3:
            meaning = "약간 유사"
        else:
            meaning = "관련 없음"
        
        # 문장이 너무 길면 잘라서 표시
        sent1_display = sent1[:28] + ".." if len(sent1) > 30 else sent1
        sent2_display = sent2[:28] + ".." if len(sent2) > 30 else sent2
        
        print(f"{sent1_display:<30} {sent2_display:<30} {similarity:>6.4f}   {meaning}")
    
    print("\n[유클리드 거리 계산]")
    print("-" * 60)
    print(f"{'문장1':<30} {'문장2':<30} {'거리':<10} {'의미'}")
    print("-" * 60)
    
    for sent1, sent2 in test_pairs:
        distance = similarity_calculator.euclidean_distance(sent1, sent2)
        
        # 거리에 따른 의미 해석 (거리가 작을수록 유사)
        if distance < 5:
            meaning = "매우 유사"
        elif distance < 10:
            meaning = "유사"
        elif distance < 20:
            meaning = "약간 유사"
        else:
            meaning = "관련 없음"
        
        sent1_display = sent1[:28] + ".." if len(sent1) > 30 else sent1
        sent2_display = sent2[:28] + ".." if len(sent2) > 30 else sent2
        
        print(f"{sent1_display:<30} {sent2_display:<30} {distance:>6.4f}   {meaning}")
    
    print("\n[가장 유사한 문장 찾기]")
    print("-" * 60)
    query = "나는 프로그래밍을 좋아한다"
    candidates = [
        "코딩은 재미있다",
        "나는 음악을 좋아한다",
        "프로그래밍은 어렵다",
        "나는 개발자가 되고 싶다",
        "오늘 날씨가 좋다"
    ]
    
    print(f"기준 문장: '{query}'")
    print(f"\n후보 문장들:")
    for i, candidate in enumerate(candidates, 1):
        print(f"  {i}. {candidate}")
    
    similar_sentences = similarity_calculator.find_most_similar(query, candidates, topn=3)
    
    print(f"\n가장 유사한 문장들:")
    for i, (sentence, similarity) in enumerate(similar_sentences, 1):
        print(f"  {i}. {sentence} (유사도: {similarity:.4f})")
    
    print("\n[배치 유사도 계산]")
    print("-" * 60)
    sentences = [
        "나는 사과를 좋아한다",
        "나는 바나나를 좋아한다",
        "고양이는 귀여운 동물이다",
        "강아지는 친절한 동물이다",
        "자동차는 빠르다"
    ]
    
    similarity_matrix = similarity_calculator.batch_cosine_similarity(sentences)
    
    print("문장들 간의 유사도 행렬:")
    print(f"{'':<25}", end="")
    for i, sent in enumerate(sentences):
        print(f"{i+1:<8}", end="")
    print()
    
    for i, sent in enumerate(sentences):
        sent_display = sent[:23] + ".." if len(sent) > 25 else sent
        print(f"{sent_display:<25}", end="")
        for j in range(len(sentences)):
            print(f"{similarity_matrix[i][j]:>7.3f} ", end="")
        print()


if __name__ == "__main__":
    main()

