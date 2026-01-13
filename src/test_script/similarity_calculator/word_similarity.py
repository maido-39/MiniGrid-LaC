"""
Word2Vec을 사용한 단어 간 유사도 및 거리 계산 모듈

이 모듈은 Word2Vec 모델을 사용하여 단어 간의 의미적 유사도를 계산합니다.
코사인 유사도와 유클리드 거리를 계산할 수 있습니다.
"""

import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings('ignore')


class Word2VecSimilarity:
    """
    Word2Vec을 사용한 단어 간 유사도 계산 클래스
    
    이 클래스는 Word2Vec 모델을 로드하거나 학습시켜서
    단어 간의 의미적 거리를 계산할 수 있게 해줍니다.
    """
    
    def __init__(self, model_path=None, sentences=None):
        """
        Word2VecSimilarity 클래스 초기화
        
        Args:
            model_path (str, optional): 
                사전 학습된 Word2Vec 모델 파일 경로 (.bin 또는 .model)
                예: 'word2vec.model' 또는 'GoogleNews-vectors-negative300.bin'
            sentences (list, optional): 
                모델을 새로 학습시킬 문장 리스트
                예: [['안녕', '하세요'], ['좋은', '아침']]
        """
        self.model = None
        
        # 모델 경로가 제공된 경우, 사전 학습된 모델 로드
        if model_path:
            print(f"사전 학습된 모델 로드 중: {model_path}")
            try:
                # .bin 파일인 경우 (Google News 등)
                if model_path.endswith('.bin'):
                    self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
                else:
                    # .model 파일인 경우
                    self.model = Word2Vec.load(model_path)
                print("모델 로드 완료!")
            except Exception as e:
                print(f"모델 로드 실패: {e}")
                print("새 모델을 학습합니다...")
                self._train_model(sentences)
        
        # 문장이 제공된 경우, 새 모델 학습
        elif sentences:
            print("새 Word2Vec 모델 학습 중...")
            self._train_model(sentences)
        
        # 둘 다 없는 경우, 간단한 예제 데이터로 학습
        else:
            print("예제 데이터로 모델 학습 중...")
            example_sentences = [
                ['사과', '빨간색', '과일'],
                ['바나나', '노란색', '과일'],
                ['자동차', '빨간색', '차량'],
                ['버스', '큰', '차량'],
                ['고양이', '작은', '동물'],
                ['강아지', '귀여운', '동물'],
            ]
            self._train_model(example_sentences)
    
    def _train_model(self, sentences):
        """
        Word2Vec 모델을 학습시키는 내부 메서드
        
        Args:
            sentences (list): 단어로 분리된 문장들의 리스트
                예: [['단어1', '단어2'], ['단어3', '단어4']]
        """
        # Word2Vec 모델 학습
        # - sentences: 학습할 문장 리스트
        # - vector_size: 단어 벡터의 차원 수 (100차원)
        # - window: 주변 단어를 몇 개까지 볼지 (5개)
        # - min_count: 최소 등장 횟수 (1번 이상 등장한 단어만 학습)
        # - workers: 학습에 사용할 CPU 코어 수
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=100,      # 단어 벡터의 차원 (차원이 클수록 더 정확하지만 느림)
            window=5,             # 주변 단어를 5개까지 고려
            min_count=1,         # 최소 1번 이상 등장한 단어만 학습
            workers=4            # 4개의 CPU 코어 사용
        )
        print("모델 학습 완료!")
    
    def get_word_vector(self, word):
        """
        특정 단어의 벡터를 가져오는 메서드
        
        Args:
            word (str): 벡터를 가져올 단어
            
        Returns:
            numpy.ndarray: 단어의 벡터 표현 (100차원 배열)
        """
        try:
            # 모델에서 단어의 벡터를 가져옴
            return self.model.wv[word]
        except KeyError:
            # 단어가 모델에 없는 경우
            print(f"경고: '{word}' 단어가 모델에 없습니다.")
            return None
    
    def cosine_similarity(self, word1, word2):
        """
        두 단어 간의 코사인 유사도를 계산하는 메서드
        
        코사인 유사도는 -1부터 1까지의 값을 가지며,
        - 1에 가까울수록: 두 단어가 매우 유사함
        - 0에 가까울수록: 두 단어가 관련이 없음
        - -1에 가까울수록: 두 단어가 반대 의미
        
        Args:
            word1 (str): 첫 번째 단어
            word2 (str): 두 번째 단어
            
        Returns:
            float: 코사인 유사도 값 (-1 ~ 1)
        """
        # 각 단어의 벡터 가져오기
        vec1 = self.get_word_vector(word1)
        vec2 = self.get_word_vector(word2)
        
        if vec1 is None or vec2 is None:
            return None
        
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
    
    def euclidean_distance(self, word1, word2):
        """
        두 단어 간의 유클리드 거리를 계산하는 메서드
        
        유클리드 거리는 두 벡터 사이의 직선 거리입니다.
        거리가 작을수록 두 단어가 더 유사합니다.
        
        Args:
            word1 (str): 첫 번째 단어
            word2 (str): 두 번째 단어
            
        Returns:
            float: 유클리드 거리 (0 이상의 값, 작을수록 유사)
        """
        # 각 단어의 벡터 가져오기
        vec1 = self.get_word_vector(word1)
        vec2 = self.get_word_vector(word2)
        
        if vec1 is None or vec2 is None:
            return None
        
        # 유클리드 거리 계산 공식:
        # distance = sqrt(sum((vec1[i] - vec2[i])^2))
        # numpy의 linalg.norm을 사용하면 간단하게 계산 가능
        distance = np.linalg.norm(vec1 - vec2)
        return float(distance)
    
    def find_most_similar(self, word, topn=5):
        """
        주어진 단어와 가장 유사한 단어들을 찾는 메서드
        
        Args:
            word (str): 기준 단어
            topn (int): 반환할 유사 단어의 개수 (기본값: 5)
            
        Returns:
            list: (단어, 유사도) 튜플의 리스트
        """
        try:
            # Word2Vec 모델의 most_similar 메서드 사용
            similar_words = self.model.wv.most_similar(word, topn=topn)
            return similar_words
        except KeyError:
            print(f"경고: '{word}' 단어가 모델에 없습니다.")
            return []


def main():
    """
    Word2Vec 유사도 계산 예제 실행 함수
    """
    print("=" * 60)
    print("Word2Vec 단어 간 유사도 계산 예제")
    print("=" * 60)
    print()
    
    # Word2VecSimilarity 객체 생성 (예제 데이터로 학습)
    similarity_calculator = Word2VecSimilarity()
    
    # 테스트할 단어 쌍들
    test_pairs = [
        ('사과', '바나나'),      # 둘 다 과일
        ('사과', '자동차'),      # 관련 없음
        ('고양이', '강아지'),    # 둘 다 동물
        ('자동차', '버스'),      # 둘 다 차량
    ]
    
    print("\n[코사인 유사도 계산]")
    print("-" * 60)
    print(f"{'단어1':<10} {'단어2':<10} {'유사도':<15} {'의미'}")
    print("-" * 60)
    
    for word1, word2 in test_pairs:
        similarity = similarity_calculator.cosine_similarity(word1, word2)
        if similarity is not None:
            # 유사도에 따른 의미 해석
            if similarity > 0.5:
                meaning = "매우 유사"
            elif similarity > 0.2:
                meaning = "유사"
            elif similarity > -0.2:
                meaning = "관련 없음"
            else:
                meaning = "반대"
            
            print(f"{word1:<10} {word2:<10} {similarity:>6.4f}        {meaning}")
    
    print("\n[유클리드 거리 계산]")
    print("-" * 60)
    print(f"{'단어1':<10} {'단어2':<10} {'거리':<15} {'의미'}")
    print("-" * 60)
    
    for word1, word2 in test_pairs:
        distance = similarity_calculator.euclidean_distance(word1, word2)
        if distance is not None:
            # 거리에 따른 의미 해석 (거리가 작을수록 유사)
            if distance < 5:
                meaning = "매우 유사"
            elif distance < 10:
                meaning = "유사"
            else:
                meaning = "관련 없음"
            
            print(f"{word1:<10} {word2:<10} {distance:>6.4f}        {meaning}")
    
    print("\n[가장 유사한 단어 찾기]")
    print("-" * 60)
    test_word = '사과'
    similar_words = similarity_calculator.find_most_similar(test_word, topn=3)
    print(f"'{test_word}'와 가장 유사한 단어들:")
    for word, similarity in similar_words:
        print(f"  - {word}: {similarity:.4f}")


if __name__ == "__main__":
    main()

