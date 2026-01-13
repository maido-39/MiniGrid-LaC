"""
유사도 계산 통합 예제

Word2Vec과 SBERT를 모두 사용하는 간단한 예제입니다.
"""

from word_similarity import Word2VecSimilarity
from sentence_similarity import SBERTSimilarity


def main():
    """
    Word2Vec과 SBERT를 모두 사용하는 통합 예제
    """
    print("=" * 70)
    print("유사도 계산 통합 예제")
    print("=" * 70)
    print()
    
    # ============================================================
    # 1. Word2Vec 예제 - 단어 간 유사도
    # ============================================================
    print("\n" + "=" * 70)
    print("1. Word2Vec - 단어 간 유사도 계산")
    print("=" * 70)
    
    # Word2VecSimilarity 객체 생성
    word_sim = Word2VecSimilarity()
    
    # 단어 쌍들의 유사도 계산
    word_pairs = [
        ('사과', '바나나'),
        ('고양이', '강아지'),
        ('자동차', '버스'),
    ]
    
    print("\n[단어 간 코사인 유사도]")
    for word1, word2 in word_pairs:
        similarity = word_sim.cosine_similarity(word1, word2)
        print(f"  '{word1}' ↔ '{word2}': {similarity:.4f}")
    
    # ============================================================
    # 2. SBERT 예제 - 문장 간 유사도
    # ============================================================
    print("\n" + "=" * 70)
    print("2. SBERT - 문장 간 유사도 계산")
    print("=" * 70)
    
    # SBERTSimilarity 객체 생성
    sentence_sim = SBERTSimilarity()
    
    # 문장 쌍들의 유사도 계산
    sentence_pairs = [
        ('오늘 날씨가 좋다', '오늘 날씨가 맑다'),
        ('나는 사과를 좋아한다', '나는 바나나를 좋아한다'),
        ('고양이는 귀여운 동물이다', '자동차는 빠르다'),
    ]
    
    print("\n[문장 간 코사인 유사도]")
    for sent1, sent2 in sentence_pairs:
        similarity = sentence_sim.cosine_similarity(sent1, sent2)
        print(f"  '{sent1}' ↔ '{sent2}': {similarity:.4f}")
    
    # ============================================================
    # 3. 실제 활용 예제 - 유사한 문장 찾기
    # ============================================================
    print("\n" + "=" * 70)
    print("3. 실제 활용 예제 - 가장 유사한 문장 찾기")
    print("=" * 70)
    
    query = "나는 프로그래밍을 좋아한다"
    candidates = [
        "코딩은 재미있다",
        "나는 음악을 좋아한다",
        "프로그래밍은 어렵다",
        "나는 개발자가 되고 싶다",
        "오늘 날씨가 좋다"
    ]
    
    print(f"\n기준 문장: '{query}'")
    print("\n후보 문장들:")
    for i, candidate in enumerate(candidates, 1):
        print(f"  {i}. {candidate}")
    
    # 가장 유사한 문장 찾기
    similar_sentences = sentence_sim.find_most_similar(query, candidates, topn=3)
    
    print("\n가장 유사한 문장 Top 3:")
    for i, (sentence, similarity) in enumerate(similar_sentences, 1):
        print(f"  {i}. {sentence} (유사도: {similarity:.4f})")
    
    print("\n" + "=" * 70)
    print("예제 실행 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()

