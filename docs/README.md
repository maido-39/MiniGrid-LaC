# MiniGrid-LaC 문서

이 디렉토리에는 MiniGrid-LaC 프로젝트의 문서가 포함되어 있습니다.

## 문서 목록

### MiniGrid 기초
- [MiniGrid 예제 환경 목록](./minigrid-environments.md) - MiniGrid에 존재하는 모든 내장 환경 목록
- [MiniGrid 오브젝트 및 속성](./minigrid-objects.md) - MiniGrid에서 사용 가능한 오브젝트 타입과 속성
- [환경 생성 가이드](./environment-creation.md) - MiniGrid 환경 생성 방법
- [베스트 프랙티스](./best-practices.md) - 공식 튜토리얼 기반 권장사항

### API 문서
- [Wrapper API](./wrapper-api.md) - MiniGridEmojiWrapper API 문서 (절대 좌표 이동 포함)
- [Wrapper 메서드 가이드](./wrapper-methods.md) - Wrapper의 모든 메서드 설명
- [VLM 핸들러 시스템 가이드](./vlm-handlers.md) - 다양한 VLM 모델 사용하기 (OpenAI, Qwen, Gemma, Gemini)
- [Similarity Calculator API](./similarity-calculator-api.md) - Word2Vec 및 SBERT 유사도 계산 API

### 사용 가이드
- [키보드 제어 가이드](./keyboard-control.md) - 키보드로 환경 제어하기
- [VLM 테스트 스크립트 가이드](./test-vlm-guide.md) - VLM 모델 테스트 및 비교 가이드
- [dev-memory 테스트 스크립트 가이드](./dev-memory-test-guide.md) - 프롬프트 + 이미지로 VLM 호출 후 JSON·memory·렌더된 프롬프트 출력 (프롬프트 개발용)
- [이모지 맵 JSON 로더 가이드](./emoji-map-loader.md) - JSON 파일에서 이모지 맵 로드하기
- [SLAM 스타일 FOV 맵핑 가이드](./slam-fov-mapping.md) - 탐색 영역 추적 및 시야 제한 기능
- [이모지 사용 가이드](./EMOJI_USAGE_GUIDE.md) - 이모지 객체 사용하기
- [Memory Prompt & Render 가이드](./memory-prompt-render-guide.md) - 메모리 문법(`$memory[키]`) 및 렌더 규칙
- [Grounding 지식 시스템 가이드](./grounding-system.md) - Grounding 시스템 상세 설명 ⭐ **신규**
- [Entropy 및 Trust 계산 가이드](./entropy-trust-calculation.md) - VLM action 불확실성 분석
- [VLM Action Uncertainty 가이드](./vlm-action-uncertainty.md) - Action 불확실도 측정 및 시각화

### LLM API 문서
- [API Key 생성 및 설정 가이드](./LLM-API/api-key-setup.md) - OpenAI, Gemini, Vertex AI API Key 설정 방법
- [Gemini Thinking 기능 가이드](./LLM-API/gemini-thinking.md) - Gemini 2.5/3 시리즈의 Thinking 기능 사용법

## 빠른 시작

### Import 경로

프로젝트는 `utils` 경로를 사용합니다:

```python
# 권장: utils 경로 사용
from utils.map_manager.emoji_map_loader import load_emoji_map_from_json
from utils.map_manager.minigrid_customenv_emoji import MiniGridEmojiWrapper
from utils.vlm.vlm_controller import VLMController
from utils.vlm.vlm_wrapper import VLMWrapper
from utils.miscellaneous.scenario_runner import ScenarioExperiment
from utils.miscellaneous.episode_manager import EpisodeManager
from utils.miscellaneous.grounding_file_manager import GroundingFileManager
```

### 시작 가이드

1. **API Key 설정**: [API Key 생성 및 설정 가이드](./LLM-API/api-key-setup.md) 참고 (VLM 사용 전 필수)
2. **환경 생성**: [환경 생성 가이드](./environment-creation.md) 참고
3. **키보드 제어**: [키보드 제어 가이드](./keyboard-control.md) 참고
4. **이모지 맵 사용**: [이모지 맵 JSON 로더 가이드](./emoji-map-loader.md) 참고 (권장)
5. **절대 좌표 이동**: [Wrapper API](./wrapper-api.md#절대-좌표-이동-absolute-movement)의 절대 좌표 이동 섹션 참고
6. **VLM 사용**: [VLM 핸들러 시스템 가이드](./vlm-handlers.md) 참고
7. **Gemini Thinking 기능**: [Gemini Thinking 기능 가이드](./LLM-API/gemini-thinking.md) 참고
8. **Entropy 분석**: [Entropy 및 Trust 계산 가이드](./entropy-trust-calculation.md) 참고

## 주요 기능 문서

### 1. 메인 실행 스크립트

- **`minigrid_lac.py`**: 모듈화된 실험 시스템을 사용하는 메인 엔트리 포인트
  - ScenarioExperiment 클래스 사용
  - Episode 관리 및 Grounding 생성
  - 종합 로깅 시스템

### 2. 실험 스크립트

- **`scenario2_test_entropy_comparison.py`**: Entropy 비교 실험 (Logprobs 기반)
  - 3가지 조건(H(X), H(X|S), H(X|L,S))으로 VLM 호출
  - Trust 값 계산
  - Logprobs 기반 확률 분포 분석

- **`scenario2_test_entropy_comparison_refined_entropy.py`**: Verbalized Entropy 비교 실험 ⭐ **신규**
  - Tian et al. (2023) 기반 Verbalized Confidence 방식
  - Step-wise 확률 분포 (step1/step2/step3) 추출
  - VLM이 직접 출력하는 확률로 Entropy 계산
  - 가중 평균 Entropy 및 Trust 계산

### 3. 핵심 모듈

- **ScenarioExperiment**: 메인 실험 러너 클래스
  - VLM 기반 자동 제어
  - Grounding 지식 시스템
  - Episode 관리
  - 종합 로깅

- **EpisodeManager**: 에피소드 관리
  - 에피소드별 디렉토리 생성
  - 에피소드 JSON 로그 저장
  - 에피소드 메타데이터 관리

- **GroundingFileManager**: Grounding 지식 파일 관리
  - 에피소드별 Grounding 생성
  - JSON/TXT 형식 저장
  - 최신 Grounding 파일 관리
  - 여러 Grounding 파일 병합 지원 (JSON/TXT 혼합 가능)

- **VLM Handlers**: 다양한 VLM 모델 지원
  - OpenAI (GPT-4o, GPT-4o-mini, GPT-4-turbo)
  - Gemini (Gemini 2.5 Flash, Gemini 1.5 Pro/Flash)
  - Qwen (Qwen2-VL, Qwen2.5-VL)
  - Gemma (Gemma-2)

## 설정 파일

### global_variables.py

주요 설정은 `src/utils/miscellaneous/global_variables.py`에서 변경할 수 있습니다:

```python
# VLM Configuration
VLM_MODEL = "gemini-2.5-flash-vertex"  # 사용할 VLM 모델
VLM_TEMPERATURE = 0.5                   # 생성 온도
VLM_MAX_TOKENS = 3000                   # 최대 토큰 수
LOGPROBS_ENABLED = True                 # Logprobs 활성화 여부
LOGPROBS_TOPK = 5                       # Logprobs top-k 개수

# Map Configuration
MAP_FILE_NAME = "example_map.json"      # 기본 맵 파일 이름

# Grounding System Configuration
USE_NEW_GROUNDING_SYSTEM = True         # 새 Grounding 시스템 사용 여부
GROUNDING_GENERATION_MODE = "episode"  # Grounding 생성 모드
GROUNDING_SAVE_FORMAT = "both"         # Grounding 저장 형식 ("json" | "txt" | "both")
GROUNDING_FILE_PATH = "..."            # Grounding 파일 경로 (여러 파일 지원: 쉼표로 구분)
GROUNDING_MERGE_FORMAT = "txt"        # Grounding 병합 형식 ("txt" | "json" | "both")

# Entropy Configuration
USE_VERBALIZED_ENTROPY = True          # Verbalized Entropy 방식 사용 여부
LOGPROBS_ENABLED = True                # Logprobs 활성화 여부
```

## 참고 자료

- [MiniGrid 공식 문서](https://minigrid.farama.org/)
- [MiniGrid 환경 생성 튜토리얼](https://minigrid.farama.org/content/create_env_tutorial/)
- [OpenAI API 문서](https://platform.openai.com/docs)
- [Google Gemini API 문서](https://ai.google.dev/docs)
- [Qwen VLM GitHub](https://github.com/QwenLM/Qwen-VL)
