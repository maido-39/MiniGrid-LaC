######################################################
#                                                    #
#                        GLOBAL                      #
#                      VARIABLES                     #
#                                                    #
######################################################


""""""




######################################################
#                                                    #
#                      VARIABLES                     #
#                                                    #
######################################################


# VLM Configuration
# VLM_MODEL: 사용할 Vision Language Model 선택
#   OpenAI: "gpt-4o" (기본, 균형), "gpt-4o-mini" (빠름), "gpt-4", "gpt-4-turbo", "gpt-5"
#   Gemini: "gemini-2.5-flash" (빠름, thinking 지원), "gemini-1.5-pro" (정확), "gemini-1.5-flash" (빠름)
#   Vertex AI: "gemini-2.5-flash-vertex" (logprobs 지원)
#   Qwen: "qwen2-vl-2b", "qwen2-vl-7b", "qwen2-vl-72b", "qwen2.5-vl-3b", "qwen2.5-vl-7b", "qwen2.5-vl-32b"
#   Gemma: "gemma-2-2b", "gemma-2-9b", "gemma-2-27b"
VLM_MODEL = "gemini-2.5-flash-vertex"
# VLM_TEMPERATURE: 생성 무작위성 조절 (0.0=결정적, 1.0=무작위, 권장: 0.0)
VLM_TEMPERATURE = 0.5
# VLM_MAX_TOKENS: 최대 출력 토큰 수 (응답 길이 제한)
VLM_MAX_TOKENS = 3000
# VLM_THINKING_BUDGET: Gemini 2.5 Flash thinking 예산 (토큰 단위, None=기본값, 0=비활성)
VLM_THINKING_BUDGET = 0
# LOGPROBS_ENABLED: logprobs 활성화 여부 (Vertex AI Gemini 전용)
#   - True: logprobs 활성 (모델명이 gemini-*-vertex / -logprobs 일 때만 동작)
#   - False: logprobs 비활성 (OpenAI 모델이나 일반 Gemini API에서는 자동 비활성)
LOGPROBS_ENABLED = True
# LOGPROBS_TOPK: logprobs top-k 개수 (각 토큰에 대해 상위 k개 확률 반환, 권장: 5)
LOGPROBS_TOPK = 5
# DEBUG: VLM 디버그 출력 활성화 여부
#   - True: VLM API 호출 시 상세한 디버그 정보 출력 (응답, 토큰 사용량, inference time 등)
#   - False: 디버그 정보 출력 안 함
DEBUG = True

# Map Configuration
# MAP_FILE_NAME: 사용할 맵 파일 이름 (config/ 디렉토리 아래에 있어야 함)
#   예: "example_map.json", "scenario135_example_map.json"
MAP_FILE_NAME = "example_map.json"

# Mission/Task Setup
DEFAULT_INITIAL_MISSION = "Explore your surroundings and gather as much information as possible."
DEFAULT_MISSION = "Pursue your current mission!"

# Path Directory to Prompts Text Files
PROMPT_DIR = "utils/prompts"

# Grounding System Configuration
USE_NEW_GROUNDING_SYSTEM = True  # 새 Grounding 시스템 사용 여부
GROUNDING_GENERATION_MODE = "episode"  # "episode"만 지원 (에피소드 종료 시 일괄 처리)
GROUNDING_SAVE_FORMAT = "both"  # "json" | "txt" | "both"
EPISODE_TERMINATION_KEYWORDS = ["end"]  # Episode 종료 키워드

# Grounding Generation VLM Configuration
# Grounding 생성 전용 모델 설정 (None이면 VLM_MODEL 사용)
GROUNDING_VLM_MODEL = None  # None | "gpt-4o" | "gemini-2.5-flash" | etc.
GROUNDING_VLM_TEMPERATURE = 0.3  # Grounding 생성용 temperature (기본값: 0.3)
GROUNDING_VLM_MAX_TOKENS = 2000  # Grounding 생성용 max_tokens (기본값: 2000)
# Note: Grounding은 에피소드 종료 시 일괄 처리되므로 비동기 옵션 제거

# Grounding 사용 설정
# 사용할 Grounding 파일 경로 (None이면 Grounding 사용 안 함)
# 단일 파일: "logs/grounding/grounding_latest.txt"
# 여러 파일: ["logs/grounding/grounding_latest.txt", "logs/grounding/custom_grounding.txt"] 또는 쉼표로 구분된 문자열
# GROUNDING_FILE_PATH = "logs/scenario2_absolute_example_map_20260123_170550/episode_1_20260123_170552_scenario2_test_absolutemove_modularized/grounding_episode_1.txt,logs/scenario2_absolute_example_map_20260123_165532/episode_1_20260123_165535_scenario2_test_absolutemove_modularized/grounding_episode_1.txt"  # None | str | List[str] | "file1.txt,file2.txt"
GROUNDING_FILE_PATH = None
# 현재 에피소드에서 생성된 Grounding은 다음 에피소드부터 사용 가능 (즉석 적용 안 됨)

# Environment Rendering Configuration
# RENDER_GOAL: Goal (초록색 목표 지점) 렌더링 여부
#   - True: Goal 렌더링 (초록색으로 표시)
#   - False: Goal 렌더링 안 함 (시각적으로 표시되지 않음)
RENDER_GOAL = False

#
ENV_ID = "MyCustomEnv-v0"