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
VLM_TEMPERATURE = 0.0
# VLM_MAX_TOKENS: 최대 출력 토큰 수 (응답 길이 제한)
VLM_MAX_TOKENS = 3000
# VLM_THINKING_BUDGET: Gemini 2.5 Flash thinking 예산 (토큰 단위, None=기본값, 0=비활성)
VLM_THINKING_BUDGET = 2000
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

#
ENV_ID = "MyCustomEnv-v0"