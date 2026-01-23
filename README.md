# MiniGrid-LaC

MiniGrid 환경에서 Language-conditioned 강화학습을 위한 프로젝트입니다.

## 개요

이 프로젝트는 MiniGrid 환경에서 Vision Language Model (VLM)을 활용한 언어 기반 에이전트 제어 시스템을 구현합니다. VLM을 통해 자연어 명령을 이해하고, 절대 좌표 이동, 이모지 맵, Grounding 지식 시스템, Entropy 기반 불확실성 분석 등 다양한 기능을 제공합니다.

## 주요 기능

- **VLM 기반 자동 제어**: GPT-4o, Gemini, Qwen, Gemma 등 다양한 VLM 모델 지원
- **절대 좌표 이동**: 로봇 방향과 무관하게 상/하/좌/우 직접 이동
- **이모지 맵 시스템**: JSON 파일로 이모지 기반 맵 정의 및 로드
- **Grounding 지식 시스템**: 사용자 피드백을 통한 실수 학습 및 누적
- **Entropy 분석**: VLM의 action 불확실성 정량화 및 Trust 계산
- **Episode 관리**: 에피소드별 로깅 및 Grounding 생성
- **다양한 VLM 핸들러**: OpenAI, Gemini, Qwen, Gemma 통일된 인터페이스
- **Logprobs 지원**: Vertex AI Gemini를 통한 확률 분포 분석

## Project Structure

```
multigrid-LaC/
├── src/                          # Source code directory
│   ├── utils/                    # Core utility modules
│   │   ├── map_manager/          # Map and environment management
│   │   │   ├── minigrid_customenv_emoji.py    # Main environment wrapper (emoji support, absolute movement)
│   │   │   └── emoji_map_loader.py            # JSON map loader for emoji-based maps
│   │   ├── vlm/                  # Vision Language Model modules
│   │   │   ├── vlm_wrapper.py                 # VLM API wrapper (unified interface)
│   │   │   ├── vlm_postprocessor.py          # VLM response parser and validator
│   │   │   ├── vlm_processor.py              # VLM processing logic
│   │   │   ├── vlm_controller.py             # Generic VLM controller for environment control
│   │   │   ├── vlm_manager.py                # VLM handler manager (multi-provider support)
│   │   │   └── handlers/                     # VLM provider handlers
│   │   │       ├── base.py                    # Base handler class
│   │   │       ├── openai_handler.py         # OpenAI GPT-4o handler
│   │   │       ├── gemini_handler.py         # Google Gemini handler
│   │   │       ├── qwen_handler.py           # Qwen VLM handler
│   │   │       └── gemma_handler.py          # Gemma handler
│   │   ├── miscellaneous/                    # Miscellaneous utilities
│   │   │   ├── scenario_runner.py            # ScenarioExperiment class (main experiment runner)
│   │   │   ├── episode_manager.py           # Episode management and logging
│   │   │   ├── grounding_file_manager.py     # Grounding knowledge file management
│   │   │   ├── visualizer.py                # Visualization utilities
│   │   │   ├── global_variables.py          # Global configuration variables
│   │   │   └── safe_minigrid_registration.py # Safe MiniGrid environment registration
│   │   ├── prompt_manager/                   # Prompt management
│   │   │   ├── prompt_organizer.py          # System/user prompt organization
│   │   │   ├── prompt_interp.py             # Prompt interpolation
│   │   │   └── terminal_formatting_utils.py # Terminal formatting utilities
│   │   ├── user_manager/                     # User interaction
│   │   │   └── user_interact.py             # User interaction handler
│   │   ├── prompts/                          # Prompt templates
│   │   │   ├── system_prompt_start.txt      # System prompt template
│   │   │   ├── task_prompt.txt               # Task prompt template
│   │   │   ├── grounding_generation_prompt.txt    # Grounding generation prompt
│   │   │   ├── reflexion_prompt.txt         # Reflexion prompt
│   │   │   └── feedback_prompt.txt          # Feedback prompt
│   │   └── scripts/                          # Utility scripts
│   │       └── json_to_csv_converter.py     # JSON to CSV converter
│   ├── legacy/                   # Legacy code (maintained for backward compatibility)
│   │   ├── scenario2_test_absolutemove.py   # Legacy scenario 2 script
│   │   └── VLM_interact_minigrid-absolute_emoji.py  # Legacy VLM interaction
│   ├── dev-*/                    # Development branches (experimental features)
│   │   ├── dev-scenario_2/       # Scenario 2 development
│   │   └── dev-action_uncertainty/ # Action uncertainty estimation experiments
│   ├── test_script/              # Test and example scripts
│   │   ├── emoji_test/           # Emoji rendering tests
│   │   ├── keyboard_control/    # Keyboard control examples
│   │   ├── action_entropy/      # Action entropy analysis
│   │   ├── etc/                  # Miscellaneous test scripts
│   │   └── similarity_calculator/ # Text similarity utilities
│   ├── asset/                    # Resource files
│   │   ├── arrow.png             # Robot arrow marker image
│   │   └── fonts/                # Font files for emoji rendering
│   ├── config/                   # Configuration files
│   │   ├── example_map.json      # Example emoji map configuration
│   │   ├── scenario135_example_map.json  # Scenario 135 example map
│   │   └── test_pickup_map.json  # Test pickup map
│   ├── minigrid_lac.py          # Main entry point (recommended)
│   ├── scenario2_test_entropy_comparison.py  # Entropy comparison experiment
│   └── scenario2_test_absolutemove_modularized.py  # Modularized scenario 2 script
├── logs/                         # Experiment logs (generated at runtime)
├── docs/                         # Documentation
└── requirements.txt             # Python dependencies
```

### Directory Purposes

- **`src/utils/`**: Core reusable utility modules
  - **`map_manager/`**: Environment creation and map loading utilities
  - **`vlm/`**: VLM integration modules for robot control
  - **`miscellaneous/`**: Experiment runner, episode management, grounding system
  - **`prompt_manager/`**: Prompt organization and formatting
  - **`user_manager/`**: User interaction handling

- **`src/legacy/`**: Legacy code maintained for backward compatibility
  - Old scripts and modules (use new modularized versions instead)

- **`src/dev-*/`**: Experimental development branches
  - Active development features that may be merged into main library later

- **`src/test_script/`**: Test and example scripts
  - Various test scripts, examples, and utility scripts

- **`src/asset/`**: Static resource files
  - Images, fonts, and other assets used by the environment

- **`src/config/`**: Configuration files
  - JSON map files and other configuration data

### Import Usage

All modules can be imported using the `utils` path:

```python
# Recommended imports
from utils.map_manager.emoji_map_loader import load_emoji_map_from_json
from utils.map_manager.minigrid_customenv_emoji import MiniGridEmojiWrapper
from utils.vlm.vlm_controller import VLMController
from utils.vlm.vlm_wrapper import VLMWrapper
from utils.miscellaneous.scenario_runner import ScenarioExperiment
from utils.miscellaneous.episode_manager import EpisodeManager
from utils.miscellaneous.grounding_file_manager import GroundingFileManager
```

## 문서

프로젝트의 상세한 문서는 [`docs/`](docs/) 폴더에서 확인할 수 있습니다:

### MiniGrid 기초
- [MiniGrid 예제 환경 목록](docs/minigrid-environments.md) - MiniGrid에 존재하는 모든 내장 환경 목록
- [MiniGrid 오브젝트 및 속성](docs/minigrid-objects.md) - MiniGrid에서 사용 가능한 오브젝트 타입과 속성
- [환경 생성 가이드](docs/environment-creation.md) - MiniGrid 환경 생성 방법
- [베스트 프랙티스](docs/best-practices.md) - MiniGrid 환경 생성 권장사항

### API 문서
- [Wrapper API](docs/wrapper-api.md) - MiniGridEmojiWrapper API 문서 (절대 좌표 이동 포함)
- [Wrapper 메서드 가이드](docs/wrapper-methods.md) - Wrapper의 모든 메서드 설명
- [VLM 핸들러 시스템 가이드](docs/vlm-handlers.md) - 다양한 VLM 모델 사용하기 (OpenAI, Qwen, Gemma, Gemini)
- [Similarity Calculator API](docs/similarity-calculator-api.md) - Word2Vec 및 SBERT 유사도 계산 API

### 사용 가이드
- [API Key 생성 및 설정 가이드](docs/LLM-API/api-key-setup.md) - OpenAI, Gemini, Vertex AI API Key 설정 방법
- [키보드 제어 가이드](docs/keyboard-control.md) - 키보드 제어 예제 설명
- [VLM 테스트 스크립트 가이드](docs/test-vlm-guide.md) - VLM 모델 테스트 및 비교 가이드
- [이모지 맵 JSON 로더 가이드](docs/emoji-map-loader.md) - JSON 파일에서 이모지 맵 로드하기
- [SLAM 스타일 FOV 맵핑 가이드](docs/slam-fov-mapping.md) - 탐색 영역 추적 및 시야 제한 기능
- [이모지 사용 가이드](docs/EMOJI_USAGE_GUIDE.md) - 이모지 객체 사용하기
- [Entropy 및 Trust 계산 가이드](docs/entropy-trust-calculation.md) - VLM action 불확실성 분석
- [VLM Action Uncertainty 가이드](docs/vlm-action-uncertainty.md) - Action 불확실도 측정 및 시각화

### LLM API 문서
- [Gemini Thinking 기능 가이드](docs/LLM-API/gemini-thinking.md) - Gemini 2.5/3 시리즈의 Thinking 기능 사용법

## 설치

### 필수 요구사항

- Python 3.8 이상 (Python 3.10 권장)
- API 키 (VLM 기능 사용 시):
  - OpenAI API 키 (GPT-4o 등 사용 시)
  - Gemini API 키 (Gemini 모델 사용 시)
  - Vertex AI 설정 (logprobs 기능 사용 시)
  - DashScope API 키 (Qwen 모델 사용 시)

**📖 API Key 설정 방법**: [API Key 생성 및 설정 가이드](docs/LLM-API/api-key-setup.md) 참고

### Conda를 사용한 설치 (권장)

```bash
# 리포지토리 클론
git clone https://github.com/maido-39/MiniGrid-LaC.git
cd MiniGrid-LaC

# Conda 환경 생성 (Python 3.10 권장)
conda create -n minigrid python=3.10 -y
conda activate minigrid

# 의존성 설치
pip install -r requirements.txt

# API 키 설정 (.env 파일 생성)
# 자세한 설정 방법은 docs/LLM-API/api-key-setup.md 참고
echo "OPENAI_API_KEY=your-api-key-here" > .env
echo "GEMINI_API_KEY=your-api-key-here" >> .env  # Gemini 사용 시
echo "DASHSCOPE_API_KEY=your-api-key-here" >> .env  # Qwen 사용 시
```

### pip를 사용한 설치

```bash
# 리포지토리 클론
git clone https://github.com/maido-39/MiniGrid-LaC.git
cd MiniGrid-LaC

# 가상환경 생성 (선택사항이지만 권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt

# API 키 설정 (.env 파일 생성)
# 자세한 설정 방법은 docs/LLM-API/api-key-setup.md 참고
echo "OPENAI_API_KEY=your-api-key-here" > .env
echo "GEMINI_API_KEY=your-api-key-here" >> .env  # Gemini 사용 시
echo "DASHSCOPE_API_KEY=your-api-key-here" >> .env  # Qwen 사용 시
```

### 설치 확인

```bash
# Python 버전 확인
python --version  # Python 3.8 이상이어야 함

# 패키지 설치 확인
python -c "import minigrid; import gymnasium; import openai; import cv2; print('All packages installed successfully!')"
```

## 사용법

### 실행 전 준비

모든 스크립트는 `src/` 디렉토리에서 실행하거나, 프로젝트 루트에서 `PYTHONPATH`를 설정해야 합니다:

```bash
# 방법 1: src/ 디렉토리에서 실행 (권장)
cd src
python minigrid_lac.py

# 방법 2: 프로젝트 루트에서 PYTHONPATH 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python src/minigrid_lac.py
```

### 메인 실행 스크립트

#### `minigrid_lac.py` - 메인 엔트리 포인트 ⭐ **권장**

**설명**: 모듈화된 실험 시스템을 사용하는 메인 실행 스크립트입니다. ScenarioExperiment 클래스를 사용하여 모든 기능을 제공합니다.

**기능**:
- JSON 파일에서 이모지 맵 로드
- 절대 좌표 이동
- VLM을 통한 자동 에이전트 제어
- 영구 메모리 시스템 및 Grounding 지식 시스템
- Episode 관리 및 로깅
- 종합 로깅 (이미지, JSON, CSV, VLM I/O 로그)

**실행 방법**:
```bash
cd src
# 기본 맵 파일 사용 (global_variables.py의 MAP_FILE_NAME 사용)
python minigrid_lac.py

# 특정 JSON 맵 파일 지정
python minigrid_lac.py config/example_map.json

# 도움말 보기
python minigrid_lac.py --help
```

**설정**: `src/utils/miscellaneous/global_variables.py`에서 변경 가능
- `VLM_MODEL`: 사용할 VLM 모델 (기본값: "gemini-2.5-flash-vertex")
- `VLM_TEMPERATURE`: 생성 온도 (기본값: 0.5)
- `VLM_MAX_TOKENS`: 최대 토큰 수 (기본값: 3000)
- `LOGPROBS_ENABLED`: Logprobs 활성화 여부 (기본값: True)
- `MAP_FILE_NAME`: 기본 맵 파일 이름 (기본값: "example_map.json")
- `USE_NEW_GROUNDING_SYSTEM`: 새 Grounding 시스템 사용 여부 (기본값: True)

**로그 출력**:
- `logs/scenario2_absolute_<map_name>_<timestamp>/` 디렉토리에 저장
  - `episode_<N>_<timestamp>_<script_name>/`: 각 에피소드별 디렉토리
    - `step_XXXX.png`: 각 스텝의 환경 이미지
    - `episode_<N>.json`: 에피소드 JSON 로그
    - `grounding_episode_<N>.json`: Grounding 지식 (JSON)
    - `grounding_episode_<N>.txt`: Grounding 지식 (TXT)
  - `grounding/`: 최신 Grounding 파일
    - `grounding_latest.json`: 최신 Grounding (JSON)
    - `grounding_latest.txt`: 최신 Grounding (TXT)
  - `experiment_log.json`: 전체 실험 JSON 로그 (누적)
  - `experiment_log.csv`: 실험 데이터 CSV (누적)

---

#### `scenario2_test_entropy_comparison.py` - Entropy 비교 실험

**설명**: VLM의 action 불확실성을 분석하기 위한 Entropy 비교 실험 스크립트입니다. 3가지 조건(H(X), H(X|S), H(X|L,S))으로 VLM을 호출하여 Trust 값을 계산합니다.

**기능**:
- 3가지 조건으로 동시 VLM 호출
- Entropy 계산 및 Trust 값 계산
- Logprobs 기반 확률 분포 분석
- CSV 로깅 (Entropy 및 Trust 값 포함)

**실행 방법**:
```bash
cd src
# 기본 맵 파일 사용
python scenario2_test_entropy_comparison.py

# 특정 JSON 맵 파일 지정
python scenario2_test_entropy_comparison.py config/scenario135_example_map.json

# 도움말 보기
python scenario2_test_entropy_comparison.py --help
```

**요구사항**:
- `LOGPROBS_ENABLED = True` (global_variables.py)
- Vertex AI Gemini 모델 사용 (logprobs 지원)

**상세 가이드**: [Entropy 및 Trust 계산 가이드](docs/entropy-trust-calculation.md)

---

### 예제 스크립트

#### 1. `test_script/keyboard_control/keyboard_control.py` - 키보드 제어 예제

**설명**: MiniGrid 환경을 키보드로 직접 제어하는 간단한 예제 스크립트입니다.

**실행 방법**:
```bash
cd src
python test_script/keyboard_control/keyboard_control.py
```

**조작법**:
- `w`: 앞으로 이동 (move forward)
- `a`: 왼쪽으로 회전 (turn left)
- `d`: 오른쪽으로 회전 (turn right)
- `s`: 뒤로 이동 (move backward)
- `r`: 환경 리셋
- `q`: 종료

---

#### 2. `dev-scenario_2/scenario2_keyboard_control.py` - 시나리오 2 키보드 제어 (절대 좌표 이동)

**설명**: 시나리오 2 환경을 키보드로 직접 제어하는 스크립트입니다. 절대 좌표 이동을 사용합니다.

**실행 방법**:
```bash
cd src
python dev-scenario_2/scenario2_keyboard_control.py
```

**조작법**:
- `w`: 위로 이동 (North)
- `s`: 아래로 이동 (South)
- `a`: 왼쪽으로 이동 (West)
- `d`: 오른쪽으로 이동 (East)
- `p`: pickup
- `x`: drop
- `t`: toggle
- `r`: 환경 리셋
- `q`: 종료

---

#### 3. `test_script/etc/test_vlm.py` - VLM 모델 테스트 및 비교

**설명**: 다양한 VLM 모델을 테스트하고 비교할 수 있는 스크립트입니다.

**실행 방법**:
```bash
cd src
# 기본 이미지와 기본 프롬프트 사용
python test_script/etc/test_vlm.py

# 로컬 이미지 파일 사용
python test_script/etc/test_vlm.py --image path/to/image.jpg

# 사용자 프롬프트 지정
python test_script/etc/test_vlm.py --prompt "What objects are in this image?"
```

**상세 가이드**: [VLM 테스트 스크립트 가이드](docs/test-vlm-guide.md)

---

## 빠른 시작 예제

### 간단한 환경 생성 및 제어

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.map_manager.emoji_map_loader import load_emoji_map_from_json
from utils.map_manager.minigrid_customenv_emoji import MiniGridEmojiWrapper
from utils.miscellaneous.safe_minigrid_registration import safe_minigrid_reg

# MiniGrid 환경 등록
safe_minigrid_reg()

# JSON 맵 파일에서 환경 로드
wrapper = load_emoji_map_from_json('config/example_map.json')

# 환경 리셋
obs, info = wrapper.reset()

# 절대 좌표 이동 (상/하/좌/우)
obs, reward, done, truncated, info = wrapper.step_absolute('move up')    # 위로 이동
obs, reward, done, truncated, info = wrapper.step_absolute('move right') # 오른쪽으로 이동
obs, reward, done, truncated, info = wrapper.step_absolute(0)            # 위로 이동 (인덱스)
obs, reward, done, truncated, info = wrapper.step_absolute('north')       # 위로 이동 (별칭)

# 현재 상태 확인
state = wrapper.get_state()
print(f"Agent position: {state['agent_pos']}")
print(f"Agent direction: {state['agent_dir']}")

# 환경 이미지 가져오기 (VLM 입력용)
image = wrapper.get_image()
print(f"Image shape: {image.shape}")  # (height, width, 3)
```

### VLM을 사용한 자동 제어

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.map_manager.emoji_map_loader import load_emoji_map_from_json
from utils.vlm.vlm_controller import VLMController
from utils.miscellaneous.safe_minigrid_registration import safe_minigrid_reg

# MiniGrid 환경 등록
safe_minigrid_reg()

# 환경 생성
wrapper = load_emoji_map_from_json('config/example_map.json')
wrapper.reset()

# VLM 컨트롤러 생성
controller = VLMController(
    env=wrapper,
    model="gpt-4o",
    temperature=0.0
)

# VLM으로 액션 생성 및 실행
obs, reward, done, truncated, info, vlm_response = controller.step(
    mission="Go to the blue pillar"
)

print(f"Action: {vlm_response['action']}")
print(f"Reasoning: {vlm_response.get('reasoning', 'N/A')}")
```

### ScenarioExperiment를 사용한 완전한 실험

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.miscellaneous.scenario_runner import ScenarioExperiment
from utils.miscellaneous.safe_minigrid_registration import safe_minigrid_reg

# MiniGrid 환경 등록
safe_minigrid_reg()

# 실험 생성 및 실행
experiment = ScenarioExperiment(
    json_map_path="config/example_map.json"
)
experiment.run()
```

## 주요 기능 상세

### 1. 절대 좌표 이동

로봇의 현재 방향과 무관하게 상/하/좌/우로 직접 이동할 수 있습니다:

```python
wrapper.step_absolute('move up')      # 위로
wrapper.step_absolute('move down')    # 아래로
wrapper.step_absolute('move left')    # 왼쪽으로
wrapper.step_absolute('move right')   # 오른쪽으로
```

### 2. 이모지 맵 시스템

JSON 파일로 이모지 기반 맵을 정의하고 로드할 수 있습니다:

```json
{
  "size": 10,
  "map": [
    "🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦",
    "🟦🟫🟫🟫🟫🟫🟫🟫🟫🟦",
    "🟦🟫🟫🟫🟫🟫🟫🟫🟫🟦",
    ...
  ],
  "objects": {
    "🟦": {"type": "wall", "color": "blue"},
    "🟫": {"type": "floor", "color": "brown"}
  }
}
```

### 3. Grounding 지식 시스템

사용자 피드백을 통해 실수를 학습하고 누적합니다:

- 에피소드 종료 시 자동 Grounding 생성
- JSON/TXT 형식으로 저장
- 다음 에피소드부터 자동 적용

### 4. Entropy 및 Trust 계산

VLM의 action 불확실성을 정량화합니다:

- **H(X)**: Language Instruction과 Grounding 없이의 엔트로피
- **H(X|S)**: Grounding만 제공했을 때의 엔트로피
- **H(X|L,S)**: Grounding과 Language Instruction 모두 제공했을 때의 엔트로피
- **Trust T**: `(H(X) - H(X|S)) / (H(X) - H(X|L,S))`

### 5. Episode 관리

에피소드별로 로그를 관리하고 Grounding을 생성합니다:

- 각 에피소드별 디렉토리 생성
- 에피소드 JSON 로그 저장
- Grounding 파일 자동 생성

## 라이선스

MIT License

## 기여

이슈와 Pull Request를 환영합니다!
