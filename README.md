# MiniGrid-LaC

MiniGrid 환경에서 Language-conditioned 강화학습을 위한 프로젝트입니다.

## 개요

이 프로젝트는 MiniGrid 환경에서 언어 지시(language instruction)를 활용한 강화학습 에이전트를 구현합니다.

## Project Structure

```
multigrid-LaC/
├── src/                          # Source code directory
│   ├── lib/                      # Core library modules
│   │   ├── map_manager/          # Map and environment management
│   │   │   ├── minigrid_customenv_emoji.py    # Main environment wrapper (emoji support, absolute movement)
│   │   │   └── emoji_map_loader.py            # JSON map loader for emoji-based maps
│   │   └── vlm/                  # Vision Language Model modules
│   │       ├── vlm_wrapper.py                 # VLM API wrapper (OpenAI GPT-4o)
│   │       ├── vlm_postprocessor.py          # VLM response parser and validator
│   │       ├── vlm_controller.py             # Generic VLM controller for environment control
│   │       ├── vlm_manager.py                 # VLM handler manager (multi-provider support)
│   │       └── handlers/                      # VLM provider handlers (OpenAI, Qwen, Gemma, etc.)
│   ├── legacy/                   # Legacy code (maintained for backward compatibility)
│   │   ├── relative_movement/    # Relative movement-based control (deprecated)
│   │   │   └── custom_environment.py          # Legacy environment wrapper
│   │   └── vlm_rels/             # Legacy VLM-related modules
│   │       ├── minigrid_vlm_controller.py     # Legacy MiniGrid-specific VLM controller
│   │       └── minigrid_vlm_helpers.py        # Legacy visualization helpers
│   ├── dev-*/                    # Development branches (experimental features)
│   │   ├── dev-scenario_2/       # Scenario 2 development
│   │   └── dev-action_uncertainty/ # Action uncertainty estimation experiments
│   ├── test_script/              # Test and example scripts
│   │   ├── emoji_test/           # Emoji rendering tests
│   │   ├── keyboard_control/    # Keyboard control examples
│   │   ├── etc/                  # Miscellaneous test scripts
│   │   └── similarity_calculator/ # Text similarity utilities
│   ├── asset/                    # Resource files
│   │   ├── arrow.png             # Robot arrow marker image
│   │   └── fonts/                # Font files for emoji rendering
│   ├── config/                   # Configuration files (moved from root)
│   │   └── example_map.json      # Example emoji map configuration
│   ├── scenario2_test_absolutemove.py  # Main experiment script (absolute movement)
│   └── VLM_interact_minigrid-(absolute,emoji).py  # VLM interaction example
├── config/                        # Configuration files
│   └── example_map.json          # Example emoji map (JSON format)
├── logs/                         # Experiment logs (generated at runtime)
├── docs/                          # Documentation
└── requirements.txt              # Python dependencies
```

### Directory Purposes

- **`src/lib/`**: Core reusable library modules
  - **`map_manager/`**: Environment creation and map loading utilities
  - **`vlm/`**: VLM integration modules for robot control

- **`src/legacy/`**: Legacy code maintained for backward compatibility
  - **`relative_movement/`**: Old relative movement-based control (use `lib.map_manager` instead)
  - **`vlm_rels/`**: Legacy VLM modules (use `lib.vlm` instead)

- **`src/dev-*/`**: Experimental development branches
  - Active development features that may be merged into main library later

- **`src/test_script/`**: Test and example scripts
  - Various test scripts, examples, and utility scripts

- **`src/asset/`**: Static resource files
  - Images, fonts, and other assets used by the environment

- **`config/`**: Configuration files
  - JSON map files and other configuration data

### Import Usage

All modules can be imported using simplified paths thanks to `__init__.py`:

```python
# Simplified imports (recommended)
from lib import MiniGridEmojiWrapper, load_emoji_map_from_json
from lib import ChatGPT4oVLMWrapper, VLMResponsePostProcessor, VLMController
from legacy import CustomRoomWrapper, MiniGridVLMController

# Full paths are also available if needed
# from lib.map_manager.minigrid_customenv_emoji import MiniGridEmojiWrapper
# from lib.vlm.vlm_wrapper import ChatGPT4oVLMWrapper
```

### IDE Support

All major methods in `lib/` have comprehensive docstrings following Google-style conventions. When you hover over a function name in your IDE, you'll see:
- Detailed description of the method's purpose
- Complete parameter documentation with types and defaults
- Return value descriptions
- Usage examples
- Notes and important information

Example:
```python
from lib import MiniGridEmojiWrapper

# Hover over get_image() to see full documentation
wrapper = MiniGridEmojiWrapper(size=10)
image = wrapper.get_image()  # ← Hover here for detailed docs
```

## 문서

프로젝트의 상세한 문서는 [`docs/`](docs/) 폴더에서 확인할 수 있습니다:

### MiniGrid 기초
- [MiniGrid 예제 환경 목록](docs/minigrid-environments.md) - MiniGrid에 존재하는 모든 내장 환경 목록
- [MiniGrid 오브젝트 및 속성](docs/minigrid-objects.md) - MiniGrid에서 사용 가능한 오브젝트 타입과 속성
- [환경 생성 가이드](docs/environment-creation.md) - MiniGrid 환경 생성 방법
- [베스트 프랙티스](docs/best-practices.md) - MiniGrid 환경 생성 권장사항

### API 문서
- [커스텀 환경 API](docs/custom-environment-api.md) - CustomRoomEnv API 문서
- [Wrapper API](docs/wrapper-api.md) - CustomRoomWrapper API 문서 (절대 좌표 이동 포함)
- [Wrapper 메서드 가이드](docs/wrapper-methods.md) - CustomRoomWrapper의 모든 메서드 설명

**Note**: All major methods in `lib/` have comprehensive docstrings. Hover over any method name in your IDE to see detailed documentation including:
- Parameter descriptions with types and defaults
- Return value documentation
- Usage examples
- Important notes and warnings

### 사용 가이드
- [API Key 생성 및 설정 가이드](docs/LLM-API/api-key-setup.md) - OpenAI, Gemini, Vertex AI API Key 설정 방법
- [키보드 제어 가이드](docs/keyboard-control.md) - 키보드 제어 예제 설명
- [VLM 테스트 스크립트 가이드](docs/test-vlm-guide.md) - VLM 모델 테스트 및 비교 가이드
- [이모지 맵 JSON 로더 가이드](docs/emoji-map-loader.md) - JSON 파일에서 이모지 맵 로드하기
- [SLAM 스타일 FOV 맵핑 가이드](docs/slam-fov-mapping.md) - 탐색 영역 추적 및 시야 제한 기능
- [이모지 사용 가이드](docs/EMOJI_USAGE_GUIDE.md) - 이모지 객체 사용하기

## 기능

- MiniGrid 환경 통합
- Language-conditioned 정책 학습
- 강화학습 알고리즘 구현

## 설치

### 필수 요구사항

- Python 3.8 이상
- API 키 (VLM 기능 사용 시):
  - OpenAI API 키 (GPT-4o 등 사용 시)
  - Gemini API 키 (Gemini 모델 사용 시)
  - Vertex AI 설정 (logprobs 기능 사용 시)

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
python scenario2_test_absolutemove.py

# 방법 2: 프로젝트 루트에서 PYTHONPATH 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python src/scenario2_test_absolutemove.py

# 방법 3: Python 코드에서 sys.path 설정
# 스크립트 내부에 다음 코드 추가:
# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
```

### 예제 스크립트

이 프로젝트는 MiniGrid 환경에서 VLM(Vision Language Model)을 활용한 언어 기반 제어를 위한 여러 예제 스크립트를 제공합니다.

#### 1. `test_script/keyboard_control/keyboard_control.py` - 키보드 제어 예제

**설명**: MiniGrid 환경을 키보드로 직접 제어하는 간단한 예제 스크립트입니다. 환경의 기본 동작을 이해하고 테스트하기에 적합합니다.

**기능**:
- 키보드 입력으로 에이전트 제어
- OpenCV를 통한 실시간 환경 시각화
- 환경 리셋 및 종료 기능

**실행 방법**:
```bash
cd src
python test_script/keyboard_control/keyboard_control.py
```

**조작법**:
- `w`: 앞으로 이동 (move forward)
- `a`: 왼쪽으로 회전 (turn left)
- `d`: 오른쪽으로 회전 (turn right)
- `s`: 뒤로 이동 (move backward) - 일부 환경에서만 지원
- `r`: 환경 리셋
- `q`: 종료

**사용 환경**: `MiniGrid-Empty-8x8-v0` (기본 빈 환경)

---

#### 2. `VLM_interact_minigrid-(absolute,emoji).py` - VLM 상호작용 예제

**설명**: VLM을 사용하여 MiniGrid 환경을 제어하는 예제입니다. 절대 좌표 이동과 이모지 맵을 지원합니다.

**기능**:
- VLM을 통한 자동 에이전트 제어
- 절대 좌표 이동 (상/하/좌/우 직접 이동)
- 이모지 맵 지원
- CLI 및 OpenCV 시각화

**실행 방법**:
```bash
# OpenAI API 키 설정 필요
export OPENAI_API_KEY=your-api-key

cd src
python VLM_interact_minigrid-\(absolute,emoji\).py
```

**설정**:
- VLM 모델: `gpt-4o` (코드 상단에서 변경 가능)
- Temperature: `0.0`
- Max Tokens: `1000`

**사용 환경**: 시나리오 2 환경 (파란 기둥 2x2, 보라색 테이블 1x3)

**Mission**: "파란 기둥으로 가서 오른쪽으로 돌고, 테이블 옆에 멈추시오"

---

#### 3. `legacy/relative_movement/scenario2_test.py` - 시나리오 2 실험 (전체 기능, Legacy)

**주의**: 이 스크립트는 Legacy 코드입니다. 새로운 프로젝트는 `scenario2_test_absolutemove.py`를 사용하세요.

**설명**: 시나리오 2 실험 환경에서 VLM을 통한 완전한 제어 시스템입니다. 로깅, 영구 메모리, Grounding 지식, 예측 경로 시각화 등 모든 기능이 포함된 완전한 실험 스크립트입니다.

**기능**:
- VLM을 통한 자동 에이전트 제어
- 시나리오 2 환경 (파란 기둥, 보라색 테이블)
- **영구 메모리 시스템**: 이전 행동 요약 및 진행 상황 추적
- **Grounding 지식 시스템**: 사용자 피드백을 통한 실수 학습 및 누적
- **예측 경로 시각화**: VLM이 예측한 액션 궤적을 CLI 및 OpenCV에 표시
- **종합 로깅**: 이미지, JSON, CSV, VLM I/O 로그 저장
- CLI 및 OpenCV 시각화

**실행 방법**:
```bash
# OpenAI API 키 설정 필요
export OPENAI_API_KEY=your-api-key

cd src
python legacy/relative_movement/scenario2_test.py
```

**설정** (코드 상단에서 변경 가능):
```python
VLM_MODEL = "gpt-4o"  # 사용할 모델명
VLM_TEMPERATURE = 0.0  # 생성 온도
VLM_MAX_TOKENS = 1000  # 최대 토큰 수
ACTION_PREDICTION_COUNT = 5  # VLM이 예측할 액션 개수
```

**사용 환경**: 시나리오 2 환경
- 크기: 10x10
- 파란 기둥: 2x2 Grid (통과불가)
- 보라색 테이블: 1x3 Grid (통과불가)
- 시작점: (1, 8)
- 종료점: (8, 1)

**Mission**: "파란 기둥으로 가서 오른쪽으로 돌고, 테이블 옆에 멈추시오"

**로그 출력**:
- `logs/scenario2_YYYYMMDD_HHMMSS/` 디렉토리에 저장
  - `step_XXXX.png`: 각 스텝의 환경 이미지
  - `experiment_log.json`: 모든 스텝의 JSON 로그 (누적)
  - `vlm_io_log.txt`: VLM 입력/출력 로그 (누적)
  - `experiment_log.csv`: 실험 데이터 CSV (누적)
  - `system_prompt.txt`: System Prompt 전체 내용
  - `permanent_memory.txt`: 영구 메모리 및 Grounding 지식

**특징**:
- **영구 메모리**: 각 스텝에서 VLM이 이전 행동을 요약하고 현재 진행 상황을 업데이트
- **Grounding 지식**: 사용자 피드백이 감지되면 VLM이 실수를 분석하고 교훈을 기록 (누적)
- **예측 궤적**: VLM이 여러 액션을 연속적으로 예측하고, 첫 번째만 실행하며 나머지는 시각화
- **피드백 감지**: 사용자 프롬프트에서 자연어 피드백을 자동 감지하여 Grounding 업데이트

---

#### 4. `scenario2_test_absolutemove.py` - 시나리오 2 실험 (절대 좌표 이동 버전) ⭐ **권장**

**설명**: 시나리오 2 실험 환경에서 절대 좌표 이동을 사용하는 VLM 제어 시스템입니다. JSON 파일에서 맵을 로드하며, 절대 좌표 이동을 통해 더 직관적인 제어가 가능합니다.

**기능**:
- JSON 파일에서 이모지 맵 로드 (`lib.map_manager.emoji_map_loader` 사용)
- 절대 좌표 이동 (상/하/좌/우 직접 이동)
- VLM을 통한 자동 에이전트 제어
- 영구 메모리 시스템 및 Grounding 지식 시스템
- 종합 로깅 (이미지, JSON, CSV, VLM I/O 로그)

**실행 방법**:
```bash
# OpenAI API 키 설정 필요
export OPENAI_API_KEY=your-api-key

cd src
# 기본 맵 파일 사용 (config/example_map.json)
python scenario2_test_absolutemove.py

# 특정 JSON 맵 파일 지정
python scenario2_test_absolutemove.py ../config/example_map.json
```

**설정** (코드 상단에서 변경 가능):
```python
VLM_MODEL = "gpt-4o"
VLM_TEMPERATURE = 0.0
VLM_MAX_TOKENS = 1000
```

**맵 파일 형식**: JSON 파일 (`example_map.json` 참고)
- 이모지로 맵 레이아웃 정의
- 각 이모지의 타입과 속성 정의
- 시작 위치 및 목표 위치 지정

**특징**:
- **절대 좌표 이동**: 로봇 방향과 무관하게 상/하/좌/우로 직접 이동
- **JSON 맵 로드**: 코드 수정 없이 JSON 파일만 변경하여 다양한 맵 생성
- **이모지 지원**: 이모지 객체를 사용한 시각적 맵 표현

**상세 가이드**: [이모지 맵 로더 가이드](docs/emoji-map-loader.md)

---

#### 5. `dev-scenario_2/scenario2_keyboard_control.py` - 시나리오 2 키보드 제어 (절대 좌표 이동)

**설명**: 시나리오 2 환경을 키보드로 직접 제어하는 스크립트입니다. 절대 좌표 이동을 사용하여 더 직관적인 제어가 가능합니다.

**기능**:
- JSON 파일에서 이모지 맵 로드
- 절대 좌표 이동 (w/a/s/d 키로 상/하/좌/우 이동)
- OpenCV 시각화
- 실시간 상태 표시

**실행 방법**:
```bash
cd src
# 기본 맵 파일 사용
python dev-scenario_2/scenario2_keyboard_control.py

# 특정 JSON 맵 파일 지정
python dev-scenario_2/scenario2_keyboard_control.py ../../config/example_map.json
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

**특징**:
- 절대 좌표 이동으로 직관적인 제어
- JSON 맵 파일로 쉽게 맵 변경
- 이모지 객체 지원

---

### 추가 스크립트

#### `test_script/keyboard_control/keyboard_control_fov.py` - 시야 제한 기능 포함

키보드 제어 예제에 시야 제한(FOV, Field of View) 기능이 추가된 버전입니다. MiniGrid 내장 환경을 선택할 수 있습니다.

**실행 방법**:
```bash
cd src
python test_script/keyboard_control/keyboard_control_fov.py
```

**선택 가능한 환경**:
1. FourRooms (4개의 방 구조)
2. MultiRoom-N6 (6개의 방)
3. DoorKey-16x16 (문과 열쇠)
4. KeyCorridorS6R3 (복도와 열쇠)
5. Playground (놀이터)
6. Empty-16x16 (빈 환경)

**추가 조작법**:
- `f`: 시야 제한 토글 (켜기/끄기)
- `+`: 시야 범위 증가
- `-`: 시야 범위 감소

---

#### `test_script/keyboard_control/keyboard_control_fov_mapping.py` - SLAM 스타일 FOV 맵핑

키보드 제어 예제에 SLAM(Simultaneous Localization and Mapping) 스타일의 시야 제한 기능이 추가된 버전입니다.

**실행 방법**:
```bash
cd src
python test_script/keyboard_control/keyboard_control_fov_mapping.py
```

**주요 기능**:
- 탐색한 영역 추적
- 현재 시야 범위 내: 밝게 표시
- 탐색했던 곳 (시야 밖): 어둡게(반투명하게) 표시
- 중요한 객체(열쇠, 문, 목표)가 있는 곳: 탐색했어도 밝게 유지
- 아직 탐색하지 않은 곳: 검은색으로 표시

**조작법**: `keyboard_control_fov.py`와 동일

**상세 가이드**: [SLAM 스타일 FOV 맵핑 가이드](docs/slam-fov-mapping.md)

---

#### 6. `test_script/etc/test_vlm.py` - VLM 모델 테스트 및 비교

**설명**: 다양한 VLM(Vision Language Model) 모델을 테스트하고 비교할 수 있는 스크립트입니다. 이미지, 프롬프트, 모델을 쉽게 변경하여 테스트할 수 있습니다.

**기능**:
- 다양한 VLM 모델 지원 (OpenAI, Qwen, Gemma)
- 유연한 이미지 입력 (URL, 로컬 파일, 자동 생성)
- 명령줄 인터페이스로 이미지와 프롬프트 지정
- 다중 모델 동시 테스트 및 결과 비교

**실행 방법**:
```bash
# minigrid conda 환경 활성화 (필수)
conda activate minigrid

cd src
# 기본 이미지와 기본 프롬프트 사용
python test_script/etc/test_vlm.py

# 로컬 이미지 파일 사용
python test_script/etc/test_vlm.py --image path/to/image.jpg

# URL에서 이미지 다운로드
python test_script/etc/test_vlm.py --image https://picsum.photos/400/300

# 사용자 프롬프트 지정
python test_script/etc/test_vlm.py --prompt "What objects are in this image?"

# 시스템 프롬프트와 사용자 프롬프트 모두 지정
python test_script/etc/test_vlm.py --system "You are an expert image analyst." --prompt "Analyze this image in detail."

# 이미지와 프롬프트 모두 지정
python test_script/etc/test_vlm.py -i path/to/image.jpg --command "Describe the colors in this image"
```

**명령줄 옵션**:
- `--image`, `-i`: 이미지 파일 경로 또는 URL
- `--system-prompt`, `--system`: 시스템 프롬프트
- `--user-prompt`, `--prompt`, `--command`: 사용자 프롬프트/명령어
- `--help`, `-h`: 도움말 메시지 표시

**지원 모델**:
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-4`, `gpt-5`
- **Qwen (로컬)**: `Qwen/Qwen2-VL-2B-Instruct`, `Qwen/Qwen2-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-32B-Instruct` 등
- **Gemma (로컬)**: `google/gemma-2-2b-it`, `google/gemma-2-9b-it`, `google/gemma-2-27b-it`

**설정**:
- 모델 설정은 `test_vlm.py` 파일 내 `TEST_MODELS` 리스트에서 수정 가능
- 기본 이미지 URL과 기본 프롬프트도 파일 상단에서 변경 가능

**상세 가이드**: [VLM 테스트 스크립트 가이드](docs/test-vlm-guide.md)

---

## 빠른 시작 예제

### 간단한 환경 생성 및 제어

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from minigrid import register_minigrid_envs
from lib import MiniGridEmojiWrapper, load_emoji_map_from_json

# MiniGrid 환경 등록
register_minigrid_envs()

# JSON 맵 파일에서 환경 로드
wrapper = load_emoji_map_from_json('../config/example_map.json')

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

from lib import (
    MiniGridEmojiWrapper, 
    ChatGPT4oVLMWrapper, 
    VLMResponsePostProcessor,
    VLMController
)

# 방법 1: VLMController 사용 (권장 - 가장 간단)
from lib import VLMController

# 환경 생성
wrapper = load_emoji_map_from_json('../config/example_map.json')
wrapper.reset()

# VLM 컨트롤러 생성
controller = VLMController(
    env=wrapper,
    model="gpt-4o",
    temperature=0.0
)

# VLM으로 액션 생성 및 실행 (한 번에)
obs, reward, done, truncated, info, vlm_response = controller.step(
    mission="Go to the blue pillar"
)

print(f"Action: {vlm_response['action']}")
print(f"Reasoning: {vlm_response.get('reasoning', 'N/A')}")

# 방법 2: 개별 컴포넌트 사용 (더 세밀한 제어)
wrapper = MiniGridEmojiWrapper(size=10)
wrapper.reset()

vlm = ChatGPT4oVLMWrapper(model="gpt-4o")
postprocessor = VLMResponsePostProcessor(required_fields=["action", "reasoning"])

# 현재 환경 이미지 가져오기
image = wrapper.get_image()

# VLM으로 액션 생성
response_raw = vlm.generate(
    image=image,
    system_prompt="You are a robot controller. Use absolute directions.",
    user_prompt="Move to the goal."
)

# 응답 파싱
response = postprocessor.process(response_raw)
action_str = response['action']

# 액션 실행
obs, reward, done, truncated, info = wrapper.step_absolute(action_str)
```

### 주요 API 사용법

모든 주요 메서드는 IDE에서 hover하면 상세한 문서를 볼 수 있습니다:

```python
from lib import MiniGridEmojiWrapper, VLMController, load_emoji_map_from_json

# 환경 생성
wrapper = load_emoji_map_from_json('../config/example_map.json')
# ↑ Hover to see: Loads emoji map from JSON and creates environment

# 환경 리셋
obs, info = wrapper.reset()
# ↑ Hover to see: Reset environment to initial state

# 절대 방향 이동
obs, reward, done, truncated, info = wrapper.step_absolute("move up")
# ↑ Hover to see: Execute absolute direction action with detailed parameter docs

# 이미지 가져오기
image = wrapper.get_image(fov_range=3, fov_width=2)
# ↑ Hover to see: Get environment image with optional FOV limitations

# 상태 정보
state = wrapper.get_state()
# ↑ Hover to see: Get current environment state information

# VLM 컨트롤러
controller = VLMController(env=wrapper)
# ↑ Hover to see: Initialize VLM controller with detailed parameter docs

# 액션 생성 및 실행
response = controller.generate_action(mission="Reach the goal")
# ↑ Hover to see: Generate action using VLM with examples
```

## 라이선스

MIT License

## 기여

이슈와 Pull Request를 환영합니다!

