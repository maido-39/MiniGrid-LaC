# MiniGrid-LaC

MiniGrid 환경에서 Language-conditioned 강화학습을 위한 프로젝트입니다.

## 개요

이 프로젝트는 MiniGrid 환경에서 언어 지시(language instruction)를 활용한 강화학습 에이전트를 구현합니다.

## 문서

프로젝트의 상세한 문서는 [`docs/`](docs/) 폴더에서 확인할 수 있습니다:

### MiniGrid 기초
- [MiniGrid 예제 환경 목록](docs/minigrid-environments.md) - MiniGrid에 존재하는 모든 내장 환경 목록
- [MiniGrid 오브젝트 및 속성](docs/minigrid-objects.md) - MiniGrid에서 사용 가능한 오브젝트 타입과 속성
- [환경 생성 가이드](docs/environment-creation.md) - MiniGrid 환경 생성 방법
- [베스트 프랙티스](docs/best-practices.md) - MiniGrid 환경 생성 권장사항

### API 문서
- [커스텀 환경 API](docs/custom-environment-api.md) - CustomRoomEnv API 문서
- [Wrapper API](docs/wrapper-api.md) - CustomRoomWrapper API 문서
- [Wrapper 메서드 가이드](docs/wrapper-methods.md) - CustomRoomWrapper의 모든 메서드 설명

### 사용 가이드
- [키보드 제어 가이드](docs/keyboard-control.md) - 키보드 제어 예제 설명

## 기능

- MiniGrid 환경 통합
- Language-conditioned 정책 학습
- 강화학습 알고리즘 구현

## 설치

### 필수 요구사항

- Python 3.8 이상
- OpenAI API 키 (VLM 기능 사용 시)

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

# OpenAI API 키 설정 (.env 파일 생성)
echo "OPENAI_API_KEY=your-api-key-here" > .env
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

# OpenAI API 키 설정 (.env 파일 생성)
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 설치 확인

```bash
# Python 버전 확인
python --version  # Python 3.8 이상이어야 함

# 패키지 설치 확인
python -c "import minigrid; import gymnasium; import openai; import cv2; print('All packages installed successfully!')"
```

## 사용법

### 예제 스크립트

이 프로젝트는 MiniGrid 환경에서 VLM(Vision Language Model)을 활용한 언어 기반 제어를 위한 여러 예제 스크립트를 제공합니다.

#### 1. `keyboard_control.py` - 키보드 제어 예제

**설명**: MiniGrid 환경을 키보드로 직접 제어하는 간단한 예제 스크립트입니다. 환경의 기본 동작을 이해하고 테스트하기에 적합합니다.

**기능**:
- 키보드 입력으로 에이전트 제어
- OpenCV를 통한 실시간 환경 시각화
- 환경 리셋 및 종료 기능

**실행 방법**:
```bash
python keyboard_control.py
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

#### 2. `minigrid_vlm_interact.py` - VLM 상호작용 (간소화 버전)

**설명**: VLM을 사용하여 MiniGrid 환경을 제어하는 간소화된 버전입니다. 로깅, 메모리, Grounding 등 복잡한 기능은 제거하고 핵심 기능만 포함하여 VLM 제어의 기본 동작을 이해하기 쉽게 구성되었습니다.

**기능**:
- VLM을 통한 자동 에이전트 제어
- 시나리오 2 환경 (파란 기둥, 보라색 테이블)
- CLI 및 OpenCV 시각화
- 사용자 프롬프트 입력

**실행 방법**:
```bash
# OpenAI API 키 설정 필요
export OPENAI_API_KEY=your-api-key

python minigrid_vlm_interact.py
```

**설정**:
- VLM 모델: `gpt-4o` (코드 상단에서 변경 가능)
- Temperature: `0.0`
- Max Tokens: `1000`

**사용 환경**: 시나리오 2 환경 (파란 기둥 2x2, 보라색 테이블 1x3)

**Mission**: "파란 기둥으로 가서 오른쪽으로 돌고, 테이블 옆에 멈추시오"

---

#### 3. `scenario2_test.py` - 시나리오 2 실험 (전체 기능)

**주의 : 현재 VLM 동작 불안정함!!!!, Grounding,VLM 기능 에러는 안나는데 의도적으로 동작하지 않음!!!**

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

python scenario2_test.py
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

### 추가 스크립트

#### `keyboard_control_fov.py` - 시야 제한 기능 포함

키보드 제어 예제에 시야 제한(FOV, Field of View) 기능이 추가된 버전입니다. MiniGrid 내장 환경을 선택할 수 있습니다.

**실행 방법**:
```bash
python keyboard_control_fov.py
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

## 구조

```
.
├── README.md
├── requirements.txt
├── train.py
├── eval.py
└── ...
```

## 라이선스

MIT License

## 기여

이슈와 Pull Request를 환영합니다!

