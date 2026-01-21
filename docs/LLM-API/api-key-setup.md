# API Key 생성 및 설정 가이드

이 문서는 MiniGrid-LaC 프로젝트에서 사용하는 다양한 LLM API의 키 생성 및 설정 방법을 안내합니다.

## 목차

1. [OpenAI API Key](#openai-api-key)
2. [Gemini API Key](#gemini-api-key)
3. [Gemini Vertex AI 설정](#gemini-vertex-ai-설정)
4. [환경 변수 설정 방법](#환경-변수-설정-방법)
5. [문제 해결](#문제-해결)

---

## OpenAI API Key

### 1. API Key 생성

1. **OpenAI 웹사이트 접속**
   - [OpenAI Platform](https://platform.openai.com/)에 접속합니다.
   - 계정이 없다면 회원가입을 진행합니다.

2. **API Keys 페이지 이동**
   - 로그인 후, 좌측 메뉴에서 **"API keys"** 또는 **"API Keys"**를 클릭합니다.
   - 또는 직접 [API Keys 페이지](https://platform.openai.com/api-keys)로 이동합니다.

3. **새 API Key 생성**
   - **"+ Create new secret key"** 버튼을 클릭합니다.
   - Key 이름을 입력합니다 (예: "minigrid-lac-dev").
   - **"Create secret key"** 버튼을 클릭합니다.

4. **API Key 복사**
   - ⚠️ **중요**: API Key는 이 시점에서만 표시됩니다. 반드시 안전한 곳에 복사해 저장하세요.
   - Key 형식: `sk-proj-...` (약 50자 정도의 문자열)

### 2. API Key 설정

#### 방법 1: 환경 변수 설정 (권장)

**Linux/macOS:**
```bash
export OPENAI_API_KEY="sk-proj-your-api-key-here"
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-proj-your-api-key-here"
```

**Windows (CMD):**
```cmd
set OPENAI_API_KEY=sk-proj-your-api-key-here
```

#### 방법 2: .env 파일 사용 (권장)

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 다음 내용을 추가합니다:

```bash
OPENAI_API_KEY=sk-proj-your-api-key-here
```

`.env` 파일은 Git에 커밋하지 않도록 `.gitignore`에 추가되어 있습니다.

#### 방법 3: 코드에서 직접 전달

```python
from utils.vlm.vlm_wrapper import VLMWrapper

wrapper = VLMWrapper(
    model="gpt-4o",
    api_key="sk-proj-your-api-key-here"  # 직접 전달 (보안상 권장하지 않음)
)
```

### 3. 사용 가능한 모델

- `gpt-4o`: GPT-4o 모델 (최신, 권장)
- `gpt-4o-mini`: GPT-4o Mini 모델 (경량, 저렴)
- `gpt-4`: GPT-4 모델
- `gpt-4-turbo`: GPT-4 Turbo 모델

### 4. 비용 정보

- OpenAI API는 사용량 기반 과금입니다.
- 자세한 가격 정보는 [OpenAI Pricing](https://openai.com/api/pricing/)을 참고하세요.
- 무료 크레딧이 제공될 수 있습니다 (신규 계정).

---

## Gemini API Key

### 1. API Key 생성

1. **Google AI Studio 접속**
   - [Google AI Studio](https://aistudio.google.com/)에 접속합니다.
   - Google 계정으로 로그인합니다.

2. **API Key 생성**
   - 좌측 메뉴에서 **"Get API key"** 또는 **"API Keys"**를 클릭합니다.
   - 또는 직접 [API Keys 페이지](https://aistudio.google.com/app/apikey)로 이동합니다.

3. **새 API Key 생성**
   - **"Create API Key"** 버튼을 클릭합니다.
   - Google Cloud 프로젝트를 선택하거나 새로 생성합니다.
   - API Key가 생성되면 복사합니다.

4. **API Key 복사**
   - ⚠️ **중요**: API Key를 안전한 곳에 저장하세요.
   - Key 형식: `AIza...` (약 40자 정도의 문자열)

### 2. API Key 설정

#### 방법 1: 환경 변수 설정 (권장)

**Linux/macOS:**
```bash
export GEMINI_API_KEY="AIza-your-api-key-here"
# 또는
export GOOGLE_API_KEY="AIza-your-api-key-here"
```

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="AIza-your-api-key-here"
# 또는
$env:GOOGLE_API_KEY="AIza-your-api-key-here"
```

#### 방법 2: .env 파일 사용 (권장)

프로젝트 루트 디렉토리의 `.env` 파일에 다음을 추가합니다:

```bash
GEMINI_API_KEY=AIza-your-api-key-here
# 또는
GOOGLE_API_KEY=AIza-your-api-key-here
```

#### 방법 3: 코드에서 직접 전달

```python
from utils.vlm.vlm_wrapper import VLMWrapper

wrapper = VLMWrapper(
    model="gemini-2.5-flash",
    api_key="AIza-your-api-key-here"  # 직접 전달 (보안상 권장하지 않음)
)
```

### 3. 사용 가능한 모델

- `gemini-2.5-flash`: Gemini 2.5 Flash (최신, 빠름)
- `gemini-1.5-pro`: Gemini 1.5 Pro (고성능)
- `gemini-1.5-flash`: Gemini 1.5 Flash (빠름)
- `gemini-pro`: Legacy Gemini Pro
- `gemini-pro-vision`: Legacy Gemini Pro Vision

### 4. 비용 정보

- Gemini API는 무료 티어를 제공합니다 (월 사용량 제한).
- 자세한 가격 정보는 [Gemini API Pricing](https://ai.google.dev/pricing)을 참고하세요.

---

## Gemini Vertex AI 설정

Vertex AI는 Google Cloud Platform의 관리형 AI 서비스입니다. Logprobs 기능을 사용하려면 Vertex AI가 필요합니다.

### 1. 사전 요구사항

- Google Cloud Platform (GCP) 계정
- 활성화된 GCP 프로젝트
- Vertex AI API 활성화
- Service Account 및 JSON 키 파일

### 2. GCP 프로젝트 설정

1. **Google Cloud Console 접속**
   - [Google Cloud Console](https://console.cloud.google.com/)에 접속합니다.
   - 프로젝트를 선택하거나 새로 생성합니다.

2. **Vertex AI API 활성화**
   - 좌측 메뉴에서 **"APIs & Services" > "Library"**로 이동합니다.
   - "Vertex AI API"를 검색하고 **"Enable"**을 클릭합니다.

3. **Service Account 생성**
   - 좌측 메뉴에서 **"IAM & Admin" > "Service Accounts"**로 이동합니다.
   - **"Create Service Account"**를 클릭합니다.
   - Service Account 이름을 입력합니다 (예: "vertex-ai-service").
   - **"Create and Continue"**를 클릭합니다.

4. **역할(Role) 부여**
   - **"Vertex AI User"** 역할을 선택합니다.
   - 또는 **"Cloud Platform"** 전체 권한을 부여할 수 있습니다.
   - **"Continue"**를 클릭합니다.

5. **Service Account 키 생성**
   - 생성된 Service Account를 클릭합니다.
   - **"Keys"** 탭으로 이동합니다.
   - **"Add Key" > "Create new key"**를 클릭합니다.
   - Key type으로 **"JSON"**을 선택합니다.
   - **"Create"**를 클릭하면 JSON 파일이 다운로드됩니다.

### 3. JSON 키 파일 설정

다운로드한 JSON 키 파일을 안전한 위치에 저장합니다:

```bash
# 예시 경로
/path/to/your-project/hello-key-123-12345.json
```

⚠️ **보안 주의사항**:
- JSON 키 파일은 절대 Git에 커밋하지 마세요.
- `.gitignore`에 추가되어 있는지 확인하세요.
- 파일 권한을 제한하세요: `chmod 600 key.json`

### 4. 환경 변수 설정

#### 방법 1: 환경 변수 설정 (권장)

**Linux/macOS:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/key.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"  # 선택사항, 기본값: us-central1
```

**Windows (PowerShell):**
```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\key.json"
$env:GOOGLE_CLOUD_PROJECT="your-project-id"
$env:GOOGLE_CLOUD_LOCATION="us-central1"
```

#### 방법 2: .env 파일 사용 (권장)

프로젝트 루트 디렉토리의 `.env` 파일에 다음을 추가합니다:

```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/key.json
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

#### 방법 3: 코드에서 직접 전달

```python
from utils.vlm.vlm_wrapper import VLMWrapper

wrapper = VLMWrapper(
    model="gemini-2.5-flash-vertex",
    logprobs=5,
    credentials="/path/to/your/key.json",  # 또는 credentials 객체
    project_id="your-project-id",
    location="us-central1"
)
```

### 5. 사용 가능한 모델

- `gemini-2.5-flash-vertex`: Vertex AI를 통한 Gemini 2.5 Flash (logprobs 지원)
- `gemini-2.5-flash-logprobs`: logprobs 지원 모델 (별칭)

### 6. Logprobs 기능

Vertex AI를 사용하면 각 토큰의 확률 분포(logprobs)를 확인할 수 있습니다:

```python
response, logprobs_metadata = wrapper.generate_with_logprobs(
    system_prompt="You are a helpful assistant.",
    user_prompt="What is the capital of France?",
    debug=True
)

# logprobs_metadata에는 다음이 포함됩니다:
# - tokens: 각 토큰 문자열
# - token_logprobs: 각 토큰의 log 확률
# - top_logprobs: 각 위치의 top-k 후보들
# - entropies: 각 토큰의 Shannon entropy
```

### 7. 비용 정보

- Vertex AI는 GCP의 과금 정책을 따릅니다.
- 자세한 가격 정보는 [Vertex AI Pricing](https://cloud.google.com/vertex-ai/pricing)을 참고하세요.
- 무료 크레딧이 제공될 수 있습니다 (신규 GCP 계정).

---

## 환경 변수 설정 방법

### 영구적으로 설정하기

#### Linux/macOS

**~/.bashrc 또는 ~/.zshrc에 추가:**
```bash
# ~/.bashrc 또는 ~/.zshrc 파일에 추가
export OPENAI_API_KEY="sk-proj-your-api-key-here"
export GEMINI_API_KEY="AIza-your-api-key-here"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

변경사항 적용:
```bash
source ~/.bashrc  # 또는 source ~/.zshrc
```

#### Windows

**시스템 환경 변수 설정:**
1. **"시스템 속성" > "고급" > "환경 변수"**로 이동
2. **"사용자 변수"** 또는 **"시스템 변수"**에서 **"새로 만들기"** 클릭
3. 변수 이름과 값을 입력
4. **"확인"** 클릭

### 임시로 설정하기

현재 터미널 세션에서만 유효:

**Linux/macOS:**
```bash
export OPENAI_API_KEY="sk-proj-your-api-key-here"
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-proj-your-api-key-here"
```

### .env 파일 사용 (권장)

프로젝트 루트에 `.env` 파일을 생성:

```bash
# .env 파일
OPENAI_API_KEY=sk-proj-your-api-key-here
GEMINI_API_KEY=AIza-your-api-key-here
GOOGLE_API_KEY=AIza-your-api-key-here  # GEMINI_API_KEY와 동일

# Vertex AI 설정
GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

프로젝트는 `python-dotenv`를 사용하여 자동으로 `.env` 파일을 로드합니다.

---

## 문제 해결

### OpenAI API Key 관련

#### "API key not provided" 오류

**원인**: 환경 변수가 설정되지 않았거나 잘못된 키를 사용했습니다.

**해결 방법**:
1. 환경 변수가 올바르게 설정되었는지 확인:
   ```bash
   echo $OPENAI_API_KEY  # Linux/macOS
   # 또는
   echo $env:OPENAI_API_KEY  # Windows PowerShell
   ```

2. `.env` 파일이 프로젝트 루트에 있는지 확인

3. API Key가 올바른 형식인지 확인 (예: `sk-proj-...`)

#### "Incorrect API key provided" 오류

**원인**: 잘못된 API Key를 사용했습니다.

**해결 방법**:
1. [OpenAI Platform](https://platform.openai.com/api-keys)에서 API Key를 다시 확인
2. Key가 활성화되어 있는지 확인
3. Key에 충분한 크레딧이 있는지 확인

### Gemini API Key 관련

#### "API key not provided" 오류

**해결 방법**:
1. `GEMINI_API_KEY` 또는 `GOOGLE_API_KEY` 환경 변수 확인
2. `.env` 파일 확인
3. API Key 형식 확인 (예: `AIza...`)

#### "API key not valid" 오류

**해결 방법**:
1. [Google AI Studio](https://aistudio.google.com/app/apikey)에서 Key를 다시 확인
2. Key가 활성화되어 있는지 확인
3. 사용량 제한에 도달했는지 확인

### Vertex AI 관련

#### "Credentials not provided" 오류

**해결 방법**:
1. `GOOGLE_APPLICATION_CREDENTIALS` 환경 변수가 올바른 경로를 가리키는지 확인
2. JSON 키 파일이 존재하는지 확인
3. 파일 경로가 절대 경로인지 확인

#### "Project ID not provided" 오류

**해결 방법**:
1. `GOOGLE_CLOUD_PROJECT` 환경 변수 확인
2. JSON 키 파일 내의 `project_id` 필드 확인

#### "Permission denied" 오류

**해결 방법**:
1. Service Account에 **"Vertex AI User"** 역할이 부여되었는지 확인
2. JSON 키 파일의 권한 확인: `chmod 600 key.json`
3. GCP 프로젝트에서 Vertex AI API가 활성화되었는지 확인

#### "503 UNAVAILABLE" 오류

**원인**: Vertex AI 서비스가 일시적으로 과부하 상태입니다.

**해결 방법**:
1. 잠시 후 다시 시도 (자동 재시도 로직이 포함되어 있음)
2. 다른 리전(location)을 시도: `us-central1`, `us-east1`, `asia-northeast1` 등

### 일반적인 문제

#### 환경 변수가 로드되지 않음

**해결 방법**:
1. 터미널을 재시작
2. `.env` 파일이 프로젝트 루트에 있는지 확인
3. `python-dotenv`가 설치되어 있는지 확인:
   ```bash
   pip install python-dotenv
   ```

#### 여러 API Key 충돌

**해결 방법**:
- 각 API는 독립적으로 작동하므로 충돌하지 않습니다.
- 필요한 API Key만 설정하면 됩니다.

---

## 보안 권장사항

1. **API Key 보호**
   - API Key를 코드에 하드코딩하지 마세요.
   - `.env` 파일을 Git에 커밋하지 마세요 (`.gitignore`에 포함되어 있음).
   - 공개 저장소에 Key를 업로드하지 마세요.

2. **Service Account 키 보호**
   - JSON 키 파일을 안전한 위치에 저장하세요.
   - 파일 권한을 제한하세요: `chmod 600 key.json`
   - Git에 커밋하지 마세요.

3. **환경 변수 사용**
   - 가능하면 환경 변수나 `.env` 파일을 사용하세요.
   - 프로덕션 환경에서는 보안 관리 시스템을 사용하세요.

---

## 관련 문서

- [VLM 핸들러 시스템 가이드](../vlm-handlers.md) - 다양한 VLM 모델 사용하기
- [Gemini Thinking 기능 가이드](./gemini-thinking.md) - Gemini Thinking 기능 사용법
- [VLM Wrapper API](../wrapper-api.md) - Wrapper 클래스 API

---

## 추가 도움말

문제가 계속되면 다음을 확인하세요:

1. **프로젝트 문서**: [docs/README.md](../README.md)
2. **예시 코드**: `src/utils/vlm/example/vlm_example_logprobs.py`
3. **공식 문서**:
   - [OpenAI API Documentation](https://platform.openai.com/docs)
   - [Gemini API Documentation](https://ai.google.dev/docs)
   - [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
