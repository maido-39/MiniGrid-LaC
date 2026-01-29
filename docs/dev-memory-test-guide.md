# dev-memory 테스트 스크립트 가이드

프롬프트 파일과 이미지를 넣어 VLM(gemini-2.5-flash, GCP key)을 한 번 돌리고, **반환 JSON**, **memory 블록**, **memory로 렌더된 프롬프트**를 출력해 프롬프트·메모리 개발을 빠르게 할 수 있도록 하는 스크립트입니다.

- **스크립트 위치**: `src/dev-memory/run_memory_dev.py`
- **관련 문서**: [Memory Prompt & Render 가이드](./memory-prompt-render-guide.md) — 메모리 문법(`$memory[키]`) 및 렌더 규칙

---

## 요구사항

- Python 환경 (프로젝트 `requirements.txt` 설치)
- GCP 키: `USE_GCP_KEY=True` 시 Vertex AI 사용 (`GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` 환경 변수).  
  또는 `GOOGLE_APPLICATION_CREDENTIALS`로 서비스 계정 JSON 경로 지정.

---

## 사용법

**실행 위치**: 프로젝트의 `src/` 디렉터리에서 실행합니다.

```bash
cd src
python dev-memory/run_memory_dev.py --prompt system_prompt_start.txt --image path/to/image.png
```

### 인자

| 인자 | 필수 | 설명 |
|------|------|------|
| `-p`, `--prompt` | O | 프롬프트 파일 이름 (`utils/prompts/` 아래, 예: `system_prompt_start.txt`) |
| `-i`, `--image` | O | 입력 이미지 경로 |
| `--user-prompt` | - | 사용자(미션) 프롬프트 (기본: "Go to the target. Select one action.") |
| `--out-json` | - | 파싱된 JSON을 저장할 파일 경로 |
| `--out-rendered` | - | memory로 렌더된 프롬프트를 저장할 파일 경로 |

### 예시

```bash
# 기본 실행
python dev-memory/run_memory_dev.py -p system_prompt_start.txt -i logs_good/run_xxx/step_0001.png

# 사용자 프롬프트 지정
python dev-memory/run_memory_dev.py -p system_prompt_start.txt -i image.png --user-prompt "Go to the restroom."

# JSON과 렌더된 프롬프트를 파일로 저장
python dev-memory/run_memory_dev.py -p system_prompt_start.txt -i image.png --out-json out/parsed.json --out-rendered out/rendered.txt
```

---

## 출력

1. **Parsed JSON**  
   VLM 응답을 파싱한 전체 JSON (action, reasoning, grounding, memory 등).

2. **Memory**  
   위 JSON의 `memory` 블록만 따로 출력.

3. **Rendered prompt**  
   프롬프트 템플릿에서 `$memory[키]` 등이 위 `memory` 값으로 치환된 최종 문자열.

이를 통해 프롬프트 템플릿과 memory 스키마/렌더 결과를 한 번에 확인할 수 있습니다.
