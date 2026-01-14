# MiniGrid-LaC 문서

이 디렉토리에는 MiniGrid-LaC 프로젝트의 문서가 포함되어 있습니다.

## 문서 목록

### MiniGrid 기초
- [MiniGrid 예제 환경 목록](./minigrid-environments.md) - MiniGrid에 존재하는 모든 내장 환경 목록
- [MiniGrid 오브젝트 및 속성](./minigrid-objects.md) - MiniGrid에서 사용 가능한 오브젝트 타입과 속성
- [환경 생성 가이드](./environment-creation.md) - MiniGrid 환경 생성 방법
- [베스트 프랙티스](./best-practices.md) - 공식 튜토리얼 기반 권장사항

### API 문서
- [CustomRoomEnv API](./custom-environment-api.md) - 커스텀 환경 클래스 API (Legacy)
- [CustomRoomWrapper API](./wrapper-api.md) - Wrapper 클래스 API (Legacy, VLM 연동 지원)
- [Wrapper 메서드 가이드](./wrapper-methods.md) - CustomRoomWrapper의 모든 메서드 설명 (Legacy)
- [MiniGridEmojiWrapper API](./wrapper-api.md#절대-좌표-이동-absolute-movement) - 이모지 및 절대 좌표 이동 지원 (권장)
- [Similarity Calculator API](./similarity-calculator-api.md) - Word2Vec 및 SBERT 유사도 계산 API

### 사용 가이드
- [키보드 제어 가이드](./keyboard-control.md) - 키보드로 환경 제어하기
- [VLM 핸들러 시스템 가이드](./vlm-handlers.md) - 다양한 VLM 모델 사용하기 (OpenAI, Qwen, Gemma)
- [이모지 맵 JSON 로더 가이드](./emoji-map-loader.md) - JSON 파일에서 이모지 맵 로드하기
- [SLAM 스타일 FOV 맵핑 가이드](./slam-fov-mapping.md) - 탐색 영역 추적 및 시야 제한 기능
- [이모지 사용 가이드](./EMOJI_USAGE_GUIDE.md) - 이모지 객체 사용하기

## 빠른 시작

### Import 경로

프로젝트는 간소화된 import를 지원합니다:

```python
# 권장: 간소화된 import
from lib import MiniGridEmojiWrapper, load_emoji_map_from_json
from lib import ChatGPT4oVLMWrapper, VLMResponsePostProcessor
from legacy import CustomRoomWrapper  # Legacy 코드

# 전체 경로도 사용 가능
from lib.map_manager.minigrid_customenv_emoji import MiniGridEmojiWrapper
```

### 시작 가이드

1. **환경 생성**: [환경 생성 가이드](./environment-creation.md) 참고
2. **키보드 제어**: [키보드 제어 가이드](./keyboard-control.md) 참고
3. **이모지 맵 사용**: [이모지 맵 JSON 로더 가이드](./emoji-map-loader.md) 참고 (권장)
4. **절대 좌표 이동**: [Wrapper API](./wrapper-api.md#절대-좌표-이동-absolute-movement)의 절대 좌표 이동 섹션 참고
5. **VLM 사용**: [VLM 핸들러 시스템 가이드](./vlm-handlers.md) 참고
6. **Legacy API**: [CustomRoomEnv API](./custom-environment-api.md) 참고 (레거시 코드)

## 참고 자료

- [MiniGrid 공식 문서](https://minigrid.farama.org/)
- [MiniGrid 환경 생성 튜토리얼](https://minigrid.farama.org/content/create_env_tutorial/)

