# 최종 Merge 계획 (사용자 요구사항 반영)

## 요구사항 정리

### 1. ⬛ 문제 수정
**문제**: 최외곽의 ⬛ 이모지가 JSON의 emoji_objects 정의를 무시하고 무조건 brick emoji object로 변환됨

**원인**: Stan 버전 240-255줄에서 ⬛를 만나면 무조건 brick으로 변환하고 continue

**해결**: 
- ⬛를 만났을 때 먼저 emoji_objects에 정의가 있는지 확인
- 정의가 있으면 그 정의를 사용 (원본 방식)
- 정의가 없으면 기본 wall로 처리

### 2. 기타 수정사항
- ✅ DEFAULT_EMOJI_OBJECTS 제거
- ✅ 🟥 제거, 🤖만 시작위치로 사용
- ✅ 정사각형 검증 유지 (비정사각형 허용 안 함)
- ✅ 에러 메시지 개선 (row_lengths 정보 포함)
- ✅ src/lib → src/utils 폴더명 변경
- ✅ Stan commit/utils/prompt_manager를 src/utils로 이동

---

## 구현 계획

### Phase 1: 폴더 구조 변경
1. `src/lib` → `src/utils` 폴더명 변경
2. 모든 import 경로 수정 (`from lib.` → `from utils.`)

### Phase 2: prompt_manager 이동
1. `Stan commit/utils/prompt_manager/` → `src/utils/prompt_manager/` 복사
2. 파일 내 import 경로 수정:
   - `utils.miscellaneous` → `utils.miscellaneous`
   - `utils.prompt_manager` → `utils.prompt_manager` (이미 올바름)

### Phase 3: emoji_map_loader.py 수정
1. DEFAULT_EMOJI_OBJECTS 제거 (56-99줄)
2. 🟥 마커 제거 (🤖만 시작위치로)
3. ⬛ 문제 수정:
   - Stan의 무조건 brick 변환 로직 제거
   - 원본 방식으로 변경: emoji_objects 정의 우선 사용
4. 에러 메시지 개선 (row_lengths 정보 포함)
5. 정사각형 검증 유지

---

## 상세 구현 내용

### emoji_map_loader.py 수정사항

#### 1. DEFAULT_EMOJI_OBJECTS 제거
```python
# 제거할 부분 (56-99줄)
DEFAULT_EMOJI_OBJECTS = { ... }
```

#### 2. _parse_map_data 수정
- 🟥 마커 처리 제거 (🤖만 시작위치로)
- emoji_objects 처리 변경:
  ```python
  # 기존: DEFAULT_EMOJI_OBJECTS와 merge
  # 변경: JSON의 emoji_objects만 사용 (없으면 에러)
  ```

#### 3. _parse_emoji_map 수정
- ⬛ 무조건 brick 변환 로직 제거 (240-255줄)
- 원본 방식으로 변경:
  ```python
  # ⬛를 만났을 때
  if emoji not in self.emoji_objects:
      # 정의가 없으면 경고하고 무시
      print(f"Warning: Undefined emoji '{emoji}'...")
      continue
  
  # 정의가 있으면 그대로 사용
  emoji_def = self.emoji_objects[emoji]
  obj_type = emoji_def.get('type', 'wall')
  # ... 기존 로직
  ```

#### 4. 에러 메시지 개선
```python
# 기존
raise ValueError(f"All rows must have the same length...")

# 개선 (Stan 버전)
raise ValueError(
    f"All rows in the map must have the same length. "
    f"Expected length: {row_lengths[0]}, "
    f"Problem row numbers: {inconsistent_rows} "
    f"(row lengths: {[row_lengths[i] for i in inconsistent_rows]})"
)
```

---

## 체크리스트

### Phase 1: 폴더 구조
- [ ] `src/lib` → `src/utils` 이름 변경
- [ ] 모든 파일의 import 경로 수정
- [ ] `__init__.py` 파일들 업데이트

### Phase 2: prompt_manager
- [ ] `Stan commit/utils/prompt_manager/` → `src/utils/prompt_manager/` 복사
- [ ] 파일 내 import 경로 확인 및 수정
- [ ] `utils/miscellaneous/global_variables.py` 확인 (PROMPT_DIR 등)

### Phase 3: emoji_map_loader.py
- [ ] DEFAULT_EMOJI_OBJECTS 제거
- [ ] 🟥 마커 제거
- [ ] ⬛ 문제 수정 (JSON 정의 우선)
- [ ] 에러 메시지 개선
- [ ] 정사각형 검증 유지
- [ ] 테스트

---

## 주의사항

1. **import 경로**: 모든 `from lib.` → `from utils.`로 변경 필요
2. **의존성**: prompt_manager가 `utils.miscellaneous.global_variables`를 사용하므로 해당 파일도 확인 필요
3. **테스트**: 수정 후 실제 JSON 파일로 테스트 필요

