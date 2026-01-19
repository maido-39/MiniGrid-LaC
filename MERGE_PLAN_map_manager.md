# Stan Commit → src/lib/map_manager Merge Plan

## 개요
`Stan commit/utils/map_manager/` 폴더의 파일들을 `src/lib/map_manager/`로 merge합니다.

## 파일 목록
1. `emoji_map_loader.py` (378줄 → 503줄)
2. `minigrid_customenv_emoji.py` (1107줄 → 1436줄)

---

## 1. emoji_map_loader.py Merge 계획

### 1.1 주요 차이점 분석

#### Stan 버전의 추가/수정 사항:
1. **terminal_formatting_utils 사용** (tfu.cprint)
   - 경고 메시지에 색상 포맷팅 추가 (206줄)
   - 사용 예제에서 tfu.cprint 사용 (359-375줄)

2. **⬛를 brick emoji object로 변환하는 특별 로직** (240-255줄)
   ```python
   # 1. When you find a black square (⬛), always convert it to a 'brick' emoji object.
   if emoji == '⬛':
       obj_config = {
           'type': 'emoji',
           'pos': (x, y),
           'emoji_name': 'brick',
           'color': 'grey',
           'can_pickup': False,
           'can_overlap': False,
           'use_emoji_color': True
       }
       objects.append(obj_config)
       continue
   ```

3. **비정사각형 맵 허용** (204-209줄)
   - 원본: 정사각형 맵만 허용 (에러 발생)
   - Stan: 비정사각형 맵 허용 (경고만 출력, size = max(rows, cols))

4. **에러 메시지 개선**
   - 원본: 간단한 에러 메시지
   - Stan: 더 자세한 에러 메시지 (row_lengths 정보 포함)

#### 원본에만 있는 기능:
1. **DEFAULT_EMOJI_OBJECTS** (56-99줄)
   - 기본 이모지 객체 정의
   - JSON에 없어도 기본값 사용 가능

2. **🤖/🟥/🎯 마커 자동 처리** (262-318줄)
   - 🤖 또는 🟥: start_pos 자동 설정
   - 🎯: goal_pos 자동 설정
   - 마커를 ⬜️로 자동 교체

3. **더 엄격한 검증**
   - 정사각형 맵 필수
   - 일관성 없는 행 길이에 대한 상세한 에러 메시지

4. **더 나은 문서화**
   - load_emoji_map_from_json 함수에 상세한 docstring

### 1.2 Merge 전략

**기본 원칙**: 원본을 기준으로 하되, Stan의 개선사항을 통합

#### 통합할 Stan의 기능:
1. ✅ **terminal_formatting_utils 사용** (선택적)
   - 경고 메시지에만 사용 (에러는 그대로)
   - import 경로 수정: `utils.prompt_manager.terminal_formatting_utils` → `utils.terminal_formatting_utils`
   - 또는 선택적으로 사용 (utils가 없어도 동작하도록)

2. ✅ **⬛를 brick emoji로 변환하는 로직**
   - 원본의 DEFAULT_EMOJI_OBJECTS와 충돌하지 않도록 주의
   - emoji_objects에 ⬛ 정의가 있으면 그것을 우선 사용
   - 없으면 brick emoji로 변환

3. ⚠️ **비정사각형 맵 허용** (사용자 확인 필요)
   - 원본은 정사각형만 허용
   - Stan은 비정사각형 허용
   - **결정 필요**: 정사각형만 허용할지, 비정사각형도 허용할지

4. ✅ **에러 메시지 개선**
   - row_lengths 정보 포함
   - 더 자세한 에러 메시지

#### 유지할 원본 기능:
1. ✅ DEFAULT_EMOJI_OBJECTS
2. ✅ 🤖/🟥/🎯 마커 자동 처리
3. ✅ 정사각형 맵 검증 (비정사각형 허용 시 제거)
4. ✅ 상세한 docstring

### 1.3 구현 순서
1. 원본 파일 백업
2. terminal_formatting_utils import 추가 (선택적)
3. ⬛ → brick 변환 로직 추가 (조건부)
4. 에러 메시지 개선
5. 비정사각형 맵 처리 (사용자 확인 후)
6. 테스트

---

## 2. minigrid_customenv_emoji.py Merge 계획

### 2.1 주요 차이점 분석

#### 파일 크기:
- Stan: 1107줄
- 원본: 1436줄
- **원본이 329줄 더 많음** → 원본에 더 많은 기능이 있음

#### 예상 차이점:
1. **주석 언어**: Stan은 한국어, 원본은 영어
2. **import 경로**: Stan은 절대 import, 원본은 상대 import
3. **기능 차이**: 원본에만 있는 기능들이 있을 수 있음

### 2.2 Merge 전략

**기본 원칙**: 원본을 기준으로 하되, Stan의 수정사항을 찾아 통합

#### 확인 필요 사항:
1. Stan 버전에서 추가/수정된 기능이 있는지
2. 원본에만 있는 기능이 있는지
3. 코드 로직 차이가 있는지

#### 구현 순서:
1. 두 파일의 주요 함수/클래스 비교
2. Stan 버전의 수정사항 식별
3. 원본에 통합
4. 테스트

---

## 3. 확인 필요 사항

### 3.1 emoji_map_loader.py
1. **비정사각형 맵 허용 여부**
   - [ ] 정사각형만 허용 (원본 방식)
   - [ ] 비정사각형도 허용 (Stan 방식)

2. **terminal_formatting_utils 사용 여부**
   - [ ] 사용 (경고 메시지에 색상 추가)
   - [ ] 사용 안 함 (기본 print 사용)

3. **⬛ → brick 변환 로직**
   - [ ] 항상 변환 (Stan 방식)
   - [ ] emoji_objects 정의 우선 (원본 방식)
   - [ ] 둘 다 지원 (조건부 변환)

### 3.2 minigrid_customenv_emoji.py
1. **상세 비교 필요**
   - 두 파일의 함수/클래스 목록 비교
   - 차이점 식별

---

## 4. 구현 체크리스트

### emoji_map_loader.py
- [ ] 원본 파일 백업
- [ ] terminal_formatting_utils import 추가 (선택적)
- [ ] ⬛ → brick 변환 로직 추가
- [ ] 에러 메시지 개선
- [ ] 비정사각형 맵 처리 (확인 후)
- [ ] import 경로 수정 (상대 import로)
- [ ] 주석 영어로 통일
- [ ] 테스트

### minigrid_customenv_emoji.py
- [ ] 원본 파일 백업
- [ ] 두 파일 상세 비교
- [ ] Stan의 수정사항 식별
- [ ] 원본에 통합
- [ ] import 경로 수정
- [ ] 주석 영어로 통일
- [ ] 테스트

---

## 5. 다음 단계

1. 사용자 확인 받기
2. emoji_map_loader.py부터 구현
3. minigrid_customenv_emoji.py 구현
4. 전체 테스트

