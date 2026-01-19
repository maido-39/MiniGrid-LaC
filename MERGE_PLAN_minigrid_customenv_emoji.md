# minigrid_customenv_emoji.py Merge 계획

## 파일 정보
- **Stan 버전**: 1107줄
- **원본 버전**: 1436줄
- **유사도**: 71.1%
- **차이**: 329줄 (원본이 더 많음)

---

## 주요 차이점 분석

### 1. 주석 언어
- **Stan**: 한국어 주석
- **원본**: 영어 주석
- **조치**: 원본 유지 (영어)

### 2. Import 차이

#### Stan 버전:
```python
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX  # encode() 내부에서 import
# imagetext_py는 try-except 내부에서 import
```

#### 원본 버전:
```python
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX  # 상단에서 import
import hashlib
from imagetext_py import FontDB, Writer, Paint, TextAlign  # 상단에서 직접 import
```

**조치**: 원본 방식 유지 (상단 import)

### 3. EXTENDED_COLORS (원본에만 있음)

원본에만 있는 기능:
```python
EXTENDED_COLORS = {
    'orange': _base_color_count,
    'brown': _base_color_count + 1,
    'pink': _base_color_count + 2,
    # ... 총 12개 추가 색상
}
COLOR_TO_IDX.update(EXTENDED_COLORS)
```

**조치**: 원본 유지 (추가 색상 지원)

### 4. EMOJI_MAP 확장 (원본이 더 많음)

#### Stan 버전: 12개 이모지
```python
EMOJI_MAP = {
    'tree', 'mushroom', 'flower', 'cat', 'grass', 'rock',
    'box', 'chair', 'apple', 'desktop', 'workstation', 'brick'
}
```

#### 원본 버전: 21개 이모지 (9개 추가)
```python
EMOJI_MAP = {
    # Stan과 동일한 12개 +
    'restroom', 'storage', 'preperation', 'kitchen', 
    'plating', 'dining', 'water', 'broom'
}
```

**조치**: 원본 유지 (더 많은 이모지 지원)

### 5. EmojiObject.encode() 메서드

#### Stan 버전 (단순):
```python
def encode(self):
    from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
    obj_type_idx = OBJECT_TO_IDX['box']
    color_idx = COLOR_TO_IDX[self.color]
    return (obj_type_idx, color_idx, 0)
```

#### 원본 버전 (개선):
```python
def encode(self):
    obj_type_idx = OBJECT_TO_IDX['box']
    color_idx = COLOR_TO_IDX.get(self.color, COLOR_TO_IDX['grey'])
    
    # Use emoji_name hash to create unique state value
    # This ensures different emojis with same color get different encodings
    emoji_hash = int(hashlib.md5(self.emoji_name.encode()).hexdigest()[:8], 16)
    state = emoji_hash % 256
    return (obj_type_idx, color_idx, state)
```

**조치**: 원본 유지 (같은 색상의 다른 이모지 구분 가능)

### 6. EmojiObject.cache_key (원본에만 있음)

원본에만 있는 property:
```python
@property
def cache_key(self):
    """A cache key used for rendering."""
    return (self.type, self.emoji_name, self.color, id(self))
```

**조치**: 원본 유지

### 7. 파일 경로 처리

#### Stan 버전:
```python
script_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
font_path = os.path.join(script_dir, 'fonts', 'NotoEmoji-Regular.ttf')
```

#### 원본 버전:
```python
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # src/ directory
font_path = os.path.join(src_dir, 'asset', 'fonts', 'NotoEmoji-Regular.ttf')
```

**조치**: 원본 유지 (더 정확한 경로, asset/fonts/ 사용)

### 8. imagetext_py Import 처리

#### Stan 버전:
```python
if self.use_emoji_color:
    try:
        from imagetext_py import FontDB, Writer, Paint, TextAlign
        # ...
    except ImportError:
        raise ImportError("imagetext_py is required...")
```

#### 원본 버전:
```python
from imagetext_py import FontDB, Writer, Paint, TextAlign  # 상단에서 import

if self.use_emoji_color:
    # 직접 사용
```

**조치**: 원본 유지 (상단 import)

### 9. 에러 처리

#### Stan 버전:
- imagetext_py 없으면 에러 발생

#### 원본 버전:
- imagetext_py 없으면 에러 발생 (동일)

**조치**: 동일

---

## Merge 전략

### 기본 원칙
**원본을 기준으로 유지** - 원본이 더 많은 기능과 개선사항을 가지고 있음

### 확인 필요 사항
1. Stan 버전에만 있는 기능이나 수정사항이 있는지 확인
2. 코드 로직 차이가 있는지 확인
3. 테스트 필요

### 구현 순서
1. 원본 파일 백업
2. Stan 버전과 원본 비교 (상세)
3. Stan의 수정사항 식별
4. 원본에 통합 (필요시)
5. 주석 영어로 통일 (이미 되어 있음)
6. import 경로 확인
7. 테스트

---

## 체크리스트

- [ ] 원본 파일 백업
- [ ] Stan 버전의 수정사항 식별
- [ ] 원본에 통합할 Stan의 수정사항 확인
- [ ] 주석 언어 확인 (영어)
- [ ] import 경로 확인
- [ ] 파일 경로 처리 확인 (src_dir, asset/fonts/)
- [ ] 테스트

---

## 예상 결과

- 원본의 모든 기능 유지
- EXTENDED_COLORS 유지
- 확장된 EMOJI_MAP 유지
- 개선된 encode() 메서드 유지
- cache_key property 유지
- 정확한 파일 경로 처리 유지

