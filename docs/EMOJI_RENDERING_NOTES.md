# OpenCV 이모지 렌더링 코드 설계 노트

## 📋 목차
1. [코드 작성 배경](#코드-작성-배경)
2. [아키텍처 설계](#아키텍처-설계)
3. [핵심 함수 설명](#핵심-함수-설명)
4. [작동 원리](#작동-원리)
5. [참고 자료](#참고-자료)

---

## 코드 작성 배경

### 문제점
OpenCV의 `cv2.putText()` 함수는 **이모지와 같은 유니코드 문자를 직접 지원하지 않습니다**. 
- OpenCV는 기본적으로 ASCII 문자와 제한된 유니코드만 지원
- 이모지는 멀티바이트 유니코드 문자로, OpenCV의 텍스트 렌더링 엔진으로는 처리 불가

### 해결 방법
**PIL/Pillow를 중간 레이어로 사용**하여 이모지를 렌더링한 후, OpenCV 형식으로 변환

```
이모지 문자 → PIL/Pillow (폰트로 렌더링) → NumPy 배열 → OpenCV 형식
```

---

## 아키텍처 설계

### 설계 원칙
1. **단순성**: 복잡한 폰트 탐색 로직 제거, 단일 폰트 사용
2. **명확성**: 각 함수가 하나의 명확한 역할 수행
3. **재사용성**: 다른 프로젝트에서도 쉽게 사용 가능한 독립적인 함수

### 코드 구조
```
emoji_opencv_display.py
├── find_emoji_font()      # 폰트 로드
├── draw_emoji_on_image()  # 이모지 렌더링
└── 예제 코드              # 사용법 데모
```

---

## 핵심 함수 설명

### 1. `find_emoji_font(font_size)`

**목적**: NotoEmoji-Regular.ttf 폰트를 로드

**왜 이렇게 작성했나?**
- **단일 폰트 사용**: 복잡한 폰트 탐색 로직 제거로 코드 단순화
- **로컬 폰트 우선**: `fonts/` 폴더의 폰트를 직접 참조하여 의존성 최소화
- **에러 처리**: 폰트 로드 실패 시 `None` 반환으로 안전한 폴백

```python
def find_emoji_font(font_size: int = 64) -> Optional[ImageFont.FreeTypeFont]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(script_dir, 'fonts', 'NotoEmoji-Regular.ttf')
    
    try:
        return ImageFont.truetype(font_path, font_size)
    except:
        return None
```

**핵심 포인트**:
- `os.path.dirname(os.path.abspath(__file__))`: 스크립트 위치 기준 상대 경로 계산
- `ImageFont.truetype()`: TrueType 폰트 파일을 PIL 폰트 객체로 변환

---

### 2. `draw_emoji_on_image(image, emoji_char, position, font_size, font)`

**목적**: NumPy 배열(OpenCV 이미지)에 이모지를 그리기

**왜 이렇게 작성했나?**

#### 2.1 이미지 형식 변환 (RGB → RGBA)
```python
pil_img = Image.fromarray(img_copy.astype(np.uint8)).convert('RGBA')
```
- **이유**: PIL의 `ImageDraw`는 RGBA 모드를 지원하여 투명도 처리 가능
- **변환 과정**: NumPy 배열 → PIL Image → RGBA 모드

#### 2.2 폰트 자동 로드
```python
if font is None:
    font = find_emoji_font(font_size)
```
- **이유**: 폰트를 미리 로드하지 않아도 함수가 자동으로 처리
- **장점**: 사용 편의성 향상

#### 2.3 다단계 폴백 처리
```python
if font:
    try:
        draw.text((x, y), emoji_char, font=font, fill=fill_color)
    except:
        try:
            draw.text((x, y), emoji_char, fill=fill_color)  # 폰트 없이 시도
        except:
            pass
```
- **이유**: 폰트 로드 실패 시에도 기본 폰트로 시도하여 최대한 렌더링 시도
- **안전성**: 예외 발생 시에도 프로그램이 중단되지 않음

#### 2.4 색상 형식 (RGBA)
```python
fill_color = (255, 255, 255, 255)  # RGBA: 흰색
```
- **이유**: RGBA 모드에서 투명도 채널(Alpha) 포함
- **형식**: (Red, Green, Blue, Alpha), 각 값 0-255

#### 2.5 최종 변환 (RGBA → RGB)
```python
rgb_img = pil_img.convert('RGB')
return np.array(rgb_img)
```
- **이유**: OpenCV는 RGB 형식을 사용하므로 변환 필요
- **변환 과정**: PIL RGBA → PIL RGB → NumPy 배열

---

## 작동 원리

### 전체 프로세스 흐름

```
1. 입력: NumPy 배열 (OpenCV 이미지, RGB 형식)
   ↓
2. NumPy → PIL Image 변환
   image.copy() → Image.fromarray() → convert('RGBA')
   ↓
3. PIL ImageDraw로 이모지 렌더링
   ImageDraw.Draw() → draw.text(emoji_char, font, fill)
   ↓
4. PIL → NumPy 변환
   convert('RGB') → np.array()
   ↓
5. 출력: NumPy 배열 (이모지가 그려진 이미지, RGB 형식)
```

### 데이터 형식 변환 상세

| 단계 | 형식 | 설명 |
|------|------|------|
| 입력 | `np.ndarray` (H, W, 3) | OpenCV RGB 이미지 |
| 중간 | `PIL.Image` (RGBA) | 투명도 지원 이미지 |
| 렌더링 | `PIL.ImageDraw` | 이모지 그리기 |
| 출력 | `np.ndarray` (H, W, 3) | OpenCV RGB 이미지 |

### 왜 PIL을 중간에 사용하나?

1. **이모지 지원**: PIL은 TrueType 폰트를 통한 이모지 렌더링 지원
2. **폰트 관리**: `ImageFont` 모듈로 폰트 로드 및 관리 용이
3. **이미지 처리**: 다양한 이미지 형식 변환 지원
4. **OpenCV 호환**: NumPy 배열로 쉽게 변환 가능

---

## 참고 자료

### 공식 문서
1. **Pillow (PIL) 공식 문서**
   - ImageDraw 모듈: https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html
   - ImageFont 모듈: https://pillow.readthedocs.io/en/stable/reference/ImageFont.html
   - 텍스트 그리기: https://pillow.readthedocs.io/en/stable/handbook/text-anchors.html

2. **OpenCV 공식 문서**
   - 이미지 처리: https://docs.opencv.org/
   - NumPy 연동: https://docs.opencv.org/master/d0/de3/tutorial_py_intro.html

3. **Noto Emoji 폰트**
   - Google Fonts: https://fonts.google.com/noto/specimen/Noto+Emoji
   - GitHub: https://github.com/googlefonts/noto-emoji

### Stack Overflow 참고
- **이모지 렌더링 관련 질문**
  - https://stackoverflow.com/questions/37191008/load-truetype-font-to-opencv
  - OpenCV에서 TrueType 폰트 사용 방법
  - PIL/Pillow를 통한 이모지 렌더링 예제

### 기술 블로그
1. **OpenCV와 PIL 연동**
   - OpenCV 이미지를 PIL로 변환하는 방법
   - 색상 채널 변환 (RGB ↔ BGR)

2. **이모지 렌더링 베스트 프랙티스**
   - 이모지 폰트 선택 가이드
   - 유니코드 이모지 처리 방법

### 코드 설계 패턴
1. **Adapter Pattern**: PIL을 OpenCV와 OpenCV 이미지 형식 사이의 어댑터로 사용
2. **Strategy Pattern**: 폰트 로드 실패 시 폴백 전략 적용
3. **Single Responsibility**: 각 함수가 하나의 명확한 책임만 가짐

---

## 사용 예제

### 기본 사용법
```python
import numpy as np
from emoji_opencv_display import draw_emoji_on_image

# 검은 배경 이미지 생성
image = np.zeros((200, 200, 3), dtype=np.uint8)

# 이모지 그리기
image = draw_emoji_on_image(image, '🌲', (50, 50), font_size=100)
```

### 커스텀 폰트 사용
```python
from emoji_opencv_display import find_emoji_font, draw_emoji_on_image

# 폰트 미리 로드
font = find_emoji_font(font_size=128)

# 큰 이모지 그리기
image = draw_emoji_on_image(image, '🍄', (100, 100), font_size=128, font=font)
```

---

## 주의사항

1. **폰트 파일 위치**: `fonts/NotoEmoji-Regular.ttf` 파일이 스크립트와 같은 디렉토리의 `fonts/` 폴더에 있어야 함
2. **이미지 형식**: 입력 이미지는 RGB 형식 (H, W, 3)의 NumPy 배열이어야 함
3. **폰트 로드 실패**: 폰트를 찾지 못하면 `None`이 반환되며, 기본 폰트로 폴백 시도
4. **성능**: PIL 변환 과정이 있으므로 대량의 이모지를 그릴 때는 성능 고려 필요

---

## 개선 가능한 부분

1. **폰트 캐싱**: 동일한 크기의 폰트를 여러 번 로드하지 않도록 캐싱
2. **배치 렌더링**: 여러 이모지를 한 번에 그리는 함수 추가
3. **색상 옵션**: 이모지 색상을 커스터마이징할 수 있는 옵션 추가
4. **에러 메시지**: 폰트 로드 실패 시 더 명확한 에러 메시지 제공

---

**작성일**: 2025-01-06  
**버전**: 1.0  
**작성자**: AI Assistant

