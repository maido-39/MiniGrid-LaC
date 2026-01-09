"""
OpenCV에서 이모지를 렌더링하는 핵심 함수들
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
from typing import Optional, Tuple


def find_emoji_font(font_size: int = 64) -> Optional[ImageFont.FreeTypeFont]:
    """
    NotoEmoji-Regular.ttf 폰트를 로드
    
    Args:
        font_size: 폰트 크기 (기본값: 64)
        
    Returns:
        ImageFont 객체 또는 None
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(script_dir, 'fonts', 'NotoEmoji-Regular.ttf')
    
    try:
        return ImageFont.truetype(font_path, font_size)
    except:
        return None


def draw_emoji_on_image(
    image: np.ndarray,
    emoji_char: str,
    position: Tuple[int, int],
    font_size: int = 64,
    font: Optional[ImageFont.FreeTypeFont] = None
) -> np.ndarray:
    """
    이미지에 이모지를 그리는 함수
    
    Args:
        image: OpenCV 이미지 (numpy array, shape: (H, W, 3), RGB 형식)
        emoji_char: 그릴 이모지 문자
        position: 이모지를 그릴 위치 (x, y)
        font_size: 폰트 크기 (기본값: 64)
        font: 사용할 폰트 (None이면 자동으로 찾음)
        
    Returns:
        이모지가 그려진 이미지 (numpy array)
    """
    img_copy = image.copy()
    pil_img = Image.fromarray(img_copy.astype(np.uint8)).convert('RGBA')
    draw = ImageDraw.Draw(pil_img)
    
    if font is None:
        font = find_emoji_font(font_size)
    
    x, y = position
    fill_color = (255, 255, 255, 255)  # RGBA: 흰색
    
    if font:
        try:
            draw.text((x, y), emoji_char, font=font, fill=fill_color)
        except:
            try:
                draw.text((x, y), emoji_char, fill=fill_color)
            except:
                pass
    else:
        try:
            draw.text((x, y), emoji_char, fill=fill_color)
        except:
            pass
    
    rgb_img = pil_img.convert('RGB')
    return np.array(rgb_img)


# 간단한 사용 예제
if __name__ == "__main__":
    # 검은 배경 이미지 생성
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # 이모지 그리기
    emoji = '🌲'
    image = draw_emoji_on_image(image, emoji, (50, 50), font_size=100)
    
    # OpenCV로 표시
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Emoji Test', image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
