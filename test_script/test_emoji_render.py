"""
이모지 cv2 렌더링 테스트 코드
NotoColorEmoji-Regular.ttf를 사용하여 이모지를 렌더링하고 OpenCV로 표시
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# 이모지 이름과 실제 이모지 문자 매핑
EMOJI_MAP = {
    'tree': '🌲',
    'mushroom': '🍄',
    'flower': '🌼',
    'cat': '🐈',
    'grass': '🌾',
    'rock': '🗿',
    'box': '📦',
    'chair': '🪑',
    'apple': '🍎',
    'desktop': '🖥️',
    'workstation': '📱',
    'brick': '🧱',
}


def render_emoji_cv2(emoji_char: str, size: int = 64, use_emoji_color: bool = True) -> np.ndarray:
    """
    이모지를 cv2 형식(numpy array)으로 렌더링
    
    Args:
        emoji_char: 렌더링할 이모지 문자
        size: 이미지 크기 (정사각형)
        use_emoji_color: True면 원래 이모지 색상 사용, False면 흑백
    
    Returns:
        BGR 형식의 numpy array (cv2에서 사용 가능)
    """
    # 빈 이미지 생성 (RGBA)
    img = np.zeros((size, size, 4), dtype=np.uint8)
    font_size = int(size * 0.8)
    
    # 폰트 로드
    font = None
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if use_emoji_color:
            color_font_path = os.path.join(script_dir, 'fonts', 'NotoEmoji-Regular.ttf')
            if os.path.exists(color_font_path):
                font = ImageFont.truetype(color_font_path, font_size)
            else:
                print(f"Warning: {color_font_path} not found, trying regular font")
                regular_font_path = os.path.join(script_dir, 'fonts', 'NotoEmoji-Regular.ttf')
                if os.path.exists(regular_font_path):
                    font = ImageFont.truetype(regular_font_path, font_size)
        else:
            regular_font_path = os.path.join(script_dir, 'fonts', 'NotoEmoji-Regular.ttf')
            if os.path.exists(regular_font_path):
                font = ImageFont.truetype(regular_font_path, font_size)
    except Exception as e:
        print(f"Error loading font: {e}")
        font = None
    
    # PIL Image로 변환
    pil_img = Image.fromarray(img).convert('RGBA')
    draw = ImageDraw.Draw(pil_img)
    
    # 텍스트 크기 계산
    if font:
        try:
            bbox = draw.textbbox((0, 0), emoji_char, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            try:
                text_width, text_height = draw.textsize(emoji_char, font=font)
            except:
                text_width = font_size
                text_height = font_size
        except:
            text_width = font_size
            text_height = font_size
    else:
        text_width = font_size
        text_height = font_size
    
    # 중앙 정렬
    x = (size - text_width) // 2
    y = (size - text_height) // 2 - 2
    
    # 이모지 렌더링
    if use_emoji_color:
        # 원래 이모지 색상 사용
        if font:
            try:
                draw.text((x, y), emoji_char, font=font, fill=None)
            except TypeError:
                try:
                    emoji_layer = Image.new('RGBA', (size, size), (0, 0, 0, 0))
                    emoji_draw = ImageDraw.Draw(emoji_layer)
                    emoji_draw.text((x, y), emoji_char, font=font)
                    pil_img = Image.alpha_composite(pil_img, emoji_layer)
                except Exception as e:
                    print(f"Error rendering emoji: {e}")
                    try:
                        draw.text((x, y), emoji_char, font=font)
                    except:
                        pass
            except Exception as e:
                print(f"Error rendering emoji: {e}")
                try:
                    draw.text((x, y), emoji_char, font=font)
                except:
                    pass
        else:
            try:
                draw.text((x, y), emoji_char, fill=None)
            except TypeError:
                try:
                    draw.text((x, y), emoji_char)
                except:
                    pass
            except:
                try:
                    draw.text((x, y), emoji_char)
                except:
                    pass
    else:
        # 흑백 렌더링
        stroke_color = (255, 255, 255, 255)
        if font:
            try:
                draw.text((x, y), emoji_char, font=font, fill=stroke_color)
            except:
                try:
                    draw.text((x, y), emoji_char, fill=stroke_color)
                except:
                    pass
        else:
            try:
                draw.text((x, y), emoji_char, fill=stroke_color)
            except:
                pass
    
    # RGB로 변환 후 BGR로 변환 (cv2 형식)
    rgb_img = pil_img.convert('RGB')
    bgr_img = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
    
    return bgr_img


def test_emoji_rendering():
    """여러 이모지를 렌더링하고 표시하는 테스트"""
    print("이모지 렌더링 테스트 시작...")
    
    # 테스트할 이모지 목록
    test_emojis = [
        ('tree', '🌲'),
        ('mushroom', '🍄'),
        ('flower', '🌼'),
        ('cat', '🐈'),
        ('apple', '🍎'),
        ('box', '📦'),
    ]
    
    # 각 이모지 렌더링
    images = []
    labels = []
    
    for emoji_name, emoji_char in test_emojis:
        print(f"렌더링 중: {emoji_name} ({emoji_char})")
        img = render_emoji_cv2(emoji_char, size=128, use_emoji_color=True)
        images.append(img)
        labels.append(f"{emoji_name}\n{emoji_char}")
    
    # 그리드로 배치하여 표시
    cols = 3
    rows = (len(images) + cols - 1) // cols
    cell_size = 128
    padding = 10
    
    grid_width = cols * (cell_size + padding) + padding
    grid_height = rows * (cell_size + padding + 40) + padding  # 텍스트 공간 추가
    
    grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols
        
        x = padding + col * (cell_size + padding)
        y = padding + row * (cell_size + padding + 40)
        
        # 이미지 배치
        grid_img[y:y+cell_size, x:x+cell_size] = img
        
        # 레이블 추가 (cv2.putText 사용)
        label_lines = label.split('\n')
        for i, line in enumerate(label_lines):
            text_y = y + cell_size + 20 + i * 20
            cv2.putText(
                grid_img, 
                line, 
                (x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.4, 
                (0, 0, 0), 
                1
            )
    
    # 결과 표시
    print("\n렌더링 완료! 창을 닫으려면 아무 키나 누르세요.")
    cv2.imshow('Emoji Rendering Test', grid_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 결과 저장
    output_path = 'emoji_render_test.png'
    cv2.imwrite(output_path, grid_img)
    print(f"\n결과 이미지 저장: {output_path}")


def test_single_emoji(emoji_name: str = 'tree'):
    """단일 이모지 렌더링 테스트"""
    if emoji_name not in EMOJI_MAP:
        print(f"Error: '{emoji_name}' not found in EMOJI_MAP")
        print(f"Available emojis: {list(EMOJI_MAP.keys())}")
        return
    
    emoji_char = EMOJI_MAP[emoji_name]
    print(f"렌더링 중: {emoji_name} ({emoji_char})")
    
    # 컬러 버전
    img_color = render_emoji_cv2(emoji_char, size=256, use_emoji_color=True)
    
    # 흑백 버전
    img_bw = render_emoji_cv2(emoji_char, size=256, use_emoji_color=False)
    
    # 나란히 표시
    combined = np.hstack([img_color, img_bw])
    
    cv2.imshow(f'Emoji: {emoji_name} (Color | B&W)', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # 단일 이모지 테스트
        emoji_name = sys.argv[1]
        test_single_emoji(emoji_name)
    else:
        # 전체 이모지 테스트
        test_emoji_rendering()

