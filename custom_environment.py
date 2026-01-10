"""
MiniGrid 환경을 위한 Gym Wrapper

이 모듈은 MiniGrid 환경을 쉽게 생성하고 제어할 수 있는 Wrapper 클래스를 제공합니다.
VLM(Vision Language Model)과의 연동을 고려하여 설계되었습니다.

주요 기능:
- 환경 초기화 시 size, walls, room_config 등을 지정
- 현재 환경 이미지 반환 (VLM 입력용)
- 액션 공간 제어 API
- VLM이 반환한 텍스트를 액션으로 변환
"""

from minigrid import register_minigrid_envs
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Goal, Key, Ball, Box, Door, WorldObj
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# MiniGrid 환경 등록 (필수: 환경을 사용하기 전에 등록해야 함)
register_minigrid_envs()

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
}


class EmojiObject(WorldObj):
    """
    이모지를 표시하는 커스텀 객체
    
    이모지 이름, 색상, 집기 가능 여부를 설정할 수 있습니다.
    파싱 시 이모지 이름이 반환됩니다.
    항상 통과 불가능합니다 (에이전트가 올라갈 수 없음).
    """
    
    def __init__(
        self,
        emoji_name: str,
        color: str = 'yellow',
        can_pickup: bool = False
    ):
        """
        Emoji 객체 초기화
        
        Args:
            emoji_name: 이모지 이름 (예: "tree", "rock", "flower" 등)
            color: 색상 (기본값: 'yellow')
                - 지원 색상: 'red', 'green', 'blue', 'purple', 'yellow', 'grey'
            can_pickup: 집기 가능 여부 (기본값: False)
                - True: 에이전트가 앞에서 바라보면 집을 수 있음
                - False: 집을 수 없음 (장애물)
        """
        # 항상 Box 타입 사용 (통과 불가능하게 설정)
        super().__init__('box', color)
        
        # 이모지 이름 저장
        self.emoji_name = emoji_name
        self._can_pickup = can_pickup
        
        # 타입을 'emoji'로 설정하여 구분
        self.type = 'emoji'
    
    def can_pickup(self):
        """에이전트가 이 객체를 집을 수 있는지 여부"""
        return self._can_pickup
    
    def can_overlap(self):
        """에이전트가 이 객체와 겹칠 수 있는지 여부 (항상 False - 통과 불가능)"""
        return False
    
    def encode(self):
        """객체를 인코딩 (MiniGrid 호환성을 위해 'box' 타입으로 인코딩)"""
        # MiniGrid의 encode()는 OBJECT_TO_IDX를 사용하므로
        # 'emoji' 타입이 등록되어 있지 않아 KeyError 발생
        # 따라서 'box' 타입으로 인코딩하되, 이모지 이름은 별도 속성으로 저장
        from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
        
        # 'box' 타입으로 인코딩 (MiniGrid 호환성)
        obj_type_idx = OBJECT_TO_IDX['box']
        color_idx = COLOR_TO_IDX[self.color]
        state = 0
        
        return (obj_type_idx, color_idx, state)
    
    def render(self, img):
        """
        이모지를 실제로 렌더링 (OpenCV 호환)
        emoji_opencv_display.py의 로직을 활용
        
        Args:
            img: 렌더링할 이미지 배열 (numpy array, shape: (H, W, 3))
        """
        # 이모지 문자 가져오기
        emoji_char = EMOJI_MAP.get(self.emoji_name, '❓')
        
        # 이미지 크기 확인
        h, w = img.shape[:2]
        
        # 이모지 폰트 크기 (타일 크기에 맞게 조정)
        font_size = int(min(h, w) * 0.8)
        
        # 로컬 fonts 디렉토리에서 폰트 로드 (emoji_opencv_display.py 로직 활용)
        font = None
        try:
            import os
            # 현재 파일의 디렉토리 기준으로 fonts 폰트 찾기
            script_dir = os.path.dirname(os.path.abspath(__file__))
            local_font_path = os.path.join(script_dir, 'fonts', 'NotoEmoji-Regular.ttf')
            
            # 로컬 폰트 로드
            if os.path.exists(local_font_path):
                font = ImageFont.truetype(local_font_path, font_size)
        except Exception:
            font = None
        
        # RGBA 모드로 변환 (투명도 지원)
        pil_img = Image.fromarray(img.astype(np.uint8)).convert('RGBA')
        draw = ImageDraw.Draw(pil_img)
        
        # 이모지 텍스트 크기 계산
        if font:
            try:
                # textbbox 사용 (PIL 8.0.0 이상)
                bbox = draw.textbbox((0, 0), emoji_char, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except AttributeError:
                # textsize 사용 (구버전 PIL)
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
        
        # 중앙에 이모지 그리기
        x = (w - text_width) // 2
        y = (h - text_height) // 2 - 2  # 약간 위로 조정
        
        # 이모지 그리기 (RGBA 흰색)
        fill_color = (255, 255, 255, 255)
        
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
        
        # RGBA를 RGB로 변환하여 원본 이미지에 복사
        rgb_img = pil_img.convert('RGB')
        img[:] = np.array(rgb_img)
    
    def __str__(self):
        """문자열 표현 (이모지 이름 반환)"""
        return self.emoji_name
    
    def __repr__(self):
        """객체 표현"""
        return f"EmojiObject(emoji_name='{self.emoji_name}', color='{self.color}', can_pickup={self._can_pickup})"


class CustomRoomEnv(MiniGridEnv):
    """
    커스텀 방 구조를 가진 MiniGrid 환경 클래스
    
    이 클래스는 MiniGridEnv를 상속받아 커스텀 방 구조를 생성합니다.
    내부적으로 사용되며, 외부에서는 CustomRoomWrapper를 통해 사용하는 것을 권장합니다.
    """
    
    def __init__(self, size=10, room_config=None, robot_emoji=None, **kwargs):
        """
        환경 초기화
        
        Args:
            size: 환경 크기 (기본값: 10)
            room_config: 방 구조 설정 딕셔너리
            robot_emoji: 로봇 이모지 문자 (기본값: None, None이면 arrow.png 사용)
                - 예: '🤖' (로봇 이모지)
                - None: arrow.png 이미지 사용
            **kwargs: MiniGridEnv의 추가 파라미터
        """
        self.size = size
        self.room_config = room_config or {}
        self.robot_emoji = robot_emoji
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=4 * size * size,
            **kwargs
        )
    
    @staticmethod
    def _gen_mission():
        """미션 텍스트 생성"""
        return "explore"
    
    def _gen_grid(self, width, height):
        """
        그리드를 생성하는 메서드
        
        이 메서드는 환경이 리셋될 때마다 호출되어 새로운 그리드를 생성합니다.
        
        Args:
            width: 그리드 너비
            height: 그리드 높이
        """
        # 1단계: 빈 그리드 생성
        self.grid = Grid(width, height)
        
        # 2단계: 외벽 생성 (전체 그리드를 둘러싸는 벽)
        self.grid.wall_rect(0, 0, width, height)
        
        # 3단계: 커스텀 설정이 있으면 적용
        if self.room_config:
            # 3-1: 벽 배치
            if 'walls' in self.room_config:
                for wall_info in self.room_config['walls']:
                    # 벽 정보가 튜플인 경우 (기존 형태: (x, y))
                    if isinstance(wall_info, tuple):
                        if len(wall_info) == 2:
                            wall_x, wall_y = wall_info
                            wall_color = 'grey'  # 기본 색상
                        elif len(wall_info) == 3:
                            wall_x, wall_y, wall_color = wall_info
                        else:
                            continue
                    # 벽 정보가 딕셔너리인 경우 (새 형태: {'pos': (x, y), 'color': 'red'})
                    elif isinstance(wall_info, dict):
                        wall_pos = wall_info.get('pos', (0, 0))
                        wall_x, wall_y = wall_pos
                        wall_color = wall_info.get('color', 'grey')
                    else:
                        continue
                    
                    # 좌표가 유효한 범위 내에 있는지 확인
                    if 0 <= wall_x < width and 0 <= wall_y < height:
                        self.grid.set(wall_x, wall_y, Wall(wall_color))
            
            # 3-2: Goal 위치 설정 (공식 방법: put_obj 사용)
            if 'goal_pos' in self.room_config:
                goal_x, goal_y = self.room_config['goal_pos']
                if 0 <= goal_x < width and 0 <= goal_y < height:
                    # put_obj는 객체를 안전하게 배치하는 헬퍼 메서드
                    self.put_obj(Goal(), goal_x, goal_y)
            
            # 3-3: 객체 배치 (공식 방법: put_obj 사용)
            if 'objects' in self.room_config:
                for obj_info in self.room_config['objects']:
                    # 객체 정보 추출
                    obj_type = obj_info.get('type', 'key')
                    obj_pos = obj_info.get('pos', (1, 1))
                    obj_color = obj_info.get('color', 'yellow')
                    
                    obj_x, obj_y = obj_pos
                    if 0 <= obj_x < width and 0 <= obj_y < height:
                        # 객체 타입에 따라 적절한 객체 생성
                        if obj_type == 'key':
                            obj = Key(obj_color)
                        elif obj_type == 'ball':
                            obj = Ball(obj_color)
                        elif obj_type == 'box':
                            obj = Box(obj_color)
                        elif obj_type == 'door':
                            is_locked = obj_info.get('is_locked', False)
                            is_open = obj_info.get('is_open', True)
                            obj = Door(obj_color, is_locked=is_locked, is_open=is_open)
                        elif obj_type == 'emoji':
                            # 이모지 객체 생성
                            emoji_name = obj_info.get('emoji_name', 'emoji')
                            can_pickup = obj_info.get('can_pickup', False)
                            obj = EmojiObject(emoji_name=emoji_name, color=obj_color, can_pickup=can_pickup)
                        else:
                            obj = Key(obj_color)  # 기본값
                        
                        # 객체를 그리드에 배치
                        self.put_obj(obj, obj_x, obj_y)
        
        # 4단계: 에이전트 시작 위치 설정
        if self.room_config and 'start_pos' in self.room_config:
            # 명시적으로 시작 위치가 지정된 경우
            start_x, start_y = self.room_config['start_pos']
            self.agent_pos = np.array([start_x, start_y])
            self.agent_dir = 0  # 0=오른쪽, 1=아래, 2=왼쪽, 3=위
        else:
            # 시작 위치가 지정되지 않은 경우 자동으로 빈 공간에 배치
            self.place_agent()
        
        # 5단계: Mission 설정 (공식 방법)
        self.mission = self._gen_mission()
    
    def render(self):
        """
        렌더링 메서드 오버라이드
        에이전트 삼각형을 그리지 않고 arrow.png를 사용합니다.
        """
        # 기본 렌더링 수행 (에이전트 포함)
        frame = super().render()
        
        if frame is None:
            return frame
        
        # 에이전트 위치 및 방향 확인
        if not hasattr(self, 'agent_pos') or not hasattr(self, 'agent_dir'):
            return frame
        
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        agent_dir = self.agent_dir
        
        # 타일 크기 확인
        actual_tile_size = self.tile_size if hasattr(self, 'tile_size') else 32
        
        # 에이전트 타일의 픽셀 좌표 계산
        start_x = agent_x * actual_tile_size
        start_y = agent_y * actual_tile_size
        end_x = start_x + actual_tile_size
        end_y = start_y + actual_tile_size
        
        # 프레임 크기 확인
        frame_h, frame_w = frame.shape[:2]
        
        # 좌표가 프레임 범위 내에 있는지 확인
        if start_x < 0 or start_y < 0 or end_x > frame_w or end_y > frame_h:
            return frame
        
        # 로봇 표시: 이모지 또는 arrow.png 이미지
        try:
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 이모지 모드인 경우
            if self.robot_emoji is not None:
                # 프레임을 PIL 이미지로 변환
                pil_frame = Image.fromarray(frame.astype(np.uint8)).convert('RGBA')
                
                # 에이전트 타일 영역을 에이전트 없이 직접 렌더링
                cell = self.grid.get(agent_x, agent_y)
                
                # 타일만 렌더링 (에이전트 없이)
                from minigrid.core.grid import Grid
                try:
                    bg_tile_img = Grid.render_tile(
                        cell,
                        (agent_x, agent_y),
                        agent_dir=None,
                        highlight=False,
                        tile_size=actual_tile_size,
                        subdivs=3
                    )
                    
                    if bg_tile_img is not None:
                        if isinstance(bg_tile_img, np.ndarray):
                            bg_tile = Image.fromarray(bg_tile_img.astype(np.uint8)).convert('RGBA')
                        elif hasattr(bg_tile_img, 'convert'):
                            bg_tile = bg_tile_img.convert('RGBA')
                        else:
                            bg_tile = Image.fromarray(np.array(bg_tile_img)).convert('RGBA')
                    else:
                        agent_tile = pil_frame.crop((start_x, start_y, end_x, end_y))
                        tile_array = np.array(agent_tile)
                        red_mask = (
                            (tile_array[:, :, 0] > 150) &
                            (tile_array[:, :, 0] > tile_array[:, :, 1] + 50) &
                            (tile_array[:, :, 0] > tile_array[:, :, 2] + 50) &
                            (tile_array[:, :, 1] < 150) &
                            (tile_array[:, :, 2] < 150)
                        )
                        
                        if np.any(red_mask):
                            corner_size = 4
                            corners = np.concatenate([
                                tile_array[:corner_size, :corner_size].reshape(-1, 4),
                                tile_array[:corner_size, -corner_size:].reshape(-1, 4),
                                tile_array[-corner_size:, :corner_size].reshape(-1, 4),
                                tile_array[-corner_size:, -corner_size:].reshape(-1, 4)
                            ])
                            non_red_corners = corners[
                                (corners[:, 0] <= 200) | (corners[:, 1] >= 100) | (corners[:, 2] >= 100)
                            ]
                            if len(non_red_corners) > 0:
                                bg_color = np.mean(non_red_corners[:, :3], axis=0).astype(int)
                                tile_array[red_mask, 0] = bg_color[0]
                                tile_array[red_mask, 1] = bg_color[1]
                                tile_array[red_mask, 2] = bg_color[2]
                                tile_array[red_mask, 3] = 255
                            
                            bg_tile = Image.fromarray(tile_array.astype(np.uint8), 'RGBA')
                        else:
                            bg_tile = agent_tile
                except Exception:
                    agent_tile = pil_frame.crop((start_x, start_y, end_x, end_y))
                    tile_array = np.array(agent_tile)
                    red_mask = (
                        (tile_array[:, :, 0] > 150) &
                        (tile_array[:, :, 0] > tile_array[:, :, 1] + 50) &
                        (tile_array[:, :, 0] > tile_array[:, :, 2] + 50) &
                        (tile_array[:, :, 1] < 150) &
                        (tile_array[:, :, 2] < 150)
                    )
                    
                    if np.any(red_mask):
                        corner_size = 4
                        corners = np.concatenate([
                            tile_array[:corner_size, :corner_size].reshape(-1, 4),
                            tile_array[:corner_size, -corner_size:].reshape(-1, 4),
                            tile_array[-corner_size:, :corner_size].reshape(-1, 4),
                            tile_array[-corner_size:, -corner_size:].reshape(-1, 4)
                        ])
                        non_red_corners = corners[
                            (corners[:, 0] <= 200) | (corners[:, 1] >= 100) | (corners[:, 2] >= 100)
                        ]
                        if len(non_red_corners) > 0:
                            bg_color = np.mean(non_red_corners[:, :3], axis=0).astype(int)
                            tile_array[red_mask, 0] = bg_color[0]
                            tile_array[red_mask, 1] = bg_color[1]
                            tile_array[red_mask, 2] = bg_color[2]
                            tile_array[red_mask, 3] = 255
                        
                        bg_tile = Image.fromarray(tile_array.astype(np.uint8), 'RGBA')
                    else:
                        bg_tile = agent_tile
                
                # 이모지 렌더링
                font_size = int(actual_tile_size * 0.8)
                font = None
                try:
                    local_font_path = os.path.join(script_dir, 'fonts', 'NotoEmoji-Regular.ttf')
                    if os.path.exists(local_font_path):
                        font = ImageFont.truetype(local_font_path, font_size)
                except Exception:
                    font = None
                
                draw = ImageDraw.Draw(bg_tile)
                
                # 이모지 텍스트 크기 계산
                if font:
                    try:
                        bbox = draw.textbbox((0, 0), self.robot_emoji, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                    except AttributeError:
                        try:
                            text_width, text_height = draw.textsize(self.robot_emoji, font=font)
                        except:
                            text_width = font_size
                            text_height = font_size
                    except:
                        text_width = font_size
                        text_height = font_size
                else:
                    text_width = font_size
                    text_height = font_size
                
                # 중앙에 이모지 그리기
                x = (actual_tile_size - text_width) // 2
                y = (actual_tile_size - text_height) // 2 - 2
                
                # 이모지 그리기
                fill_color = (255, 255, 255, 255)
                if font:
                    try:
                        draw.text((x, y), self.robot_emoji, font=font, fill=fill_color)
                    except:
                        try:
                            draw.text((x, y), self.robot_emoji, fill=fill_color)
                        except:
                            pass
                else:
                    try:
                        draw.text((x, y), self.robot_emoji, fill=fill_color)
                    except:
                        pass
                
                # 수정된 타일을 다시 프레임에 붙이기
                pil_frame.paste(bg_tile, (start_x, start_y))
                frame = np.array(pil_frame.convert('RGB'))
            
            # arrow.png 이미지 모드 (기본)
            else:
                arrow_img_path = os.path.join(script_dir, 'asset', 'arrow.png')
                
                if os.path.exists(arrow_img_path):
                # 프레임을 PIL 이미지로 변환
                pil_frame = Image.fromarray(frame.astype(np.uint8)).convert('RGBA')
                
                # 에이전트 타일 영역을 에이전트 없이 직접 렌더링
                # 그리드에서 해당 셀만 가져와서 렌더링
                cell = self.grid.get(agent_x, agent_y)
                
                # 타일만 렌더링 (에이전트 없이)
                from minigrid.core.grid import Grid
                try:
                    # Grid.render_tile을 사용하여 타일 배경만 렌더링
                    bg_tile_img = Grid.render_tile(
                        cell,
                        (agent_x, agent_y),
                        agent_dir=None,  # 에이전트 방향 없음
                        highlight=False,
                        tile_size=actual_tile_size,
                        subdivs=3
                    )
                    
                    if bg_tile_img is not None:
                        # numpy array를 PIL Image로 변환
                        if isinstance(bg_tile_img, np.ndarray):
                            bg_tile = Image.fromarray(bg_tile_img.astype(np.uint8)).convert('RGBA')
                        elif hasattr(bg_tile_img, 'convert'):
                            bg_tile = bg_tile_img.convert('RGBA')
                        else:
                            bg_tile = Image.fromarray(np.array(bg_tile_img)).convert('RGBA')
                    else:
                        # 렌더링 실패 시 프레임에서 추출하되, 빨간색 제거
                        agent_tile = pil_frame.crop((start_x, start_y, end_x, end_y))
                        tile_array = np.array(agent_tile)
                        
                        # 빨간색 픽셀 감지 및 제거
                        red_mask = (
                            (tile_array[:, :, 0] > 150) &
                            (tile_array[:, :, 0] > tile_array[:, :, 1] + 50) &
                            (tile_array[:, :, 0] > tile_array[:, :, 2] + 50) &
                            (tile_array[:, :, 1] < 150) &
                            (tile_array[:, :, 2] < 150)
                        )
                        
                        if np.any(red_mask):
                            # 모서리에서 배경 색상 추정
                            corner_size = 4
                            corners = np.concatenate([
                                tile_array[:corner_size, :corner_size].reshape(-1, 4),
                                tile_array[:corner_size, -corner_size:].reshape(-1, 4),
                                tile_array[-corner_size:, :corner_size].reshape(-1, 4),
                                tile_array[-corner_size:, -corner_size:].reshape(-1, 4)
                            ])
                            non_red_corners = corners[
                                (corners[:, 0] <= 200) | (corners[:, 1] >= 100) | (corners[:, 2] >= 100)
                            ]
                            if len(non_red_corners) > 0:
                                bg_color = np.mean(non_red_corners[:, :3], axis=0).astype(int)
                                tile_array[red_mask, 0] = bg_color[0]
                                tile_array[red_mask, 1] = bg_color[1]
                                tile_array[red_mask, 2] = bg_color[2]
                                tile_array[red_mask, 3] = 255
                            
                            bg_tile = Image.fromarray(tile_array.astype(np.uint8), 'RGBA')
                        else:
                            bg_tile = agent_tile
                except Exception as e:
                    # 렌더링 실패 시 프레임에서 추출하고 빨간색 제거
                    agent_tile = pil_frame.crop((start_x, start_y, end_x, end_y))
                    tile_array = np.array(agent_tile)
                    
                    # 빨간색 픽셀 감지 및 제거
                    red_mask = (
                        (tile_array[:, :, 0] > 150) &
                        (tile_array[:, :, 0] > tile_array[:, :, 1] + 50) &
                        (tile_array[:, :, 0] > tile_array[:, :, 2] + 50) &
                        (tile_array[:, :, 1] < 150) &
                        (tile_array[:, :, 2] < 150)
                    )
                    
                    if np.any(red_mask):
                        # 모서리에서 배경 색상 추정
                        corner_size = 4
                        corners = np.concatenate([
                            tile_array[:corner_size, :corner_size].reshape(-1, 4),
                            tile_array[:corner_size, -corner_size:].reshape(-1, 4),
                            tile_array[-corner_size:, :corner_size].reshape(-1, 4),
                            tile_array[-corner_size:, -corner_size:].reshape(-1, 4)
                        ])
                        non_red_corners = corners[
                            (corners[:, 0] <= 200) | (corners[:, 1] >= 100) | (corners[:, 2] >= 100)
                        ]
                        if len(non_red_corners) > 0:
                            bg_color = np.mean(non_red_corners[:, :3], axis=0).astype(int)
                            tile_array[red_mask, 0] = bg_color[0]
                            tile_array[red_mask, 1] = bg_color[1]
                            tile_array[red_mask, 2] = bg_color[2]
                            tile_array[red_mask, 3] = 255
                        
                        bg_tile = Image.fromarray(tile_array.astype(np.uint8), 'RGBA')
                    else:
                        bg_tile = agent_tile
                
                # arrow.png 이미지 로드 및 리사이즈
                arrow_img = Image.open(arrow_img_path).convert('RGBA')
                arrow_img = arrow_img.resize((actual_tile_size, actual_tile_size), Image.Resampling.LANCZOS)
                
                # 방향에 따라 회전
                # MiniGrid 방향: 0=오른쪽(East), 1=아래(South), 2=왼쪽(West), 3=위(North)
                # arrow.png가 오른쪽을 향한다고 가정
                rotation_map = {
                    0: 0,      # 오른쪽 (기본)
                    1: 90,     # 아래 (시계방향 90도)
                    2: 180,    # 왼쪽 (시계방향 180도)
                    3: 270     # 위 (시계방향 270도)
                }
                rotation_angle = rotation_map.get(agent_dir, 0)
                
                if rotation_angle != 0:
                    arrow_img = arrow_img.rotate(-rotation_angle, expand=False, fillcolor=(0, 0, 0, 0))
                
                # 배경 타일 위에 arrow 이미지 합성 (투명도 유지)
                bg_tile.paste(arrow_img, (0, 0), arrow_img)
                
                # 수정된 타일을 다시 프레임에 붙이기
                pil_frame.paste(bg_tile, (start_x, start_y))
                
                # RGB로 변환하여 numpy 배열로 변환
                frame = np.array(pil_frame.convert('RGB'))
        except Exception as e:
            # 이미지 로드 실패 시 기본 렌더링 유지
            print(f"Warning: 커스텀 에이전트 이미지 로드 실패 ({e}). 기본 렌더링을 사용합니다.")
            import traceback
            traceback.print_exc()
        
        return frame


class CustomRoomWrapper:
    """
    MiniGrid 환경을 제어하기 위한 Wrapper 클래스
    
    이 클래스는 CustomRoomEnv를 감싸서 더 편리한 API를 제공합니다.
    VLM과의 연동을 고려하여 설계되었습니다.
    
    사용 예시:
        # 환경 생성
        wrapper = CustomRoomWrapper(
            size=15,
            walls=[(5, 0), (5, 1), ...],
            room_config={'start_pos': (2, 2), 'goal_pos': (10, 10)}
        )
        
        # 이미지 가져오기 (VLM에 전달)
        image = wrapper.get_image()
        
        # VLM이 반환한 액션 실행
        action_str = "move forward"  # VLM이 반환한 텍스트
        action = wrapper.parse_action(action_str)
        obs, reward, done, info = wrapper.step(action)
    """
    
    # 액션 이름과 인덱스 매핑 (VLM이 텍스트로 액션을 반환할 수 있도록)
    ACTION_NAMES = {
        0: "turn left",
        1: "turn right", 
        2: "move forward",
        3: "move backward",
        4: "pickup",
        5: "drop",
        6: "toggle"
    }
    
    # 액션 이름의 다양한 표현 (VLM이 다양한 표현을 사용할 수 있도록)
    ACTION_ALIASES = {
        "turn left": 0, "left": 0, "rotate left": 0, "turn_left": 0,
        "turn right": 1, "right": 1, "rotate right": 1, "turn_right": 1,
        "move forward": 2, "forward": 2, "go forward": 2, "move_forward": 2, "w": 2,
        "move backward": 3, "backward": 3, "go backward": 3, "move_backward": 3, "s": 3,
        "pickup": 4, "pick up": 4, "pick_up": 4, "grab": 4,
        "drop": 5, "put down": 5, "put_down": 5, "release": 5,
        "toggle": 6, "interact": 6, "use": 6, "activate": 6
    }
    
    def __init__(
        self,
        size: int = 10,
        walls: Optional[List[Tuple[int, int]]] = None,
        room_config: Optional[Dict] = None,
        render_mode: str = 'rgb_array',
        robot_emoji: Optional[str] = None,
        **kwargs
    ):
        """
        Wrapper 초기화
        
        Args:
            size: 환경 크기 (기본값: 10)
            walls: 벽 위치 리스트 (기본값: None)
                - 기존 형태: [(x1, y1), (x2, y2), ...] (기본 색상: 'grey')
                - 색상 지정: [(x1, y1, 'red'), (x2, y2, 'blue'), ...]
                - 딕셔너리 형태: [{'pos': (x, y), 'color': 'red'}, ...]
            room_config: 방 구조 설정 딕셔너리 (기본값: None)
                - start_pos: (x, y) 튜플 - 에이전트 시작 위치
                - goal_pos: (x, y) 튜플 - 목표 위치
                - walls: 벽 리스트 (위와 동일한 형태 지원)
                - objects: 객체 리스트 [{'type': 'key', 'pos': (x, y), 'color': 'yellow'}, ...]
            render_mode: 렌더링 모드 ('rgb_array' 또는 'human') (기본값: 'rgb_array')
            robot_emoji: 로봇 이모지 문자 (기본값: None, None이면 arrow.png 사용)
                - 예: '🤖' (로봇 이모지)
                - None: arrow.png 이미지 사용
            **kwargs: CustomRoomEnv의 추가 파라미터
        """
        # 입력 파라미터 저장
        self.size = size
        self.walls = walls or []
        self.render_mode = render_mode
        
        # room_config 구성 (walls가 별도로 제공된 경우 병합)
        if room_config is None:
            room_config = {}
        
        # walls가 별도로 제공된 경우 room_config에 추가
        if walls and 'walls' not in room_config:
            # 기존 walls가 있으면 병합, 없으면 새로 생성
            existing_walls = room_config.get('walls', [])
            room_config['walls'] = existing_walls + walls
        
        # 내부 환경 생성 (CustomRoomEnv 인스턴스)
        self.env = CustomRoomEnv(
            size=size,
            room_config=room_config,
            render_mode=render_mode,
            robot_emoji=robot_emoji,
            **kwargs
        )
        
        # 현재 관찰 상태 저장 (초기화 시 리셋)
        self.current_obs = None
        self.current_info = None
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        환경을 초기 상태로 리셋
        
        Args:
            seed: 랜덤 시드 (기본값: None)
        
        Returns:
            observation: 초기 관찰 (딕셔너리)
            info: 추가 정보 (딕셔너리)
        """
        # 환경 리셋
        self.current_obs, self.current_info = self.env.reset(seed=seed)
        return self.current_obs, self.current_info
    
    def step(self, action: Union[int, str]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        액션을 실행하고 다음 상태로 전이
        
        Args:
            action: 액션 (정수 인덱스 또는 액션 이름 문자열)
                - 0 또는 "turn left": 왼쪽으로 회전
                - 1 또는 "turn right": 오른쪽으로 회전
                - 2 또는 "move forward": 앞으로 이동
                - 3 또는 "move backward": 뒤로 이동
                - 4 또는 "pickup": 물체 집기
                - 5 또는 "drop": 물체 놓기
                - 6 또는 "toggle": 상호작용 (문 열기 등)
        
        Returns:
            observation: 새로운 관찰 (딕셔너리)
            reward: 보상 (float)
            terminated: 목표 달성 여부 (bool)
            truncated: 시간 초과 여부 (bool)
            info: 추가 정보 (딕셔너리)
        """
        # 액션이 문자열인 경우 정수로 변환
        if isinstance(action, str):
            action = self.parse_action(action)
        
        # 액션 실행
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 현재 상태 업데이트
        self.current_obs = obs
        self.current_info = info
        
        return obs, reward, terminated, truncated, info
    
    def get_image(self, fov_range: Optional[int] = None, fov_width: Optional[int] = None) -> np.ndarray:
        """
        현재 환경의 이미지를 반환 (VLM 입력용)
        시야 제한(fog of war) 기능을 선택적으로 적용할 수 있습니다.
        
        Args:
            fov_range: 에이전트 앞으로 볼 수 있는 거리 (칸 수). None이면 시야 제한 없음
            fov_width: 시야의 좌우 폭 (칸 수). None이면 시야 제한 없음
        
        Returns:
            image: RGB 이미지 배열 (H, W, 3) 형태의 numpy 배열
        """
        # 환경 렌더링 (RGB 배열로 반환)
        image = self.env.render()
        
        # 이미지가 None인 경우 빈 배열 반환
        if image is None:
            return np.zeros((self.size * 32, self.size * 32, 3), dtype=np.uint8)
        
        # 시야 제한 적용 (fov_range와 fov_width가 모두 지정된 경우)
        if fov_range is not None and fov_width is not None:
            image = self._apply_fog_of_war(image, fov_range, fov_width)
        
        return image
    
    def _apply_fog_of_war(self, image: np.ndarray, fov_range: int, fov_width: int) -> np.ndarray:
        """
        시야 제한을 적용하여 시야 밖의 영역을 검은색으로 마스킹
        
        Args:
            image: 원본 이미지 (H, W, 3)
            fov_range: 앞으로 볼 수 있는 거리
            fov_width: 시야의 좌우 폭
        
        Returns:
            masked_image: 시야 제한이 적용된 이미지
        """
        # 에이전트 위치 및 방향
        if not hasattr(self.env, 'agent_pos') or not hasattr(self.env, 'agent_dir'):
            return image
        
        agent_pos = self.env.agent_pos
        if isinstance(agent_pos, np.ndarray):
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        else:
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        
        agent_dir = self.env.agent_dir
        
        # 방향 벡터 (0=오른쪽, 1=아래, 2=왼쪽, 3=위)
        dir_vectors = {
            0: (1, 0),   # 오른쪽
            1: (0, 1),   # 아래
            2: (-1, 0),  # 왼쪽
            3: (0, -1)   # 위
        }
        
        # 에이전트가 바라보는 방향
        forward_dx, forward_dy = dir_vectors[agent_dir]
        
        # 이미지 복사 (원본 보존)
        masked_image = image.copy()
        h, w = image.shape[:2]
        
        # 각 셀의 크기 (MiniGrid는 일반적으로 32x32 픽셀)
        cell_size = 32
        
        # 그리드 크기
        grid_size = self.size
        
        # 각 셀에 대해 시야 범위 내인지 확인
        for grid_y in range(grid_size):
            for grid_x in range(grid_size):
                # 에이전트 위치에서 이 셀까지의 상대 위치
                dx = grid_x - agent_x
                dy = grid_y - agent_y
                
                # 에이전트 방향 기준으로 변환
                if agent_dir == 0:  # 오른쪽
                    rel_x, rel_y = dx, -dy  # y축 반전
                elif agent_dir == 1:  # 아래
                    rel_x, rel_y = dy, dx
                elif agent_dir == 2:  # 왼쪽
                    rel_x, rel_y = -dx, dy
                else:  # 위
                    rel_x, rel_y = -dy, -dx
                
                # 시야 범위 확인
                # 앞으로 fov_range 칸까지, 좌우로 각각 fov_width//2 칸까지
                in_fov = (
                    rel_x >= 0 and  # 앞쪽만
                    rel_x <= fov_range and  # 최대 거리
                    abs(rel_y) <= fov_width // 2  # 좌우 폭
                )
                
                # 시야 밖이면 검은색으로 마스킹
                if not in_fov:
                    # 픽셀 좌표 계산
                    pixel_x = grid_x * cell_size
                    pixel_y = grid_y * cell_size
                    
                    # 셀 영역을 검은색으로 마스킹
                    end_x = min(pixel_x + cell_size, w)
                    end_y = min(pixel_y + cell_size, h)
                    
                    masked_image[pixel_y:end_y, pixel_x:end_x] = [0, 0, 0]
        
        return masked_image
    
    def get_action_space(self) -> Dict:
        """
        액션 공간 정보 반환
        
        Returns:
            action_space_info: 액션 공간 정보 딕셔너리
                - n: 액션 개수
                - actions: 액션 이름 리스트
                - action_mapping: 액션 인덱스와 이름 매핑
        """
        return {
            'n': self.env.action_space.n,
            'actions': list(self.ACTION_NAMES.values()),
            'action_mapping': self.ACTION_NAMES,
            'action_aliases': self.ACTION_ALIASES
        }
    
    def get_action_names(self) -> List[str]:
        """
        액션 이름 리스트 반환 (VLM용)
        
        Returns:
            action_names: 액션 이름 리스트
        """
        return list(self.ACTION_NAMES.values())
    
    def parse_action(self, action_str: str) -> int:
        """
        VLM이 반환한 텍스트를 액션 인덱스로 변환
        
        이 메서드는 VLM이 반환한 텍스트 액션을 정수 인덱스로 변환합니다.
        다양한 표현을 지원합니다 (예: "move forward", "forward", "go forward", "2" 등).
        
        Args:
            action_str: 액션 텍스트 (예: "move forward", "turn left", "2")
        
        Returns:
            action: 액션 인덱스 (0-6)
        
        Raises:
            ValueError: 알 수 없는 액션인 경우
        """
        # 공백 제거
        action_str = action_str.strip()
        
        # 숫자 문자열인 경우 직접 변환 (예: "0", "1", "2" 등)
        try:
            action_int = int(action_str)
            # 유효한 액션 범위인지 확인 (0-6)
            if 0 <= action_int < self.env.action_space.n:
                return action_int
        except ValueError:
            # 숫자가 아니면 문자열로 처리
            pass
        
        # 소문자로 변환
        action_str_lower = action_str.lower()
        
        # 액션 별칭에서 찾기
        if action_str_lower in self.ACTION_ALIASES:
            return self.ACTION_ALIASES[action_str_lower]
        
        # 직접 매핑에서 찾기
        for idx, name in self.ACTION_NAMES.items():
            if action_str_lower == name.lower():
                return idx
        
        # 찾지 못한 경우 에러 발생
        raise ValueError(
            f"Unknown action: '{action_str}'. "
            f"Available actions: {list(self.ACTION_ALIASES.keys())} or numbers 0-{self.env.action_space.n-1}"
        )
    
    def get_state(self) -> Dict:
        """
        현재 환경 상태 정보 반환
        
        Returns:
            state: 환경 상태 딕셔너리
                - agent_pos: 에이전트 위치
                - agent_dir: 에이전트 방향
                - mission: 현재 미션
                - image: 현재 이미지
        """
        # agent_pos 처리: numpy array인 경우 copy(), tuple인 경우 그대로 반환
        agent_pos = None
        if hasattr(self.env, 'agent_pos'):
            if isinstance(self.env.agent_pos, np.ndarray):
                agent_pos = self.env.agent_pos.copy()
            else:
                # tuple이나 다른 타입인 경우 그대로 반환
                agent_pos = self.env.agent_pos
        
        return {
            'agent_pos': agent_pos,
            'agent_dir': self.env.agent_dir if hasattr(self.env, 'agent_dir') else None,
            'mission': self.env.mission if hasattr(self.env, 'mission') else None,
            'image': self.get_image()
        }
    
    def get_heading(self) -> str:
        """
        현재 로봇의 heading 방향을 문자열로 반환
        
        Returns:
            heading: 방향 문자열
                - "East" (오른쪽, agent_dir=0)
                - "South" (아래, agent_dir=1)
                - "West" (왼쪽, agent_dir=2)
                - "North" (위, agent_dir=3)
        """
        if not hasattr(self.env, 'agent_dir'):
            return "Unknown"
        
        agent_dir = self.env.agent_dir
        heading_map = {
            0: "East",   # 오른쪽
            1: "South",  # 아래
            2: "West",   # 왼쪽
            3: "North"   # 위
        }
        return heading_map.get(agent_dir, "Unknown")
    
    def get_heading_short(self) -> str:
        """
        현재 로봇의 heading 방향을 짧은 문자열로 반환
        
        Returns:
            heading: 방향 문자열
                - "E" (East, 오른쪽, agent_dir=0)
                - "S" (South, 아래, agent_dir=1)
                - "W" (West, 왼쪽, agent_dir=2)
                - "N" (North, 위, agent_dir=3)
        """
        if not hasattr(self.env, 'agent_dir'):
            return "?"
        
        agent_dir = self.env.agent_dir
        heading_map = {
            0: "E",  # East (오른쪽)
            1: "S",  # South (아래)
            2: "W",  # West (왼쪽)
            3: "N"   # North (위)
        }
        return heading_map.get(agent_dir, "?")
    
    def get_heading_description(self) -> str:
        """
        현재 로봇의 heading 방향을 상세 설명 문자열로 반환
        
        Returns:
            description: 방향 설명 문자열
                예: "facing East (right)" 또는 "facing North (up)"
        """
        heading = self.get_heading()
        if heading == "Unknown":
            return "heading direction unknown"
        
        direction_descriptions = {
            "East": "right",
            "South": "down",
            "West": "left",
            "North": "up"
        }
        direction = direction_descriptions.get(heading, "")
        return f"facing {heading} ({direction})"
    
    def parse_grid(self) -> Dict[Tuple[int, int], str]:
        """
        그리드를 파싱하여 각 위치의 객체 정보를 반환
        
        이모지 객체의 경우 이모지 이름이 반환됩니다.
        
        Returns:
            grid_map: 딕셔너리 {(x, y): object_name}
                - 이모지 객체: 이모지 이름 (예: "tree", "rock")
                - 다른 객체: 객체 타입 (예: "wall", "key", "goal")
                - 빈 공간: None 또는 빈 문자열
        """
        grid_map = {}
        
        if not hasattr(self.env, 'grid'):
            return grid_map
        
        width = self.env.grid.width
        height = self.env.grid.height
        
        for y in range(height):
            for x in range(width):
                cell = self.env.grid.get(x, y)
                
                if cell is None:
                    # 빈 공간
                    grid_map[(x, y)] = None
                elif hasattr(cell, 'type'):
                    # 이모지 객체인 경우
                    if cell.type == 'emoji' and hasattr(cell, 'emoji_name'):
                        grid_map[(x, y)] = cell.emoji_name
                    else:
                        # 다른 객체 타입
                        grid_map[(x, y)] = cell.type
                else:
                    # 객체 타입을 알 수 없는 경우
                    grid_map[(x, y)] = str(cell)
        
        return grid_map
    
    def get_emoji_at(self, x: int, y: int) -> Optional[str]:
        """
        특정 위치의 이모지 이름을 반환
        
        Args:
            x: X 좌표
            y: Y 좌표
            
        Returns:
            emoji_name: 이모지 이름 (이모지 객체가 아닌 경우 None)
        """
        if not hasattr(self.env, 'grid'):
            return None
        
        cell = self.env.grid.get(x, y)
        
        if cell is None:
            return None
        
        if hasattr(cell, 'type') and cell.type == 'emoji' and hasattr(cell, 'emoji_name'):
            return cell.emoji_name
        
        return None
    
    def close(self):
        """환경 종료 및 리소스 정리"""
        self.env.close()


# 편의 함수들 (기존 코드와의 호환성을 위해 유지)

def create_house_environment():
    """
    실내 집 환경 생성 (복도, 방, 차고 구조)
    
    Returns:
        CustomRoomWrapper: 실내 집 환경 Wrapper 인스턴스
    """
    size = 15
    
    # 벽 구조 정의 (복도와 방을 구분하는 벽)
    walls = []
    
    # 외벽 (상하좌우 경계)
    for i in range(size):
        walls.append((i, 0))  # 상단
        walls.append((i, size-1))  # 하단
        walls.append((0, i))  # 좌측
        walls.append((size-1, i))  # 우측
    
    # 내부 벽 (방 구분)
    # 복도 (중앙 세로)
    for i in range(5, 10):
        walls.append((7, i))  # 좌측 방과 복도 구분
    
    # 방 구분 벽 (가로)
    for i in range(1, 7):
        walls.append((i, 5))  # 상단 방 구분
        walls.append((i, 10))  # 하단 방 구분
    
    # 복도와 차고 구분
    for i in range(8, size-1):
        walls.append((i, 7))
    
    # 시작 위치 (거실)
    start_pos = (2, 2)
    
    # Goal 위치 (차고)
    goal_pos = (12, 12)
    
    # 객체 배치 (가위, 열쇠 등)
    objects = [
        {'type': 'key', 'pos': (3, 3), 'color': 'yellow'},  # 안방에 가위(키로 대체)
        {'type': 'key', 'pos': (12, 2), 'color': 'red'},   # 차고 열쇠
        {'type': 'ball', 'pos': (5, 8), 'color': 'blue'},  # 복도에 공
    ]
    
    room_config = {
        'start_pos': start_pos,
        'goal_pos': goal_pos,
        'walls': walls,
        'objects': objects
    }
    
    # Wrapper로 반환
    return CustomRoomWrapper(size=size, room_config=room_config)


def create_simple_room():
    """
    간단한 방 구조 예제
    
    Returns:
        CustomRoomWrapper: 간단한 방 환경 Wrapper 인스턴스
    """
    size = 8
    
    walls = [
        # 외벽
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
        (0, 7), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7),
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
        (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6),
        # 내부 벽
        (3, 2), (3, 3), (3, 4),
    ]
    
    room_config = {
        'start_pos': (1, 1),
        'goal_pos': (6, 6),
        'walls': walls,
        'objects': [
            {'type': 'key', 'pos': (5, 2), 'color': 'green'},
        ]
    }
    
    # Wrapper로 반환
    return CustomRoomWrapper(size=size, room_config=room_config)


def visualize_environment(wrapper):
    """
    환경을 시각화
    
    Args:
        wrapper: CustomRoomWrapper 인스턴스
    """
    wrapper.reset()
    img = wrapper.get_image()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title("Custom MiniGrid Environment")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('custom_environment.png', dpi=150, bbox_inches='tight')
    print("환경 이미지가 'custom_environment.png'로 저장되었습니다.")
    plt.show()


def test_environment(wrapper):
    """
    환경 테스트 (랜덤 액션 실행)
    
    Args:
        wrapper: CustomRoomWrapper 인스턴스
    """
    obs, info = wrapper.reset()
    done = False
    step_count = 0
    max_steps = 100
    
    print("환경 테스트 시작...")
    state = wrapper.get_state()
    print(f"시작 위치: {state['agent_pos']}")
    print(f"미션: {state['mission']}")
    
    while not done and step_count < max_steps:
        # 랜덤 액션
        action = wrapper.env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapper.step(action)
        done = terminated or truncated
        step_count += 1
        
        if step_count % 10 == 0:
            print(f"Step {step_count}: Reward={reward}, Done={done}")
    
    print(f"\n테스트 완료: 총 {step_count} 스텝, 최종 보상: {reward}")
    return obs, reward, done


def create_emoji_environment():
    """
    이모지 객체를 사용하는 환경 생성 예제
    
    Returns:
        CustomRoomWrapper: 이모지 객체가 포함된 환경
    """
    size = 10
    
    room_config = {
        'start_pos': (1, 1),
        'goal_pos': (8, 8),
        'walls': [],  # 외벽은 자동 생성
        'objects': [
            # 집을 수 없는 이모지 객체 (장애물)
            {'type': 'emoji', 'pos': (3, 3), 'emoji_name': 'tree', 'color': 'green', 'can_pickup': False},
            {'type': 'emoji', 'pos': (4, 4), 'emoji_name': 'rock', 'color': 'grey', 'can_pickup': False},
            {'type': 'emoji', 'pos': (5, 5), 'emoji_name': 'mountain', 'color': 'blue', 'can_pickup': False},
            
            # 집을 수 있는 이모지 객체
            {'type': 'emoji', 'pos': (2, 2), 'emoji_name': 'flower', 'color': 'yellow', 'can_pickup': True},
            {'type': 'emoji', 'pos': (6, 6), 'emoji_name': 'grass', 'color': 'green', 'can_pickup': True},
        ]
    }
    
    return CustomRoomWrapper(size=size, room_config=room_config)


def main():
    """
    메인 함수: 다양한 환경 생성 및 테스트
    """
    print("=" * 60)
    print("MiniGrid 커스텀 환경 생성 예제")
    print("=" * 60)
    
    # 예제 1: 간단한 방 구조
    print("\n[예제 1] 간단한 방 구조 생성")
    print("-" * 60)
    wrapper1 = create_simple_room()
    visualize_environment(wrapper1)
    test_environment(wrapper1)
    wrapper1.close()
    
    # 예제 1.5: 이모지 객체 사용 예제
    print("\n[예제 1.5] 이모지 객체 사용")
    print("-" * 60)
    emoji_wrapper = create_emoji_environment()
    emoji_wrapper.reset()
    
    # 그리드 파싱 테스트
    grid_map = emoji_wrapper.parse_grid()
    print("\n그리드 파싱 결과 (이모지 이름):")
    for (x, y), obj_name in grid_map.items():
        if obj_name is not None:
            print(f"  ({x}, {y}): {obj_name}")
    
    # 특정 위치의 이모지 확인
    emoji_at_3_3 = emoji_wrapper.get_emoji_at(3, 3)
    print(f"\n위치 (3, 3)의 이모지: {emoji_at_3_3}")
    
    visualize_environment(emoji_wrapper)
    emoji_wrapper.close()
    
    # 예제 2: 실내 집 환경
    print("\n[예제 2] 실내 집 환경 생성 (복도, 방, 차고)")
    print("-" * 60)
    wrapper2 = create_house_environment()
    visualize_environment(wrapper2)
    test_environment(wrapper2)
    wrapper2.close()
    
    # 예제 3: Wrapper 직접 사용
    print("\n[예제 3] Wrapper 직접 사용")
    print("-" * 60)
    custom_config = {
        'start_pos': (2, 2),
        'goal_pos': (10, 10),
        'walls': [
            # 외벽
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0),
            (0, 12), (1, 12), (2, 12), (3, 12), (4, 12), (5, 12), (6, 12), (7, 12), (8, 12), (9, 12), (10, 12), (11, 12), (12, 12),
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11),
            (12, 1), (12, 2), (12, 3), (12, 4), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11),
            # 내부 벽
            (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9),
        ],
        'objects': [
            {'type': 'key', 'pos': (4, 4), 'color': 'yellow'},
            {'type': 'ball', 'pos': (8, 8), 'color': 'blue'},
            {'type': 'box', 'pos': (3, 9), 'color': 'green'},
        ]
    }
    
    wrapper3 = CustomRoomWrapper(size=13, room_config=custom_config)
    visualize_environment(wrapper3)
    test_environment(wrapper3)
    wrapper3.close()
    
    # 예제 4: VLM 연동 예시
    print("\n[예제 4] VLM 연동 예시")
    print("-" * 60)
    wrapper4 = create_simple_room()
    wrapper4.reset()
    
    # 이미지 가져오기 (VLM에 전달)
    image = wrapper4.get_image()
    print(f"이미지 크기: {image.shape}")
    
    # 액션 공간 정보 확인
    action_space = wrapper4.get_action_space()
    print(f"액션 개수: {action_space['n']}")
    print(f"액션 목록: {action_space['actions']}")
    
    # VLM이 반환한 텍스트 액션을 실행
    vlm_actions = ["move forward", "turn right", "move forward", "pickup"]
    for action_str in vlm_actions:
        try:
            action = wrapper4.parse_action(action_str)
            obs, reward, done, truncated, info = wrapper4.step(action)
            print(f"액션 '{action_str}' 실행: Reward={reward}, Done={done}")
            if done:
                break
        except ValueError as e:
            print(f"에러: {e}")
    
    wrapper4.close()
    
    print("\n" + "=" * 60)
    print("모든 예제 실행 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
