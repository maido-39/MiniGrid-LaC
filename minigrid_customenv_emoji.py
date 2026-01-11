"""
MiniGrid 환경을 위한 Gym Wrapper (이모지 및 절대좌표 이동 지원)

이 모듈은 MiniGrid 환경을 쉽게 생성하고 제어할 수 있는 Wrapper 클래스를 제공합니다.
- 이모지 객체 지원 (emojiworld.py 기반)
- 절대 좌표 이동 지원 (상/하/좌/우 직접 이동)
- VLM(Vision Language Model)과의 연동을 고려하여 설계

주요 기능:
- 환경 초기화 시 size, walls, room_config 등을 지정
- 이모지 객체 배치 및 파싱
- 절대 방향 이동 (로봇 방향과 무관하게 상/하/좌/우 이동)
- 현재 환경 이미지 반환 (VLM 입력용)
- 액션 공간 제어 API
"""

from minigrid import register_minigrid_envs
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Goal, Key, Ball, Box, Door, WorldObj
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image, ImageDraw, ImageFont
import os

# MiniGrid 환경 등록
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
    'desktop': '🖥️',
    'workstation': '📱',
    'brick': '🧱',
}


class EmojiObject(WorldObj):
    """이모지를 표시하는 커스텀 객체"""
    
    def __init__(self, emoji_name: str, color: str = 'yellow', can_pickup: bool = False, can_overlap: bool = False, use_emoji_color: bool = True):
        super().__init__('box', color)
        self.emoji_name = emoji_name
        self._can_pickup = can_pickup
        self._can_overlap = can_overlap
        self.use_emoji_color = use_emoji_color  # True: 원래 이모지 색상 사용, False: 색 있는 선으로 그리기
        self.type = 'emoji'
        self.agent_on_top = False  # 로봇이 위에 있는지 여부
    
    def can_pickup(self):
        return self._can_pickup
    
    def can_overlap(self):
        return self._can_overlap
    
    def encode(self):
        from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
        obj_type_idx = OBJECT_TO_IDX['box']
        color_idx = COLOR_TO_IDX[self.color]
        return (obj_type_idx, color_idx, 0)
    
    def render(self, img):
        emoji_char = EMOJI_MAP.get(self.emoji_name, '❓')
        h, w = img.shape[:2]
        font_size = int(min(h, w) * 0.8)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # use_emoji_color=True일 때는 imagetext_py를 사용하여 컬러 이모지 렌더링
        if self.use_emoji_color:
            try:
                from imagetext_py import FontDB, Writer, Paint, TextAlign
                
                # 폰트 로드
                font_path = os.path.join(script_dir, 'fonts', 'NotoEmoji-Regular.ttf')
                if os.path.exists(font_path):
                    FontDB.LoadFromPath("NotoEmoji", font_path)
                    font = FontDB.Query("NotoEmoji")
                    
                    # 기존 이미지를 PIL Image로 변환
                    pil_img = Image.fromarray(img.astype(np.uint8)).convert('RGBA')
                    
                    # imagetext_py Writer를 사용하여 컬러 이모지 렌더링
                    with Writer(pil_img) as writer:
                        writer.draw_text_wrapped(
                            text=emoji_char,
                            x=w // 2,
                            y=h // 2,
                            ax=0.5,
                            ay=0.5,
                            size=font_size,
                            width=w,
                            font=font,
                            fill=Paint.Color((0, 0, 0, 255)),
                            align=TextAlign.Center,
                            draw_emojis=True  # 컬러 이모지 렌더링 활성화
                        )
                    
                    # 로봇이 위에 있으면 초록색 테두리 그리기
                    if self.agent_on_top:
                        draw = ImageDraw.Draw(pil_img)
                        border_width = 3
                        green_color = (0, 255, 0, 255)
                        draw.rectangle([(0, 0), (w-1, h-1)], outline=green_color, width=border_width)
                    
                    rgb_img = pil_img.convert('RGB')
                    img[:] = np.array(rgb_img)
                    return
            except ImportError:
                # imagetext_py가 없으면 에러 발생 (폴백 제거)
                raise ImportError("imagetext_py is required for use_emoji_color=True. Please install imagetext_py.")
            except (OSError, IOError, ValueError) as e:
                # 파일 관련 오류 시 에러 발생
                raise RuntimeError(f"Failed to render emoji with imagetext_py: {e}")
        
        # use_emoji_color=False일 때 PIL 사용 (단색)
        try:
            regular_font_path = os.path.join(script_dir, 'fonts', 'NotoEmoji-Regular.ttf')
            if os.path.exists(regular_font_path):
                font = ImageFont.truetype(regular_font_path, font_size)
            else:
                font = None
        except (OSError, IOError):
            font = None
        
        pil_img = Image.fromarray(img.astype(np.uint8)).convert('RGBA')
        draw = ImageDraw.Draw(pil_img)
        
        if font:
            try:
                bbox = draw.textbbox((0, 0), emoji_char, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except AttributeError:
                try:
                    text_width, text_height = draw.textsize(emoji_char, font=font)
                except (TypeError, ValueError):
                    text_width = font_size
                    text_height = font_size
            except (TypeError, ValueError):
                text_width = font_size
                text_height = font_size
        else:
            text_width = font_size
            text_height = font_size
        
        x = (w - text_width) // 2
        y = (h - text_height) // 2 - 2
        
        # 지정한 색상을 스트로크 색상으로 사용
        color_map = {
            'red': (255, 0, 0, 255),
            'green': (0, 255, 0, 255),
            'blue': (0, 0, 255, 255),
            'purple': (128, 0, 128, 255),
            'yellow': (255, 255, 0, 255),
            'grey': (128, 128, 128, 255),
        }
        stroke_color = color_map.get(self.color, (255, 255, 255, 255))
        
        if font:
            try:
                draw.text((x, y), emoji_char, font=font, fill=stroke_color)
            except (TypeError, ValueError, OSError):
                try:
                    draw.text((x, y), emoji_char, fill=stroke_color)
                except (TypeError, ValueError, OSError):
                    pass
        else:
            try:
                draw.text((x, y), emoji_char, fill=stroke_color)
            except (TypeError, ValueError, OSError):
                pass
        
        # 로봇이 위에 있으면 초록색 테두리 그리기
        if self.agent_on_top:
            border_width = 3
            green_color = (0, 255, 0, 255)  # 초록색
            draw.rectangle([(0, 0), (w-1, h-1)], outline=green_color, width=border_width)
        
        rgb_img = pil_img.convert('RGB')
        img[:] = np.array(rgb_img)
    
    def __str__(self):
        return self.emoji_name


def _gen_mission_default():
    """기본 미션 생성 함수"""
    return "explore"


class CustomRoomEnv(MiniGridEnv):
    """커스텀 방 구조를 가진 MiniGrid 환경 클래스"""
    
    def __init__(self, size=10, room_config=None, **kwargs):
        self.size = size
        self.room_config = room_config or {}
        mission_space = MissionSpace(mission_func=_gen_mission_default)
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=4 * size * size,
            **kwargs
        )
    
    def _gen_mission(self):
        return "explore"
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        
        if self.room_config:
            if 'walls' in self.room_config:
                for wall_info in self.room_config['walls']:
                    if isinstance(wall_info, tuple):
                        if len(wall_info) == 2:
                            wall_x, wall_y = wall_info
                            wall_color = 'grey'
                        elif len(wall_info) == 3:
                            wall_x, wall_y, wall_color = wall_info
                        else:
                            continue
                    elif isinstance(wall_info, dict):
                        wall_pos = wall_info.get('pos', (0, 0))
                        wall_x, wall_y = wall_pos
                        wall_color = wall_info.get('color', 'grey')
                    else:
                        continue
                    
                    if 0 <= wall_x < width and 0 <= wall_y < height:
                        self.grid.set(wall_x, wall_y, Wall(wall_color))
            
            if 'goal_pos' in self.room_config:
                goal_x, goal_y = self.room_config['goal_pos']
                if 0 <= goal_x < width and 0 <= goal_y < height:
                    self.put_obj(Goal(), goal_x, goal_y)
            
            if 'objects' in self.room_config:
                for obj_info in self.room_config['objects']:
                    obj_type = obj_info.get('type', 'key')
                    obj_pos = obj_info.get('pos', (1, 1))
                    obj_color = obj_info.get('color', 'yellow')
                    
                    obj_x, obj_y = obj_pos
                    if 0 <= obj_x < width and 0 <= obj_y < height:
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
                            emoji_name = obj_info.get('emoji_name', 'emoji')
                            can_pickup = obj_info.get('can_pickup', False)
                            can_overlap = obj_info.get('can_overlap', False)
                            use_emoji_color = obj_info.get('use_emoji_color', True)
                            obj = EmojiObject(
                                emoji_name=emoji_name, 
                                color=obj_color, 
                                can_pickup=can_pickup,
                                can_overlap=can_overlap,
                                use_emoji_color=use_emoji_color
                            )
                        else:
                            obj = Key(obj_color)
                        
                        self.put_obj(obj, obj_x, obj_y)
        
        if self.room_config and 'start_pos' in self.room_config:
            start_x, start_y = self.room_config['start_pos']
            self.agent_pos = np.array([start_x, start_y])
            self.agent_dir = 0
        else:
            self.place_agent()
        
        self.mission = self._gen_mission()
    
    def render(self):
        # 로봇 위치 확인 및 이모지 객체에 표시
        if hasattr(self, 'agent_pos') and hasattr(self, 'grid'):
            agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
            # 모든 이모지 객체의 agent_on_top 초기화
            for y in range(self.grid.height):
                for x in range(self.grid.width):
                    cell = self.grid.get(x, y)
                    if cell is not None and hasattr(cell, 'type') and cell.type == 'emoji':
                        cell.agent_on_top = (x == agent_x and y == agent_y)
        
        frame = super().render()
        if frame is None:
            return frame
        
        if not hasattr(self, 'agent_pos') or not hasattr(self, 'agent_dir'):
            return frame
        
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        agent_dir = self.agent_dir
        actual_tile_size = self.tile_size if hasattr(self, 'tile_size') else 32
        
        start_x = agent_x * actual_tile_size
        start_y = agent_y * actual_tile_size
        end_x = start_x + actual_tile_size
        end_y = start_y + actual_tile_size
        
        frame_h, frame_w = frame.shape[:2]
        if start_x < 0 or start_y < 0 or end_x > frame_w or end_y > frame_h:
            return frame
        
        try:
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # room_config에서 use_robot_emoji 확인
            use_robot_emoji = self.room_config.get('use_robot_emoji', False)
            robot_emoji_char = '🤖' if use_robot_emoji else None
            robot_emoji_color = self.room_config.get('robot_emoji_color', 'yellow')
            use_robot_emoji_color = self.room_config.get('use_robot_emoji_color', False)
            
            pil_frame = Image.fromarray(frame.astype(np.uint8)).convert('RGBA')
            
            cell = self.grid.get(agent_x, agent_y)
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
            
            # 로봇 이모지 모드인 경우
            if use_robot_emoji and robot_emoji_char:
                font_size = int(actual_tile_size * 0.8)
                
                # use_robot_emoji_color=True일 때는 imagetext_py를 사용하여 컬러 이모지 렌더링
                if use_robot_emoji_color:
                    try:
                        from imagetext_py import FontDB, Writer, Paint, TextAlign
                        
                        # 폰트 로드
                        font_path = os.path.join(script_dir, 'fonts', 'NotoEmoji-Regular.ttf')
                        if os.path.exists(font_path):
                            FontDB.LoadFromPath("NotoEmoji", font_path)
                            font = FontDB.Query("NotoEmoji")
                            
                            # imagetext_py Writer를 사용하여 컬러 이모지 렌더링
                            with Writer(bg_tile) as writer:
                                writer.draw_text_wrapped(
                                    text=robot_emoji_char,
                                    x=actual_tile_size // 2,
                                    y=actual_tile_size // 2,
                                    ax=0.5,
                                    ay=0.5,
                                    size=font_size,
                                    width=actual_tile_size,
                                    font=font,
                                    fill=Paint.Color((0, 0, 0, 255)),
                                    align=TextAlign.Center,
                                    draw_emojis=True  # 컬러 이모지 렌더링 활성화
                                )
                    except ImportError:
                        # imagetext_py가 없으면 PIL로 폴백 (단색)
                        use_robot_emoji_color = False
                    except (OSError, IOError, ValueError):
                        # imagetext_py 오류 시 PIL로 폴백
                        use_robot_emoji_color = False
                
                # use_robot_emoji_color=False일 때 또는 imagetext_py 사용 실패 시 PIL 사용 (단색)
                if not use_robot_emoji_color:
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
                            bbox = draw.textbbox((0, 0), robot_emoji_char, font=font)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]
                        except AttributeError:
                            try:
                                text_width, text_height = draw.textsize(robot_emoji_char, font=font)
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
                    
                    # 색상 맵 (다른 오브젝트들과 동일한 색상 시스템 사용)
                    color_map = {
                        'red': (255, 0, 0, 255),
                        'green': (0, 255, 0, 255),
                        'blue': (0, 0, 255, 255),
                        'purple': (128, 0, 128, 255),
                        'yellow': (255, 255, 0, 255),
                        'grey': (128, 128, 128, 255),
                    }
                    fill_color = color_map.get(robot_emoji_color, (255, 255, 255, 255))
                    if font:
                        try:
                            draw.text((x, y), robot_emoji_char, font=font, fill=fill_color)
                        except:
                            try:
                                draw.text((x, y), robot_emoji_char, fill=fill_color)
                            except:
                                pass
                    else:
                        try:
                            draw.text((x, y), robot_emoji_char, fill=fill_color)
                        except:
                            pass
            else:
                # arrow.png 이미지 모드 (기본)
                arrow_img_path = os.path.join(script_dir, 'asset', 'arrow.png')
                
                if os.path.exists(arrow_img_path):
                    arrow_img = Image.open(arrow_img_path).convert('RGBA')
                    arrow_img = arrow_img.resize((actual_tile_size, actual_tile_size), Image.Resampling.LANCZOS)
                    
                    rotation_map = {0: 0, 1: 90, 2: 180, 3: 270}
                    rotation_angle = rotation_map.get(agent_dir, 0)
                    
                    if rotation_angle != 0:
                        arrow_img = arrow_img.rotate(-rotation_angle, expand=False, fillcolor=(0, 0, 0, 0))
                    
                    bg_tile.paste(arrow_img, (0, 0), arrow_img)
            
            pil_frame.paste(bg_tile, (start_x, start_y))
            frame = np.array(pil_frame.convert('RGB'))
        except Exception:
            pass
        
        return frame


class MiniGridEmojiWrapper:
    """
    MiniGrid 환경을 제어하기 위한 Wrapper 클래스 (이모지 및 절대좌표 이동 지원)
    
    주요 기능:
    - 이모지 객체 배치 및 파싱
    - 절대 방향 이동 (상/하/좌/우 직접 이동)
    - 환경 생성, 조작, 정보 파싱 API
    """
    
    # 기본 액션 (상대 방향)
    ACTION_NAMES = {
        0: "turn left",
        1: "turn right", 
        2: "move forward",
        3: "move backward",
        4: "pickup",
        5: "drop",
        6: "toggle"
    }
    
    ACTION_ALIASES = {
        "turn left": 0, "left": 0, "rotate left": 0, "turn_left": 0,
        "turn right": 1, "right": 1, "rotate right": 1, "turn_right": 1,
        "move forward": 2, "forward": 2, "go forward": 2, "move_forward": 2, "w": 2,
        "move backward": 3, "backward": 3, "go backward": 3, "move_backward": 3, "s": 3,
        "pickup": 4, "pick up": 4, "pick_up": 4, "grab": 4,
        "drop": 5, "put down": 5, "put_down": 5, "release": 5,
        "toggle": 6, "interact": 6, "use": 6, "activate": 6
    }
    
    # 절대 방향 액션
    ABSOLUTE_ACTION_NAMES = {
        0: "move up",      # North
        1: "move down",    # South
        2: "move left",    # West
        3: "move right",   # East
        4: "pickup",
        5: "drop",
        6: "toggle"
    }
    
    ABSOLUTE_ACTION_ALIASES = {
        "move up": 0, "up": 0, "north": 0, "n": 0, "move north": 0, "go up": 0, "go north": 0,
        "move down": 1, "down": 1, "south": 1, "s": 1, "move south": 1, "go down": 1, "go south": 1,
        "move left": 2, "left": 2, "west": 2, "w": 2, "move west": 2, "go left": 2, "go west": 2,
        "move right": 3, "right": 3, "east": 3, "e": 3, "move east": 3, "go right": 3, "go east": 3,
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
        use_absolute_movement: bool = True,
        **kwargs
    ):
        """
        Args:
            use_absolute_movement: 절대 움직임 모드 활성화 여부 (기본값: True)
                - True: step() 메서드가 절대 방향 액션을 받아 처리 (표준)
                - False: step() 메서드가 상대 방향 액션을 받아 처리 (레거시)
        """
        self.size = size
        self.walls = walls or []
        self.render_mode = render_mode
        self.use_absolute_movement = use_absolute_movement
        
        if room_config is None:
            room_config = {}
        
        if walls and 'walls' not in room_config:
            existing_walls = room_config.get('walls', [])
            room_config['walls'] = existing_walls + walls
        
        self.env = CustomRoomEnv(
            size=size,
            room_config=room_config,
            render_mode=render_mode,
            **kwargs
        )
        
        self.current_obs = None
        self.current_info = None
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        self.current_obs, self.current_info = self.env.reset(seed=seed)
        return self.current_obs, self.current_info
    
    def step(self, action: Union[int, str]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        액션 실행
        
        use_absolute_movement=True인 경우 절대 방향 액션으로 처리,
        False인 경우 상대 방향 액션으로 처리합니다.
        """
        if self.use_absolute_movement:
            # 절대 움직임 모드: step_absolute 사용
            return self.step_absolute(action)
        else:
            # 상대 움직임 모드: 기존 방식
            if isinstance(action, str):
                action = self.parse_action(action)
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.current_obs = obs
            self.current_info = info
            return obs, reward, terminated, truncated, info
    
    def _get_target_direction(self, absolute_action: int) -> int:
        """절대 액션을 MiniGrid 방향으로 변환"""
        direction_map = {0: 3, 1: 1, 2: 2, 3: 0}  # up->North, down->South, left->West, right->East
        return direction_map.get(absolute_action, 0)
    
    def _calculate_rotation(self, current_dir: int, target_dir: int) -> list:
        """현재 방향에서 목표 방향으로 회전하기 위한 액션 시퀀스 계산"""
        if current_dir == target_dir:
            return []
        
        diff = (target_dir - current_dir) % 4
        
        if diff == 1:
            return [1]  # turn right
        elif diff == 2:
            return [1, 1]  # turn right twice
        elif diff == 3:
            return [0]  # turn left
        
        return []
    
    def step_absolute(self, action: Union[int, str]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        절대 방향 액션을 실행
        
        Args:
            action: 절대 방향 액션 (정수 인덱스 또는 액션 이름 문자열)
                - 0 또는 "move up": 위로 이동 (North)
                - 1 또는 "move down": 아래로 이동 (South)
                - 2 또는 "move left": 왼쪽으로 이동 (West)
                - 3 또는 "move right": 오른쪽으로 이동 (East)
                - 4 또는 "pickup": 물체 집기
                - 5 또는 "drop": 물체 놓기
                - 6 또는 "toggle": 상호작용
        """
        if isinstance(action, str):
            action = self.parse_absolute_action(action)
        
        if action >= 4:
            return self.step(action)
        
        current_dir = self.env.agent_dir
        target_dir = self._get_target_direction(action)
        
        rotation_actions = self._calculate_rotation(current_dir, target_dir)
        
        for rot_action in rotation_actions:
            obs, reward, terminated, truncated, info = self.step(rot_action)
            if terminated or truncated:
                return obs, reward, terminated, truncated, info
        
        obs, reward, terminated, truncated, info = self.step(2)  # move forward
        return obs, reward, terminated, truncated, info
    
    def get_image(self, fov_range: Optional[int] = None, fov_width: Optional[int] = None) -> np.ndarray:
        """현재 환경의 이미지를 반환 (VLM 입력용)"""
        image = self.env.render()
        if image is None:
            return np.zeros((self.size * 32, self.size * 32, 3), dtype=np.uint8)
        
        if fov_range is not None and fov_width is not None:
            image = self._apply_fog_of_war(image, fov_range, fov_width)
        
        return image
    
    def _apply_fog_of_war(self, image: np.ndarray, fov_range: int, fov_width: int) -> np.ndarray:
        """시야 제한을 적용하여 시야 밖의 영역을 검은색으로 마스킹"""
        if not hasattr(self.env, 'agent_pos') or not hasattr(self.env, 'agent_dir'):
            return image
        
        agent_pos = self.env.agent_pos
        if isinstance(agent_pos, np.ndarray):
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        else:
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        
        agent_dir = self.env.agent_dir
        
        masked_image = image.copy()
        h, w = image.shape[:2]
        cell_size = 32
        grid_size = self.size
        
        for grid_y in range(grid_size):
            for grid_x in range(grid_size):
                dx = grid_x - agent_x
                dy = grid_y - agent_y
                
                if agent_dir == 0:
                    rel_x, rel_y = dx, -dy
                elif agent_dir == 1:
                    rel_x, rel_y = dy, dx
                elif agent_dir == 2:
                    rel_x, rel_y = -dx, dy
                else:
                    rel_x, rel_y = -dy, -dx
                
                in_fov = (
                    rel_x >= 0 and
                    rel_x <= fov_range and
                    abs(rel_y) <= fov_width // 2
                )
                
                if not in_fov:
                    pixel_x = grid_x * cell_size
                    pixel_y = grid_y * cell_size
                    end_x = min(pixel_x + cell_size, w)
                    end_y = min(pixel_y + cell_size, h)
                    masked_image[pixel_y:end_y, pixel_x:end_x] = [0, 0, 0]
        
        return masked_image
    
    def get_action_space(self) -> Dict:
        """액션 공간 정보 반환"""
        return {
            'n': self.env.action_space.n,
            'actions': list(self.ACTION_NAMES.values()),
            'action_mapping': self.ACTION_NAMES,
            'action_aliases': self.ACTION_ALIASES
        }
    
    def get_absolute_action_space(self) -> Dict:
        """절대 방향 액션 공간 정보 반환"""
        return {
            'n': 7,
            'actions': list(self.ABSOLUTE_ACTION_NAMES.values()),
            'action_mapping': self.ABSOLUTE_ACTION_NAMES,
            'action_aliases': self.ABSOLUTE_ACTION_ALIASES
        }
    
    def parse_action(self, action_str: str) -> int:
        """액션 문자열을 인덱스로 변환 (상대 방향)"""
        action_str = action_str.strip()
        
        try:
            action_int = int(action_str)
            if 0 <= action_int < self.env.action_space.n:
                return action_int
        except ValueError:
            pass
        
        action_str_lower = action_str.lower()
        
        if action_str_lower in self.ACTION_ALIASES:
            return self.ACTION_ALIASES[action_str_lower]
        
        for idx, name in self.ACTION_NAMES.items():
            if action_str_lower == name.lower():
                return idx
        
        raise ValueError(
            f"Unknown action: '{action_str}'. "
            f"Available actions: {list(self.ACTION_ALIASES.keys())} or numbers 0-{self.env.action_space.n-1}"
        )
    
    def parse_absolute_action(self, action_str: str) -> int:
        """절대 방향 액션 문자열을 인덱스로 변환"""
        action_str = action_str.strip()
        
        try:
            action_int = int(action_str)
            if 0 <= action_int <= 6:
                return action_int
        except ValueError:
            pass
        
        action_str_lower = action_str.lower()
        
        if action_str_lower in self.ABSOLUTE_ACTION_ALIASES:
            return self.ABSOLUTE_ACTION_ALIASES[action_str_lower]
        
        for idx, name in self.ABSOLUTE_ACTION_NAMES.items():
            if action_str_lower == name.lower():
                return idx
        
        raise ValueError(
            f"Unknown absolute action: '{action_str}'. "
            f"Available actions: {list(self.ABSOLUTE_ACTION_ALIASES.keys())} or numbers 0-6"
        )
    
    def get_state(self) -> Dict:
        """현재 환경 상태 정보 반환"""
        agent_pos = None
        if hasattr(self.env, 'agent_pos'):
            if isinstance(self.env.agent_pos, np.ndarray):
                agent_pos = self.env.agent_pos.copy()
            else:
                agent_pos = self.env.agent_pos
        
        return {
            'agent_pos': agent_pos,
            'agent_dir': self.env.agent_dir if hasattr(self.env, 'agent_dir') else None,
            'mission': self.env.mission if hasattr(self.env, 'mission') else None,
            'image': self.get_image()
        }
    
    def parse_grid(self) -> Dict[Tuple[int, int], str]:
        """그리드를 파싱하여 각 위치의 객체 정보를 반환"""
        grid_map = {}
        
        if not hasattr(self.env, 'grid'):
            return grid_map
        
        width = self.env.grid.width
        height = self.env.grid.height
        
        for y in range(height):
            for x in range(width):
                cell = self.env.grid.get(x, y)
                
                if cell is None:
                    grid_map[(x, y)] = None
                elif hasattr(cell, 'type'):
                    if cell.type == 'emoji' and hasattr(cell, 'emoji_name'):
                        grid_map[(x, y)] = cell.emoji_name
                    else:
                        grid_map[(x, y)] = cell.type
                else:
                    grid_map[(x, y)] = str(cell)
        
        return grid_map
    
    def get_emoji_at(self, x: int, y: int) -> Optional[str]:
        """특정 위치의 이모지 이름을 반환"""
        if not hasattr(self.env, 'grid'):
            return None
        
        cell = self.env.grid.get(x, y)
        
        if cell is None:
            return None
        
        if hasattr(cell, 'type') and cell.type == 'emoji' and hasattr(cell, 'emoji_name'):
            return cell.emoji_name
        
        return None
    
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
    
    def close(self):
        """환경 종료 및 리소스 정리"""
        self.env.close()

