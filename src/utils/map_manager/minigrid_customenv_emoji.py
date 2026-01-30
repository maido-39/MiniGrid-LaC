"""
Gym Wrapper for MiniGrid Environment (Emoji and Absolute Coordinate Movement Support)

This module provides a Wrapper class for easily creating and controlling MiniGrid environments.
- Emoji object support (based on emojiworld.py)
- Absolute coordinate movement support (direct up/down/left/right movement)
- Designed with VLM (Vision Language Model) integration in mind

Key features:
- Specify size, walls, room_config, etc. during environment initialization
- Emoji object placement and parsing
- Absolute direction movement (up/down/left/right movement regardless of robot direction)
- Return current environment image (for VLM input)
- Action space control API
"""

from minigrid import register_minigrid_envs
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Goal, Key, Ball, Box, Door, WorldObj
from minigrid.core.mission import MissionSpace
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.minigrid_env import MiniGridEnv
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image, ImageDraw, ImageFont
import os
import hashlib
from imagetext_py import FontDB, Writer, Paint, TextAlign
from utils.miscellaneous.global_variables import RENDER_GOAL
from utils.map_manager.carrying_format import format_carrying_object as format_carrying_object_util

# Register MiniGrid environments
register_minigrid_envs()

# Extend COLOR_TO_IDX with additional colors for emoji objects
# This allows using more than the default 6 minigrid colors
_base_color_count = len(COLOR_TO_IDX)
EXTENDED_COLORS = {
    'orange': _base_color_count,
    'brown': _base_color_count + 1,
    'pink': _base_color_count + 2,
    'cyan': _base_color_count + 3,
    'lime': _base_color_count + 4,
    'navy': _base_color_count + 5,
    'teal': _base_color_count + 6,
    'magenta': _base_color_count + 7,
    'olive': _base_color_count + 8,
    'maroon': _base_color_count + 9,
    'white': _base_color_count + 10,
    'black': _base_color_count + 11,
}
COLOR_TO_IDX.update(EXTENDED_COLORS)

# Emoji name to actual emoji character mapping
EMOJI_MAP = {
    'TV' : '📺',
    'sofa' : '🛋️',
    'bed' : '🛏️',
    'desk' : '🖥️',
    'lamp' : '💡',
    'wall' : '🚧',
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
    'restroom' : '🚻',
    'storage' : '🗄️',
    'preperation' : '🧑‍🍳',
    'kitchen' : '🍳',
    'plating' : '🍽️',
    'dining' : '🍴',
    'water' : '💦',
    'waterspill' : '🫗',
    'broom' : '🧹',
    "apple" : "🍏",
    "lemon" : "🍋",
    "tomato" : "🍅",
    "bellpepper" : "🫑",
    "carrot" : "🥕",
    "hotpepper" : "🌶️",
    "toilet" : "🚽",
    "shower": "🚿",
    "sink": "🚰",
    "men_restroom": "🚹",
    "women_restroom": "🚺",
    "maintenance": "🚧",
    "bed": "🛏️",
    "laundry": "🧺",
    "soap": "🧼",
    "tools": "🛠️",
    "shirt": "👕",
    "short": "🩳",
    "dress": "👗",
    "blouse": "👚",
    "robot": "🤖",
}

# Color map for RGBA format (with alpha channel)
COLOR_MAP_RGBA = {
    'red': (255, 0, 0, 255),
    'green': (0, 255, 0, 255),
    'blue': (0, 0, 255, 255),
    'purple': (128, 0, 128, 255),
    'yellow': (255, 255, 0, 255),
    'grey': (128, 128, 128, 255),
    'orange': (255, 165, 0, 255),
    'brown': (165, 42, 42, 255),
    'pink': (255, 192, 203, 255),
    'cyan': (0, 255, 255, 255),
    'lime': (50, 205, 50, 255),
    'navy': (0, 0, 128, 255),
    'teal': (0, 128, 128, 255),
    'magenta': (255, 0, 255, 255),
    'olive': (128, 128, 0, 255),
    'maroon': (128, 0, 0, 255),
    'white': (255, 255, 255, 255),
    'black': (0, 0, 0, 255),
}

# Color map for RGB format (without alpha channel)
COLOR_MAP_RGB = {
    'red': (200, 100, 100),
    'green': (100, 200, 100),
    'blue': (100, 100, 200),
    'purple': (150, 100, 150),
    'yellow': (200, 200, 100),
    'grey': (128, 128, 128),
    'orange': (255, 165, 0),
    'brown': (165, 42, 42),
    'pink': (255, 192, 203),
    'cyan': (0, 255, 255),
    'lime': (50, 205, 50),
    'navy': (0, 0, 128),
    'teal': (0, 128, 128),
    'magenta': (255, 0, 255),
    'olive': (128, 128, 0),
    'maroon': (128, 0, 0),
    'white': (240, 240, 240),
    'black': (50, 50, 50),
}



class EmojiObject(WorldObj):
    """Custom object that displays emoji"""
    
    def __init__(self, emoji_name: str, color: str = 'yellow', can_pickup: bool = False, can_overlap: bool = False, use_emoji_color: bool = True):
        super().__init__('box', color)
        self.emoji_name = emoji_name
        self._can_pickup = can_pickup
        self._can_overlap = can_overlap
        self.use_emoji_color = use_emoji_color  # True: use original emoji color, False: draw with colored stroke
        self.type = 'emoji'
        self.agent_on_top = False  # Whether robot is on top
    
    def can_pickup(self):
        return self._can_pickup
    
    def can_overlap(self):
        return self._can_overlap
    
    @property
    def cache_key(self):
        """
        A cache key used for rendering.
        Includes the object's ID to ensure absolute uniqueness.
        """
        return (self.type, self.emoji_name, self.color, id(self))
    
    def encode(self):
        obj_type_idx = OBJECT_TO_IDX['box']
        color_idx = COLOR_TO_IDX.get(self.color, COLOR_TO_IDX['grey'])
        
        # Use emoji_name hash to create unique state value
        # This ensures different emojis with same color get different encodings
        # This prevents rendering issues when multiple emojis share the same color
        emoji_hash = int(hashlib.md5(self.emoji_name.encode()).hexdigest()[:8], 16)
        state = emoji_hash % 256  # Keep within 0-255 range for minigrid compatibility
        
        return (obj_type_idx, color_idx, state)
    
    def render(self, img):
        emoji_char = EMOJI_MAP.get(self.emoji_name, '❓')
        h, w = img.shape[:2]
        font_size = int(min(h, w) * 0.8)
        
        # Find src/ directory (go up from utils/map_manager to find src/)
        current_file = os.path.abspath(__file__)
        src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # src/ directory
        
        # When use_emoji_color=True, use imagetext_py for color emoji rendering
        if self.use_emoji_color:
            # Load font from src/asset/fonts/
            font_path = os.path.join(src_dir, 'asset', 'fonts', 'NotoEmoji-Regular.ttf')
            if os.path.exists(font_path):
                FontDB.LoadFromPath("NotoEmoji", font_path)
                font = FontDB.Query("NotoEmoji")
                
                # Convert existing image to PIL Image
                pil_img = Image.fromarray(img.astype(np.uint8)).convert('RGBA')
                
                # Render color emoji using imagetext_py Writer
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
                        draw_emojis=True  # Enable color emoji rendering
                    )
                
                # Draw green border if robot is on top
                if self.agent_on_top:
                    draw = ImageDraw.Draw(pil_img)
                    border_width = 3
                    green_color = (0, 255, 0, 255)
                    draw.rectangle([(0, 0), (w-1, h-1)], outline=green_color, width=border_width)
                
                rgb_img = pil_img.convert('RGB')
                img[:] = np.array(rgb_img)
                return
        
        # Use PIL when use_emoji_color=False (monochrome)
        try:
            regular_font_path = os.path.join(src_dir, 'asset', 'fonts', 'NotoEmoji-Regular.ttf')
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
        
        # Use specified color as stroke color
        stroke_color = COLOR_MAP_RGBA.get(self.color, (255, 255, 255, 255))
        
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
        
        # Draw green border if robot is on top
        if self.agent_on_top:
            border_width = 3
            green_color = (0, 255, 0, 255)  # Green
            draw.rectangle([(0, 0), (w-1, h-1)], outline=green_color, width=border_width)
        
        rgb_img = pil_img.convert('RGB')
        img[:] = np.array(rgb_img)
    
    def __str__(self):
        return self.emoji_name


def _gen_mission_default():
    """Default mission generation function"""
    return "explore"


class CustomRoomEnv(MiniGridEnv):
    """MiniGrid environment class with custom room structure"""
    
    def __init__(self, size=10, room_config=None, **kwargs):
        self.size = size
        self.room_config = room_config or {}
        # Store floor tile information (color by position)
        self.floor_tiles = {}  # {(x, y): color}
        mission_space = MissionSpace(mission_func=_gen_mission_default)
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=4 * size * size,
            **kwargs
        )
        
        # Disable MiniGrid's default FOV highlighting to allow full visibility
        # This ensures that when FOV is disabled, all cells have the same background color
        # Can be overridden via room_config['highlight']
        self.highlight = self.room_config.get('highlight', False)
        
        # Set agent_view_size to grid size to disable view limitations
        # This allows the agent to see the entire grid regardless of position
        # Can be overridden via room_config['agent_view_size']
        self.agent_view_size = self.room_config.get('agent_view_size', size)
    
    def _gen_mission(self):
        return "explore"
    
    def _format_carrying_object(self, carrying_obj) -> str:
        """Format carrying object(s) for terminal display; delegates to shared carrying_format."""
        return format_carrying_object_util(
            carrying_obj,
            emoji_objects=getattr(self, 'room_config', {}).get('emoji_objects') if getattr(self, 'room_config', None) else None,
        )
    
    def _print_carrying_status(self):
        """Print current carrying status to terminal"""
        if hasattr(self, 'carrying'):
            if isinstance(self.carrying, list):
                if len(self.carrying) > 0:
                    carrying_info = self._format_carrying_object(self.carrying)
                    print(f"[API] Carrying Objects ({len(self.carrying)}): {carrying_info}")
                else:
                    print("[API] Carrying Objects: None")
            elif self.carrying is not None:
                carrying_info = self._format_carrying_object(self.carrying)
                print(f"[API] Carrying Object: {carrying_info}")
            else:
                print("[API] Carrying Object: None")
        else:
            print("[API] Carrying Object: None")
    
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
            
            # Goal 렌더링 (RENDER_GOAL 설정에 따라 켜고 끄기)
            if RENDER_GOAL and 'goal_pos' in self.room_config:
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
            
            # Store floor tile information (floor color at specific positions)
            # floor_tiles is passed as dictionary: {(x, y): color}
            if 'floor_tiles' in self.room_config:
                floor_tiles_data = self.room_config['floor_tiles']
                if isinstance(floor_tiles_data, dict):
                    # Dictionary format: {(x, y): color}
                    # Store floor tiles in external coordinates (for consistency with external API)
                    for (floor_x, floor_y), floor_color in floor_tiles_data.items():
                        if 0 <= floor_x < width and 0 <= floor_y < height:
                            self.floor_tiles[(floor_x, floor_y)] = floor_color
                elif isinstance(floor_tiles_data, list):
                    # List format (backward compatibility): [{'pos': (x, y), 'color': 'blue'}, ...]
                    for floor_info in floor_tiles_data:
                        if isinstance(floor_info, tuple):
                            if len(floor_info) == 2:
                                floor_x, floor_y = floor_info
                                floor_color = 'grey'
                            elif len(floor_info) == 3:
                                floor_x, floor_y, floor_color = floor_info
                            else:
                                continue
                        elif isinstance(floor_info, dict):
                            floor_pos = floor_info.get('pos', (0, 0))
                            floor_x, floor_y = floor_pos
                            floor_color = floor_info.get('color', 'grey')
                        else:
                            continue
                        
                        if 0 <= floor_x < width and 0 <= floor_y < height:
                            self.floor_tiles[(floor_x, floor_y)] = floor_color
        
        if self.room_config and 'start_pos' in self.room_config:
            start_x, start_y = self.room_config['start_pos']
            self.agent_pos = np.array([start_x, start_y])
            self.agent_dir = 0
        else:
            self.place_agent()
        
        # Initialize carrying as a list to support multiple objects
        self.carrying = []
        
        self.mission = self._gen_mission()
    
    def step(self, action):
        """
        Override step method to handle pickup and drop actions for multiple objects
        
        Supports carrying multiple objects (self.carrying is a list).
        MiniGrid's default step method only supports one object, so we handle
        pickup and drop actions manually.
        """
        # Initialize carrying as list if not already
        if not hasattr(self, 'carrying') or not isinstance(self.carrying, list):
            self.carrying = []
        
        # Handle pickup action (action 3 in MiniGrid's action space)
        if action == 3:  # MiniGrid's pickup action
            # Get front cell position
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)
            
            # Check if front cell has an object that can be picked up
            if fwd_cell:
                can_pickup = True
                if hasattr(fwd_cell, 'can_pickup'):
                    can_pickup = fwd_cell.can_pickup()
                
                # #region agent log
                with open('/home/syaro/DeepL_WS/multigrid-LaC/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        'sessionId': 'debug-session',
                        'runId': 'run1',
                        'hypothesisId': 'D',
                        'location': 'minigrid_customenv_emoji.py:548',
                        'message': 'can_pickup check',
                        'data': {'can_pickup': can_pickup, 'has_can_pickup_method': hasattr(fwd_cell, 'can_pickup')},
                        'timestamp': int(__import__('time').time() * 1000)
                    }) + '\n')
                # #endregion
                
                if can_pickup:
                    # Remove object from grid
                    self.grid.set(*fwd_pos, None)
                    # Add to carrying list
                    self.carrying.append(fwd_cell)
                    # Print carrying status to terminal
                    self._print_carrying_status()
                    # Return success (no reward change, just pickup)
                    # Temporarily set carrying to single object for gen_obs() compatibility
                    temp_carrying = self.carrying
                    if len(self.carrying) > 0:
                        self.carrying = self.carrying[-1]  # Use last picked object
                    else:
                        self.carrying = None
                    obs = self.gen_obs()
                    # Restore carrying list
                    self.carrying = temp_carrying
                    reward = 0.0
                    terminated = False
                    truncated = False
                    info = {}
                    return obs, reward, terminated, truncated, info
        
        # Handle drop action (action 4 in MiniGrid's action space)
        if action == 4:  # MiniGrid's drop action
            if len(self.carrying) > 0:
                # Get front cell position
                fwd_pos = self.front_pos
                fwd_cell = self.grid.get(*fwd_pos)
                
                # Only drop if front cell is empty
                if fwd_cell is None:
                    # Remove last object from carrying list
                    obj_to_drop = self.carrying.pop()
                    # Place object on grid
                    self.grid.set(*fwd_pos, obj_to_drop)
                    # Print carrying status to terminal
                    self._print_carrying_status()
                    # Return success
                    # Temporarily set carrying to single object (or None) for gen_obs() compatibility
                    temp_carrying = self.carrying
                    if len(self.carrying) > 0:
                        self.carrying = self.carrying[-1]  # Use last remaining object
                    else:
                        self.carrying = None
                    obs = self.gen_obs()
                    # Restore carrying list
                    self.carrying = temp_carrying
                    reward = 0.0
                    terminated = False
                    truncated = False
                    info = {}
                    return obs, reward, terminated, truncated, info
        
        # For all other actions (movement, toggle), call parent step
        # But we need to temporarily set self.carrying to None or first item
        # for compatibility with MiniGrid's step method (e.g., toggle checks for key)
        original_carrying = self.carrying.copy() if isinstance(self.carrying, list) else self.carrying
        if len(self.carrying) > 0:
            # MiniGrid expects single object, so use first one for compatibility
            # This is important for toggle action (opening doors with keys)
            self.carrying = self.carrying[0]
        else:
            self.carrying = None
        
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Restore carrying list
        # Note: MiniGrid's step should not modify carrying for movement/toggle actions,
        # but we restore it anyway to be safe
        if isinstance(original_carrying, list):
            self.carrying = original_carrying
        else:
            # Fallback: if original wasn't a list, keep parent's result
            if self.carrying is not None and not isinstance(self.carrying, list):
                self.carrying = [self.carrying]
            elif self.carrying is None:
                self.carrying = []
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        # Check robot position and mark on emoji objects
        if hasattr(self, 'agent_pos') and hasattr(self, 'grid'):
            agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
            # Initialize agent_on_top for all emoji objects
            for y in range(self.grid.height):
                for x in range(self.grid.width):
                    cell = self.grid.get(x, y)
                    if cell is not None and hasattr(cell, 'type') and cell.type == 'emoji':
                        cell.agent_on_top = (x == agent_x and y == agent_y)
        
        frame = super().render()
        if frame is None:
            return frame
        
        # Apply background color (before floor colors)
        background_color = self.room_config.get('background_color', None)
        if background_color is not None:
            frame = self._apply_background_color(frame, background_color)
        
        floor_color = self.room_config.get('floor_color', None)
        if floor_color is not None or hasattr(self, 'floor_tiles'):
            frame = self._apply_floor_colors(frame, floor_color, apply_robot_glow=False) 
        
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
            # Find src/ directory path (go up from utils/map_manager to find src/)
            current_file = os.path.abspath(__file__)
            src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # src/ directory
            
            # Check use_robot_emoji from room_config
            use_robot_emoji = self.room_config.get('use_robot_emoji', False)
            robot_emoji_char = '🤖' if use_robot_emoji else None
            robot_emoji_color = self.room_config.get('robot_emoji_color', 'yellow')
            use_robot_emoji_color = self.room_config.get('use_robot_emoji_color', False)
            
            pil_frame = Image.fromarray(frame.astype(np.uint8)).convert('RGBA')
            
            cell = self.grid.get(agent_x, agent_y)
            try:
                # Grid.render_tile call: signature (obj, agent_dir=None, highlight=False, tile_size=32, subdivs=3)
                # Note: Linter shows error but execution works fine (execution tested)
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
            
            # If robot emoji mode
            if use_robot_emoji and robot_emoji_char:
                font_size = int(actual_tile_size * 0.8)
                
                # When use_robot_emoji_color=True, use imagetext_py for color emoji rendering
                if use_robot_emoji_color:
                    # Load font from src/asset/fonts/
                    font_path = os.path.join(src_dir, 'asset', 'fonts', 'NotoEmoji-Regular.ttf')
                    if os.path.exists(font_path):
                        FontDB.LoadFromPath("NotoEmoji", font_path)
                        font = FontDB.Query("NotoEmoji")
                        
                        # Render color emoji using imagetext_py Writer
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
                                draw_emojis=True  # Enable color emoji rendering
                            )
                
                # Use PIL (monochrome) when use_robot_emoji_color=False or imagetext_py usage failed
                if not use_robot_emoji_color:
                    font = None
                    try:
                        local_font_path = os.path.join(src_dir, 'asset', 'fonts', 'NotoEmoji-Regular.ttf')
                        if os.path.exists(local_font_path):
                            font = ImageFont.truetype(local_font_path, font_size)
                    except Exception:
                        font = None
                    
                    draw = ImageDraw.Draw(bg_tile)
                    
                    # Calculate emoji text size
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
                    
                    # Draw emoji at center
                    x = (actual_tile_size - text_width) // 2
                    y = (actual_tile_size - text_height) // 2 - 2
                    
                    # Use color map for robot emoji
                    fill_color = COLOR_MAP_RGBA.get(robot_emoji_color, (255, 255, 255, 255))
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
                # arrow.png image mode (default)
                arrow_img_path = os.path.join(src_dir, 'asset', 'arrow.png')
                
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
        
        # Make floor at robot position glow (after robot rendering)
        if hasattr(self, 'agent_pos'):
            frame = self._apply_robot_floor_glow(frame)
        
        # Apply highlight to cells within agent's field of view
        frame = self._apply_highlight(frame)
        
        # Add chess-like coordinate labels (bottom-left origin)
        frame = self._add_coordinate_labels(frame)
        
        return frame
    
    def _add_coordinate_labels(self, frame: np.ndarray) -> np.ndarray:
        """Add chess-like coordinate labels to the frame (bottom-left origin)
        
        X-axis: A, B, C, ... (alphabet, uppercase) at bottom
        Y-axis: 1, 2, 3, ... (numbers) at left
        """
        try:
            # Get grid dimensions
            if not hasattr(self, 'grid') or self.grid is None:
                return frame
            
            width = self.grid.width
            height = self.grid.height
            actual_tile_size = self.tile_size if hasattr(self, 'tile_size') else 32
            
            # Convert to PIL Image for drawing
            pil_frame = Image.fromarray(frame.astype(np.uint8)).convert('RGBA')
            draw = ImageDraw.Draw(pil_frame)
            
            # Font size
            font_size = max(14, actual_tile_size // 2)
            
            # Load NotoSans-Bold font
            font = None
            import os
            current_file = os.path.abspath(__file__)
            src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            
            font_paths = [
                os.path.join(src_dir, 'asset', 'fonts', 'NotoSans-Bold.ttf'),
                os.path.join(src_dir, 'asset', 'fonts', 'NotoSans-Regular.ttf'),
            ]
            
            for font_path in font_paths:
                try:
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, font_size)
                        break
                except Exception:
                    continue
            
            if font is None:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
            
            # X-axis labels (A, B, C, ...) - bottom row
            # MiniGrid: y=0 is top, y=height-1 is bottom (wall)
            # Labels at bottom row: y=height-2 (since height-1 is wall)
            label_y_bottom = height - 1
            
            # Label range: x=1 to width-2 (skip walls)
            for x in range(1, width - 1):  # x=1 to width-2
                # Convert x to uppercase alphabet (1->A, 2->B, 3->C, ...)
                label_text = chr(ord('A') + (x - 1))
                
                # Position: center of tile at bottom row
                x_pos = x * actual_tile_size + actual_tile_size // 2
                y_pos = label_y_bottom * actual_tile_size + actual_tile_size - font_size - 4
                
                # Get text bounding box for centering
                if font:
                    try:
                        bbox = draw.textbbox((0, 0), label_text, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                    except AttributeError:
                        try:
                            text_width, text_height = draw.textsize(label_text, font=font)
                        except:
                            text_width = font_size
                            text_height = font_size
                    except:
                        text_width = font_size
                        text_height = font_size
                else:
                    text_width = font_size
                    text_height = font_size
                
                # Draw text centered
                text_x = x_pos - text_width // 2
                text_y = y_pos - text_height // 2
                
                if font:
                    try:
                        draw.text((text_x, text_y), label_text, font=font, fill=(0, 0, 0, 255))
                    except:
                        draw.text((text_x, text_y), label_text, fill=(0, 0, 0, 255))
                else:
                    draw.text((text_x, text_y), label_text, fill=(0, 0, 0, 255))
            
            # Y-axis labels (1, 2, 3, ...) - left column
            # MiniGrid: x=0 is left (wall), so we use x=1
            # Y-axis: bottom is 1, going up (2, 3, ...)
            # MiniGrid: y=height-1 is bottom (wall), y=height-2 -> 1, y=height-3 -> 2, ...
            
            for y in range(height - 2):  # y=0 to height-3
                # Convert y to number (bottom->1, going up)
                # MiniGrid y: height-2 -> 1, height-3 -> 2, ..., 0 -> height-1
                label_number = height - 2 - y
                label_text = str(label_number)
                
                # Position: left side of tile (x=1, since x=0 is wall)
                x_pos = 0.25 * actual_tile_size  # Small offset from left edge
                y_pos = y * actual_tile_size + actual_tile_size * 1.25
                
                # Get text bounding box for centering
                if font:
                    try:
                        bbox = draw.textbbox((0, 0), label_text, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                    except AttributeError:
                        try:
                            text_width, text_height = draw.textsize(label_text, font=font)
                        except:
                            text_width = font_size
                            text_height = font_size
                    except:
                        text_width = font_size
                        text_height = font_size
                else:
                    text_width = font_size
                    text_height = font_size
                
                # Draw text centered vertically
                text_x = x_pos
                text_y = y_pos - text_height // 2
                
                if font:
                    try:
                        draw.text((text_x, text_y), label_text, font=font, fill=(0, 0, 0, 255))
                    except:
                        draw.text((text_x, text_y), label_text, fill=(0, 0, 0, 255))
                else:
                    draw.text((text_x, text_y), label_text, fill=(0, 0, 0, 255))
            
            # Convert back to numpy array
            frame = np.array(pil_frame.convert('RGB'))
            
        except Exception as e:
            # If anything fails, return original frame
            pass
        
        return frame
    
    def _apply_floor_colors(self, frame: np.ndarray, floor_color: Optional[str] = None, apply_robot_glow: bool = False) -> np.ndarray:
        """
        Apply floor color changes (called before robot rendering)
        
        Args:
            frame: Original frame
            floor_color: Overall floor color (if None, keep default color)
                        Floor color at specific positions is taken from floor_tiles
            apply_robot_glow: Whether to apply green border at robot position (default: False)
        
        Returns:
            Modified frame
        """
        if not hasattr(self, 'grid'):
            return frame
        
        agent_x, agent_y = None, None
        if apply_robot_glow and hasattr(self, 'agent_pos'):
            agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        
        actual_tile_size = self.tile_size if hasattr(self, 'tile_size') else 32
        
        # Default floor color (grey)
        default_floor_color = (128, 128, 128)
        global_floor_color = COLOR_MAP_RGB.get(floor_color, default_floor_color) if floor_color else None
        
        pil_frame = Image.fromarray(frame.astype(np.uint8)).convert('RGBA')
        draw = ImageDraw.Draw(pil_frame)
        
        # Iterate through all tiles
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                cell = self.grid.get(x, y)
                
                # Empty tile (floor) or passable emoji object
                is_floor = (cell is None) or (
                    hasattr(cell, 'type') and 
                    cell.type == 'emoji' and 
                    hasattr(cell, 'can_overlap') and 
                    cell.can_overlap
                )
                
                if is_floor:
                    tile_x = x * actual_tile_size
                    tile_y = y * actual_tile_size
                    tile_end_x = tile_x + actual_tile_size
                    tile_end_y = tile_y + actual_tile_size
                    
                    # Don't apply floor color to robot position (only add green border after robot is drawn)
                    if apply_robot_glow and agent_x is not None and agent_y is not None and x == agent_x and y == agent_y:
                        continue
                    
                    # Floor color for specific position (higher priority)
                    tile_floor_color = None
                    if hasattr(self, 'floor_tiles') and (x, y) in self.floor_tiles:
                        tile_floor_color = COLOR_MAP_RGB.get(self.floor_tiles[(x, y)], default_floor_color)
                    # Global floor color (only applied when no specific position)
                    elif global_floor_color is not None:
                        tile_floor_color = global_floor_color
                    
                    # Apply floor color
                    if tile_floor_color is not None:
                        draw.rectangle(
                            [(tile_x, tile_y), (tile_end_x - 1, tile_end_y - 1)],
                            fill=tile_floor_color + (255,)
                        )
        
        return np.array(pil_frame.convert('RGB'))
    
    def _apply_background_color(self, frame: np.ndarray, background_color: str) -> np.ndarray:
        """
        Apply custom background color to the frame (replaces black background)
        
        Args:
            frame: Original frame with black background
            background_color: Color name (e.g., 'white', 'grey', 'blue', etc.)
        
        Returns:
            Modified frame with custom background color
        """
        bg_color = COLOR_MAP_RGB.get(background_color, (0, 0, 0))  # Default to black if not found
        
        # Convert frame to numpy array if needed
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        
        # Create a mask for black pixels (background)
        # Consider pixels as black if all RGB values are below a threshold
        black_threshold = 30
        black_mask = (
            (frame[:, :, 0] < black_threshold) &
            (frame[:, :, 1] < black_threshold) &
            (frame[:, :, 2] < black_threshold)
        )
        
        # Apply background color to black pixels
        frame[black_mask] = bg_color
        
        return frame
    
    def _apply_highlight(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply highlight color to cells within agent's field of view
        
        Args:
            frame: Frame to apply highlight to
        
        Returns:
            Modified frame with highlight applied
        """
        if not self.highlight:
            return frame
        
        if not hasattr(self, 'agent_pos') or not hasattr(self, 'agent_dir'):
            return frame
        
        # Get highlight color from room_config
        highlight_color_name = self.room_config.get('highlight_color', 'yellow')
        highlight_color = COLOR_MAP_RGB.get(highlight_color_name, (200, 200, 100))  # Default to yellow
        
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        agent_dir = self.agent_dir
        actual_tile_size = self.tile_size if hasattr(self, 'tile_size') else 32
        
        # Calculate view range from agent_view_size
        # agent_view_size is typically 7 (default MiniGrid view size)
        view_range = self.agent_view_size // 2  # Forward range
        view_width = self.agent_view_size  # Total width
        
        pil_frame = Image.fromarray(frame.astype(np.uint8)).convert('RGBA')
        
        # Apply highlight to cells within view range
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                # Calculate relative position from agent
                dx = x - agent_x
                dy = y - agent_y
                
                # Transform to agent's coordinate system
                if agent_dir == 0:  # Right
                    rel_x, rel_y = dx, -dy
                elif agent_dir == 1:  # Down
                    rel_x, rel_y = dy, dx
                elif agent_dir == 2:  # Left
                    rel_x, rel_y = -dx, dy
                else:  # Up
                    rel_x, rel_y = -dy, -dx
                
                # Check if cell is within view range
                in_view = (
                    rel_x >= 0 and
                    rel_x <= view_range and
                    abs(rel_y) <= view_width // 2
                )
                
                if in_view:
                    tile_x = x * actual_tile_size
                    tile_y = y * actual_tile_size
                    
                    # Apply highlight as a semi-transparent overlay
                    highlight_overlay = Image.new('RGBA', (actual_tile_size, actual_tile_size), highlight_color + (64,))  # 25% opacity
                    pil_frame.paste(highlight_overlay, (tile_x, tile_y), highlight_overlay)
        
        return np.array(pil_frame.convert('RGB'))
    
    def _apply_robot_floor_glow(self, frame: np.ndarray) -> np.ndarray:
        """
        Add green border to floor at robot position (called after robot rendering)
        Applies glow when robot is on any non-empty cell (not just emoji objects)
        
        Args:
            frame: Frame with robot rendered
        
        Returns:
            Modified frame
        """
        if not hasattr(self, 'agent_pos') or not hasattr(self, 'grid'):
            return frame
        
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        
        # Check if robot is on a non-empty cell (any object, not just empty floor)
        cell = self.grid.get(agent_x, agent_y)
        is_on_object = (cell is not None)
        
        # Only draw green border if robot is on a non-empty cell
        if not is_on_object:
            return frame
        
        actual_tile_size = self.tile_size if hasattr(self, 'tile_size') else 32
        
        pil_frame = Image.fromarray(frame.astype(np.uint8)).convert('RGBA')
        draw = ImageDraw.Draw(pil_frame)
        
        tile_x = agent_x * actual_tile_size
        tile_y = agent_y * actual_tile_size
        tile_end_x = tile_x + actual_tile_size
        tile_end_y = tile_y + actual_tile_size
        
        # Draw green border on floor at robot position (glow like emoji)
        border_width = 3
        green_color = (0, 255, 0, 255)  # Green border
        draw.rectangle(
            [(tile_x, tile_y), (tile_end_x - 1, tile_end_y - 1)],
            outline=green_color,
            width=border_width
        )
        
        return np.array(pil_frame.convert('RGB'))


class MiniGridEmojiWrapper:
    """
    Wrapper class for controlling MiniGrid environment (supports emoji and absolute coordinate movement)
    
    Main features:
    - Emoji object placement and parsing
    - Absolute direction movement (direct up/down/left/right movement)
    - Environment creation, manipulation, and information parsing API
    """
    
    # Default actions (relative direction)
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
    
    # Absolute direction actions
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
        """Initialize MiniGrid environment wrapper with emoji support.
        
        Creates a MiniGrid environment wrapper that supports emoji objects and
        absolute direction movement. The wrapper provides a high-level interface
        for environment control, making it easy to use with VLM controllers.
        
        Args:
            size: Grid size (width and height). Defaults to 10.
            walls: List of wall positions as (x, y) tuples. If None, no walls
                are added. Defaults to None.
            room_config: Dictionary containing room configuration. Can include:
                - 'start_pos': (x, y) tuple for agent start position
                - 'goal_pos': (x, y) tuple for goal position
                - 'walls': List of (x, y, color) tuples for walls
                - 'objects': List of object dictionaries (emoji objects, etc.)
                - 'floor_tiles': Dict mapping (x, y) to color strings
                - 'use_robot_emoji': bool for robot emoji rendering
                - 'robot_emoji_color': str for robot emoji color
                If None, empty config is used. Defaults to None.
            render_mode: Rendering mode. Must be 'rgb_array' for image output.
                Defaults to 'rgb_array'.
            use_absolute_movement: Whether to enable absolute movement mode.
                - True: step() method receives and processes absolute direction
                    actions (up/down/left/right). This is the standard mode.
                - False: step() method receives and processes relative direction
                    actions (turn left/right, move forward/backward). Legacy mode.
                Defaults to True.
            **kwargs: Additional keyword arguments passed to CustomRoomEnv.
        
        Examples:
            >>> # Create a simple 10x10 environment
            >>> wrapper = MiniGridEmojiWrapper(size=10)
            >>> obs, info = wrapper.reset()
            >>> 
            >>> # Create environment with walls
            >>> walls = [(5, 5), (5, 6), (6, 5)]
            >>> wrapper = MiniGridEmojiWrapper(size=10, walls=walls)
            >>> 
            >>> # Create environment from room_config
            >>> room_config = {
            ...     'start_pos': (1, 1),
            ...     'goal_pos': (8, 8),
            ...     'objects': [
            ...         {'type': 'emoji', 'pos': (3, 3), 'emoji_name': 'tree'}
            ...     ]
            ... }
            >>> wrapper = MiniGridEmojiWrapper(size=10, room_config=room_config)
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
        """Reset the environment to initial state.
        
        Resets the environment and returns the initial observation and info.
        This should be called at the start of each episode.
        
        Args:
            seed: Random seed for environment reset. If None, uses random seed.
                Defaults to None.
        
        Returns:
            Tuple containing:
                - observation: numpy array of shape (height, width, 3) representing
                    the initial state of the environment as an RGB image.
                - info: Dictionary containing additional information about the
                    initial state (e.g., agent position, mission text).
        
        Examples:
            >>> wrapper = MiniGridEmojiWrapper(size=10)
            >>> obs, info = wrapper.reset()
            >>> print(f"Agent position: {info.get('agent_pos')}")
        """
        self.current_obs, self.current_info = self.env.reset(seed=seed)
        return self.current_obs, self.current_info
    
    def step(self, action: Union[int, str, Dict]) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute an action in the environment.
        
        Executes the given action and returns the new observation, reward,
        termination status, truncation status, and info dictionary.
        
        The behavior depends on the `use_absolute_movement` setting:
        - If True (default): Action is interpreted as an absolute direction
          (up/down/left/right). The robot will automatically rotate to face
          the correct direction before moving.
        - If False: Action is interpreted as a relative direction
          (turn left/right, move forward/backward).
        
        Args:
            action: Action to execute. Can be:
                - Integer: Action index (0-6 for absolute, 0-6 for relative)
                - String: Action name (e.g., "move up", "turn left", "pickup", "pickup:north")
                - Dict: {"pickup": "north/south/west/east"} for directional pickup
                For absolute movement: "move up", "move down", "move left",
                "move right", "pickup", "pickup:north/south/west/east", "drop", "toggle"
                For relative movement: "turn left", "turn right", "move forward",
                "move backward", "pickup", "drop", "toggle"
        
        Returns:
            Tuple containing:
                - observation: Dictionary or numpy array representing the new
                    state of the environment.
                - reward: float representing the reward for this step.
                - terminated: bool indicating if the episode has ended due to
                    reaching a terminal state (e.g., goal reached).
                - truncated: bool indicating if the episode was truncated
                    (e.g., max steps reached).
                - info: Dictionary containing additional information about
                    the step (e.g., agent position, mission status).
        
        Examples:
            >>> wrapper = MiniGridEmojiWrapper(size=10, use_absolute_movement=True)
            >>> obs, info = wrapper.reset()
            >>> 
            >>> # Move up (absolute direction)
            >>> obs, reward, done, truncated, info = wrapper.step("move up")
            >>> 
            >>> # Or use integer index
            >>> obs, reward, done, truncated, info = wrapper.step(0)  # move up
            >>> 
            >>> # Pickup with direction
            >>> obs, reward, done, truncated, info = wrapper.step("pickup:north")
            >>> obs, reward, done, truncated, info = wrapper.step({"pickup": "north"})
            >>> 
            >>> # With relative movement mode
            >>> wrapper_rel = MiniGridEmojiWrapper(size=10, use_absolute_movement=False)
            >>> obs, reward, done, truncated, info = wrapper_rel.step("move forward")
        """
        if self.use_absolute_movement:
            # Absolute movement mode: use step_absolute
            return self.step_absolute(action)
        else:
            # Relative movement mode: use existing method
            if isinstance(action, str):
                action = self.parse_action(action)
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.current_obs = obs
            self.current_info = info
            return obs, reward, terminated, truncated, info
    
    def _get_target_direction(self, absolute_action: int) -> int:
        """Convert absolute action to MiniGrid direction"""
        direction_map = {0: 3, 1: 1, 2: 2, 3: 0}  # up->North, down->South, left->West, right->East
        return direction_map.get(absolute_action, 0)
    
    def _calculate_rotation(self, current_dir: int, target_dir: int) -> list:
        """Calculate action sequence to rotate from current direction to target direction"""
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
    
    def step_absolute(self, action: Union[int, str, Dict]) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute an absolute direction action.
        
        Moves the robot in an absolute direction (North/South/East/West)
        regardless of the robot's current heading. The robot will automatically
        rotate to face the correct direction before moving.
        
        This method is useful for VLM control where the model can specify
        directions like "move up" without needing to know the robot's current
        orientation.
        
        Args:
            action: Absolute direction action. Can be:
                - Integer index: 0-6
                    - 0: Move up (North)
                    - 1: Move down (South)
                    - 2: Move left (West)
                    - 3: Move right (East)
                    - 4: Pick up object
                    - 5: Drop object
                    - 6: Toggle/interact
                - String: Action name or alias
                    - "move up", "up", "north", "n": Move up
                    - "move down", "down", "south", "s": Move down
                    - "move left", "left", "west", "w": Move left
                    - "move right", "right", "east", "e": Move right
                    - "pickup", "pick up", "grab": Pick up object (front direction)
                    - "pickup:north/south/west/east": Pick up object in specified direction
                    - "drop", "put down", "release": Drop object
                    - "toggle", "interact", "use": Toggle/interact
                - Dict: {"pickup": "north/south/west/east"} for directional pickup
        
        Returns:
            Tuple containing:
                - observation: Dictionary or numpy array of the new state.
                - reward: float reward value.
                - terminated: bool indicating terminal state reached.
                - truncated: bool indicating episode truncation.
                - info: Dictionary with additional step information.
        
        Examples:
            >>> wrapper = MiniGridEmojiWrapper(size=10)
            >>> obs, info = wrapper.reset()
            >>> 
            >>> # Move north (up) regardless of current heading
            >>> obs, reward, done, truncated, info = wrapper.step_absolute("move up")
            >>> 
            >>> # Move east (right)
            >>> obs, reward, done, truncated, info = wrapper.step_absolute(3)
            >>> 
            >>> # Pick up an object
            >>> obs, reward, done, truncated, info = wrapper.step_absolute("pickup")
        
        Note:
            For movement actions (0-3), the robot will automatically rotate
            to face the target direction before moving. Non-movement actions
            (4-6) are executed directly without rotation.
        """
        if isinstance(action, str):
            action = self.parse_absolute_action(action)
        
        # Handle directional pickup: {"pickup": "north/south/west/east"}
        if isinstance(action, dict) and "pickup" in action:
            pickup_direction = action["pickup"].lower()
            # Map direction to MiniGrid direction (0=East, 1=South, 2=West, 3=North)
            direction_map = {
                "north": 3,
                "east": 0,
                "south": 1,
                "west": 2
            }
            target_dir = direction_map.get(pickup_direction)
            if target_dir is None:
                raise ValueError(f"Invalid pickup direction: {pickup_direction}. Must be north, south, west, or east.")
            
            # Rotate to face the target direction
            current_dir = self.env.agent_dir
            rotation_actions = self._calculate_rotation(current_dir, target_dir)
            
            # Execute rotation actions
            for rot_action in rotation_actions:
                obs, reward, terminated, truncated, info = self.env.step(rot_action)
                self.current_obs = obs
                self.current_info = info
                if terminated or truncated:
                    return obs, reward, terminated, truncated, info
            
            # After rotation, perform pickup (MiniGrid action 3)
            obs, reward, terminated, truncated, info = self.env.step(3)  # pickup in MiniGrid
            self.current_obs = obs
            self.current_info = info
            return obs, reward, terminated, truncated, info
        
        # For non-movement actions (pickup, drop, toggle), execute directly
        if isinstance(action, int) and action >= 4:
            # Convert action 4 (pickup in absolute) to action 3 (pickup in MiniGrid)
            # MiniGrid uses: 0=turn left, 1=turn right, 2=move forward, 3=pickup, 4=drop, 5=toggle
            minigrid_action = action
            if action == 4:  # pickup in our system
                minigrid_action = 3  # pickup in MiniGrid
            elif action == 5:  # drop in our system
                minigrid_action = 4  # drop in MiniGrid
            elif action == 6:  # toggle in our system
                minigrid_action = 5  # toggle in MiniGrid
            
            # Call env.step() directly in relative movement mode (prevent recursion)
            obs, reward, terminated, truncated, info = self.env.step(minigrid_action)
            self.current_obs = obs
            self.current_info = info
            return obs, reward, terminated, truncated, info
        
        # For movement actions: check current direction and perform necessary rotation
        current_dir = self.env.agent_dir
        target_dir = self._get_target_direction(action)
        
        rotation_actions = self._calculate_rotation(current_dir, target_dir)
        
        # Execute rotation actions (call env.step() directly in relative movement mode)
        for rot_action in rotation_actions:
            obs, reward, terminated, truncated, info = self.env.step(rot_action)
            self.current_obs = obs
            self.current_info = info
            if terminated or truncated:
                return obs, reward, terminated, truncated, info
        
        # After rotation to target direction, move forward (call env.step() directly in relative movement mode)
        obs, reward, terminated, truncated, info = self.env.step(2)  # move forward
        self.current_obs = obs
        self.current_info = info
        return obs, reward, terminated, truncated, info
    
    def get_image(self, fov_range: Optional[int] = None, fov_width: Optional[int] = None, 
                  fov_enabled: Optional[bool] = None, fov_color: Optional[str] = None) -> np.ndarray:
        """Get the current environment state as an RGB image.
        
        Returns the current state of the environment as a numpy array image,
        suitable for input to Vision Language Models (VLMs). Optionally applies
        field-of-view (FOV) limitations to simulate limited visibility.
        
        Args:
            fov_range: Maximum forward range for field of view. If None, uses
                value from room_config or returns full visibility image. Defaults to None.
            fov_width: Maximum width (left/right) for field of view. If None, uses
                value from room_config or returns full visibility image. Defaults to None.
            fov_enabled: Whether to enable FOV masking. If None, uses value from
                room_config or defaults to False. Defaults to None.
            fov_color: Color name for FOV masking (e.g., 'black', 'grey', 'blue').
                If None, uses value from room_config or defaults to 'black'. Defaults to None.
        
        Returns:
            numpy.ndarray: RGB image of shape (height, width, 3) with dtype uint8.
                Height and width are typically size * 32 pixels.
                If FOV is enabled, areas outside the field of view are masked with
                the specified color.
        
        Examples:
            >>> wrapper = MiniGridEmojiWrapper(size=10)
            >>> obs, info = wrapper.reset()
            >>> 
            >>> # Get full image
            >>> image = wrapper.get_image()
            >>> print(f"Image shape: {image.shape}")  # (320, 320, 3)
            >>> 
            >>> # Get image with limited field of view
            >>> image_fov = wrapper.get_image(fov_range=3, fov_width=2, fov_enabled=True)
            >>> # Areas outside FOV are masked
            
            >>> # Get image with custom FOV color
            >>> image_fov = wrapper.get_image(fov_range=3, fov_width=2, 
            ...                                fov_enabled=True, fov_color='grey')
        
        Note:
            This method is typically used to get images for VLM input. The image
            shows the current state of the environment including walls, objects,
            the robot, and goal markers.
        """
        image = self.env.render()
        if image is None:
            return np.zeros((self.size * 32, self.size * 32, 3), dtype=np.uint8)
        
        # Get FOV settings from room_config if not provided
        room_config = getattr(self.env, 'room_config', {})
        if fov_enabled is None:
            fov_enabled = room_config.get('fov_enabled', False)
        if fov_color is None:
            fov_color = room_config.get('fov_color', 'black')
        if fov_range is None:
            fov_range = room_config.get('fov_range', None)
        if fov_width is None:
            fov_width = room_config.get('fov_width', None)
        
        if fov_enabled and fov_range is not None and fov_width is not None:
            image = self._apply_fog_of_war(image, fov_range, fov_width, fov_color)
        
        return image
    
    def _apply_fog_of_war(self, image: np.ndarray, fov_range: int, fov_width: int, 
                          fov_color: str = 'black') -> np.ndarray:
        """Apply field of view limitation, masking areas outside view with specified color
        
        Args:
            image: Original image to apply FOV masking to
            fov_range: Maximum forward range for field of view
            fov_width: Maximum width (left/right) for field of view
            fov_color: Color name for masking (e.g., 'black', 'grey', 'blue', etc.)
        
        Returns:
            Masked image with areas outside FOV filled with specified color
        """
        if not hasattr(self.env, 'agent_pos') or not hasattr(self.env, 'agent_dir'):
            return image
        
        mask_color = COLOR_MAP_RGB.get(fov_color, (0, 0, 0))  # Default to black if not found
        
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
                    masked_image[pixel_y:end_y, pixel_x:end_x] = mask_color
        
        return masked_image
    
    def get_action_space(self) -> Dict:
        """Return action space information"""
        return {
            'n': self.env.action_space.n,
            'actions': list(self.ACTION_NAMES.values()),
            'action_mapping': self.ACTION_NAMES,
            'action_aliases': self.ACTION_ALIASES
        }
    
    def get_absolute_action_space(self) -> Dict:
        """Get information about the absolute direction action space.
        
        Returns a dictionary containing details about the available absolute
        direction actions, including action names, aliases, and mappings.
        This is useful for understanding what actions are available and for
        displaying action options to users or VLMs.
        
        Returns:
            Dictionary containing:
                - 'n': Integer, number of actions (7)
                - 'actions': List of action name strings:
                    ["move up", "move down", "move left", "move right",
                     "pickup", "drop", "toggle"]
                - 'action_mapping': Dictionary mapping index to action name:
                    {0: "move up", 1: "move down", ...}
                - 'action_aliases': Dictionary mapping aliases to indices:
                    {"up": 0, "down": 1, "north": 0, "south": 1, ...}
        
        Examples:
            >>> wrapper = MiniGridEmojiWrapper(size=10)
            >>> action_space = wrapper.get_absolute_action_space()
            >>> 
            >>> print(f"Number of actions: {action_space['n']}")
            >>> print(f"Available actions: {action_space['actions']}")
            >>> 
            >>> # Check if an alias exists
            >>> if "up" in action_space['action_aliases']:
            ...     action_idx = action_space['action_aliases']["up"]
            ...     print(f"Action index for 'up': {action_idx}")
        """
        return {
            'n': 7,
            'actions': list(self.ABSOLUTE_ACTION_NAMES.values()),
            'action_mapping': self.ABSOLUTE_ACTION_NAMES,
            'action_aliases': self.ABSOLUTE_ACTION_ALIASES
        }
    
    def parse_action(self, action_str: str) -> int:
        """Convert action string to index (relative direction)"""
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
    
    def parse_absolute_action(self, action_str: str) -> Union[int, Dict[str, str]]:
        """Parse an absolute direction action string to its integer index or dict.
        
        Converts a string representation of an absolute direction action
        (e.g., "move up", "north", "up", "pickup:north") to its corresponding 
        integer index (0-6) or dict format for directional pickup.
        This is useful when receiving action commands from VLMs or
        other text-based sources.
        
        Args:
            action_str: String representation of the action. Can be:
                - Integer string: "0", "1", "2", etc. (0-6)
                - Action name: "move up", "move down", "move left", "move right"
                - Alias: "up", "down", "left", "right", "north", "south", etc.
                - Other actions: "pickup", "drop", "toggle"
                - Directional pickup: "pickup:north", "pickup:south", "pickup:west", "pickup:east"
                Case-insensitive.
        
        Returns:
            Integer action index (0-6) for regular actions, or dict {"pickup": "north/south/west/east"} for directional pickup:
                - 0: move up (North)
                - 1: move down (South)
                - 2: move left (West)
                - 3: move right (East)
                - 4: pickup (without direction, for backward compatibility)
                - 5: drop
                - 6: toggle
                - {"pickup": "north/south/west/east"}: pickup with direction
        
        Raises:
            ValueError: If the action string is not recognized.
        
        Examples:
            >>> wrapper = MiniGridEmojiWrapper(size=10)
            >>> 
            >>> # Parse various action strings
            >>> wrapper.parse_absolute_action("move up")  # 0
            >>> wrapper.parse_absolute_action("north")   # 0
            >>> wrapper.parse_absolute_action("up")       # 0
            >>> wrapper.parse_absolute_action("n")        # 0
            >>> wrapper.parse_absolute_action("move right")  # 3
            >>> wrapper.parse_absolute_action("east")     # 3
            >>> wrapper.parse_absolute_action("pickup")   # 4
            >>> wrapper.parse_absolute_action("pickup:north")  # {"pickup": "north"}
            >>> wrapper.parse_absolute_action("0")        # 0
        """
        action_str = action_str.strip()
        
        # Handle dict format: {"pickup": "north"} or {"pickup": "south"} etc.
        if isinstance(action_str, dict):
            if "pickup" in action_str:
                direction = action_str["pickup"].lower().strip()
                # Only accept north/south/west/east
                if direction in ["north", "south", "west", "east"]:
                    return {"pickup": direction}
                else:
                    raise ValueError(
                        f"Invalid direction for pickup: '{direction}'. "
                        f"Must be one of: north, south, west, east"
                    )
            return action_str
        
        # Handle string format: "pickup:north", "pickup:south", etc.
        if ":" in action_str:
            parts = action_str.split(":", 1)
            if len(parts) == 2:
                action_part = parts[0].strip().lower()
                direction_part = parts[1].strip().lower()
                
                if action_part in ["pickup", "pick up", "pick_up", "grab"]:
                    # Only accept north/south/west/east
                    if direction_part in ["north", "south", "west", "east"]:
                        return {"pickup": direction_part}
                    else:
                        raise ValueError(
                            f"Invalid direction for pickup: '{direction_part}'. "
                            f"Must be one of: north, south, west, east"
                        )
        
        # Handle regular integer strings
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
            f"Available actions: {list(self.ABSOLUTE_ACTION_ALIASES.keys())} or numbers 0-6, or pickup:north/south/west/east"
        )
    
    def get_state(self) -> Dict:
        """Get the current environment state information.
        
        Returns a dictionary containing the current state of the environment,
        including agent position, direction, mission text, and current image.
        
        Returns:
            Dictionary containing:
                - 'agent_pos': Tuple (x, y) of agent's current position.
                - 'agent_dir': Integer (0-3) representing agent's heading direction:
                    - 0: East (right)
                    - 1: South (down)
                    - 2: West (left)
                    - 3: North (up)
                - 'mission': String containing the current mission/objective text.
                - 'image': numpy.ndarray of shape (H, W, 3) representing the
                    current environment state as an RGB image.
        
        Examples:
            >>> wrapper = MiniGridEmojiWrapper(size=10)
            >>> obs, info = wrapper.reset()
            >>> 
            >>> # Get current state
            >>> state = wrapper.get_state()
            >>> print(f"Agent at: {state['agent_pos']}")
            >>> print(f"Facing: {state['agent_dir']}")
            >>> print(f"Mission: {state['mission']}")
            >>> 
            >>> # Use state for VLM input
            >>> image = state['image']
        """
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
        """Parse grid and return object information for each position"""
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
        """Return emoji name at specific position"""
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
        Return robot's heading direction as string
        
        Returns:
            heading: Direction string
                - "East" (right, agent_dir=0)
                - "South" (down, agent_dir=1)
                - "West" (left, agent_dir=2)
                - "North" (up, agent_dir=3)
        """
        if not hasattr(self.env, 'agent_dir'):
            return "Unknown"
        
        agent_dir = self.env.agent_dir
        heading_map = {
            0: "East",   # right
            1: "South",  # down
            2: "West",   # left
            3: "North"   # up
        }
        return heading_map.get(agent_dir, "Unknown")
    
    def get_heading_description(self) -> str:
        """
        Return detailed description string of robot's heading direction
        
        Returns:
            description: Direction description string
                e.g., "facing East (right)" or "facing North (up)"
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
        """Close environment and clean up resources"""
        self.env.close()

