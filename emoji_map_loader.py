"""
Emoji Map JSON Loader and Converter Module

Reads emoji maps from JSON files and converts them to minigrid environments.

JSON Structure:
{
  "map": {
    "emoji_render": "⬛⬛⬛⬛⬛...\n⬛⬜️⬜️⬜️...\n..." 
    or
    "emoji_render": [
      "⬛⬛⬛⬛⬛...",
      "⬛⬜️⬜️⬜️...",
      ...
    ]
    or
    "emoji_render": [
      ["⬛", "⬛", "⬛", ...],
      ["⬛", "⬜️", "⬜️", ...],
      ...
    ],
    "emoji_objects": {
      "⬛": {
        "type": "wall",
        "color": "grey",
        "can_pickup": false,
        "can_overlap": false
      },
      "🟩": {
        "type": "emoji",
        "emoji_name": "brick",
        "color": "green",
        "can_pickup": false,
        "can_overlap": false,
        "use_emoji_color": true
      },
      ...
    },
    "robot_config": {
      "use_robot_emoji": true,
      "robot_emoji_color": "red",
      "use_robot_emoji_color": true
    },
    "start_pos": [1, 1],
    "goal_pos": [12, 1]
  }
}
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from minigrid_customenv_emoji import MiniGridEmojiWrapper

# Default emoji object definitions for text file support
DEFAULT_EMOJI_OBJECTS = {
    "⬛": {
        "type": "wall",
        "color": "grey",
        "can_pickup": False,
        "can_overlap": False
    },
    "⬜️": {
        "type": "empty",
        "can_pickup": False,
        "can_overlap": True
    },
    "🟦": {
        "type": "floor",
        "color": "blue"
    },
    "🟪": {
        "type": "floor",
        "color": "purple"
    },
    "🟨": {
        "type": "floor",
        "color": "yellow"
    },
    "🟩": {
        "type": "floor",
        "color": "green"
    },
    "🤖": {
        "type": "empty",
        "can_pickup": False,
        "can_overlap": True
    },
    "🎯": {
        "type": "goal",
        "can_pickup": False,
        "can_overlap": False
    }
}


class EmojiMapLoader:
    """Emoji map JSON loader and converter class"""
    
    def __init__(self, json_path: str):
        """
        Args:
            json_path: Path to JSON file
        """
        self.json_path = Path(json_path)
        self.map_data = None
        self.emoji_render = None
        self.emoji_objects = None
        self.robot_config = None
        self.start_pos = None
        self.goal_pos = None
        self.size = None
        self.num_rows = None
        self.num_cols = None
        
        self._load_json()
        self._parse_map_data()
    
    def _load_json(self):
        """Load JSON file"""
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'map' not in data:
            raise ValueError("JSON file does not contain 'map' key.")
        
        self.map_data = data['map']
    
    def _parse_emoji_text(self, text: str) -> List[List[str]]:
        """
        Parse text-based emoji map into 2D array
        
        Args:
            text: Emoji map text separated by newlines
        
        Returns:
            2D array (list of row lists)
        """
        lines = text.strip().split('\n')
        # Remove empty lines
        lines = [line.strip() for line in lines if line.strip()]
        
        if len(lines) == 0:
            raise ValueError("Emoji map text is empty.")
        
        result = []
        for line in lines:
            emojis = []
            i = 0
            while i < len(line):
                char = line[i]
                # Variation Selector (U+FE0F) or Zero Width Joiner (U+200D) should be grouped with previous character
                if ord(char) in [0xFE0F, 0x200D]:
                    # Append to previous emoji (if already added)
                    if emojis:
                        emojis[-1] += char
                    i += 1
                    continue
                
                # Check if next character is Variation Selector or Zero Width Joiner
                if i + 1 < len(line):
                    next_char = line[i + 1]
                    if ord(next_char) in [0xFE0F, 0x200D]:
                        # Group with Variation Selector or Zero Width Joiner
                        if i + 2 < len(line) and ord(line[i + 2]) in [0xFE0F, 0x200D]:
                            # Two combining characters (rare but possible)
                            emojis.append(line[i:i+3])
                            i += 3
                        else:
                            emojis.append(line[i:i+2])
                            i += 2
                    else:
                        emojis.append(char)
                        i += 1
                else:
                    emojis.append(char)
                    i += 1
            
            # Remove empty strings or whitespace-only items
            emojis = [e for e in emojis if e.strip()]
            if emojis:
                result.append(emojis)
        
        return result
    
    def _parse_map_data(self):
        """Parse map data"""
        # Emoji render map
        if 'emoji_render' not in self.map_data:
            raise ValueError("JSON file does not contain 'emoji_render' key.")
        
        emoji_render_raw = self.map_data['emoji_render']
        
        # Check if text format (string or string array)
        if isinstance(emoji_render_raw, str):
            # Single string: separated by newlines
            self.emoji_render = self._parse_emoji_text(emoji_render_raw)
        elif isinstance(emoji_render_raw, list) and len(emoji_render_raw) > 0:
            # If first element is string, it's a text array format
            if isinstance(emoji_render_raw[0], str):
                # String array: each line is a string
                text = '\n'.join(emoji_render_raw)
                self.emoji_render = self._parse_emoji_text(text)
            else:
                # 2D array format (legacy format)
                self.emoji_render = emoji_render_raw
        else:
            raise ValueError("'emoji_render' must be a string, string array, or 2D array.")
        
        # Check map size
        if not isinstance(self.emoji_render, list) or len(self.emoji_render) == 0:
            raise ValueError("'emoji_render' parsing result is empty.")
        
        # Get row and column counts from emoji list
        self.num_rows = len(self.emoji_render)
        row_lengths = [len(row) for row in self.emoji_render]
        
        # All rows must have the same length (validate parsing result)
        if not all(length == row_lengths[0] for length in row_lengths):
            inconsistent_rows = [i for i, length in enumerate(row_lengths) 
                               if length != row_lengths[0]]
            raise ValueError(
                f"All rows in the map must have the same length. "
                f"Expected length: {row_lengths[0]}, "
                f"Problem row numbers: {inconsistent_rows} "
                f"(row lengths: {[row_lengths[i] for i in inconsistent_rows]})"
            )
        
        self.num_cols = row_lengths[0]
        
        # Error if row and column counts differ (square map required)
        if self.num_rows != self.num_cols:
            raise ValueError(
                f"Map must be square. "
                f"Row count: {self.num_rows}, Column count: {self.num_cols} "
                f"(Row and column counts do not match.)"
            )
        
        self.size = self.num_rows  # Square map: row count = column count = size
        
        # Emoji object definitions (merge defaults with JSON settings, JSON takes precedence)
        json_emoji_objects = self.map_data.get('emoji_objects', {})
        # Copy default emoji definitions and override with JSON settings
        self.emoji_objects = DEFAULT_EMOJI_OBJECTS.copy()
        self.emoji_objects.update(json_emoji_objects)  # JSON settings safely override default settings
        
        # Robot configuration
        self.robot_config = self.map_data.get('robot_config', {
            'use_robot_emoji': True,
            'robot_emoji_color': 'red',
            'use_robot_emoji_color': True
        })
        
        # Find 🤖 marker and set start_pos (takes precedence over JSON start_pos)
        robot_marker_found = False
        for y, row in enumerate(self.emoji_render):
            for x, emoji in enumerate(row):
                if emoji == '🤖':
                    self.start_pos = (x, y)
                    robot_marker_found = True
                    # Replace 🤖 with ⬜️ (treat as empty space)
                    self.emoji_render[y][x] = '⬜️'
                    break
            if robot_marker_found:
                break
        
        # Find 🎯 marker and set goal_pos (takes precedence over JSON goal_pos)
        goal_marker_found = False
        for y, row in enumerate(self.emoji_render):
            for x, emoji in enumerate(row):
                if emoji == '🎯':
                    self.goal_pos = (x, y)
                    goal_marker_found = True
                    # Replace 🎯 with ⬜️ (treat as empty space)
                    self.emoji_render[y][x] = '⬜️'
                    break
            if goal_marker_found:
                break
        
        # If 🤖 marker not found, use JSON start_pos (or default)
        if not robot_marker_found:
            self.start_pos = tuple(self.map_data.get('start_pos', [1, 1]))
        else:
            # Warning if 🤖 marker found but JSON also has start_pos
            if 'start_pos' in self.map_data:
                print(f"Warning: Found 🤖 marker and set start_pos to ({self.start_pos[0]}, {self.start_pos[1]}). "
                      f"JSON start_pos is ignored.")
        
        # If 🎯 marker not found, use JSON goal_pos (or default)
        if not goal_marker_found:
            self.goal_pos = tuple(self.map_data.get('goal_pos', [self.size - 2, self.size - 2]))
        else:
            # Warning if 🎯 marker found but JSON also has goal_pos
            if 'goal_pos' in self.map_data:
                print(f"Warning: Found 🎯 marker and set goal_pos to ({self.goal_pos[0]}, {self.goal_pos[1]}). "
                      f"JSON goal_pos is ignored.")
        
        # Clear error messages: if start or goal position is not set
        if self.start_pos is None:
            raise ValueError(
                "Robot start position not found. "
                "Place 🤖 emoji on the map or specify 'start_pos' in JSON."
            )
        
        if self.goal_pos is None:
            raise ValueError(
                "Goal position not found. "
                "Place 🎯 emoji on the map or specify 'goal_pos' in JSON."
            )
    
    def _parse_emoji_map(self) -> Tuple[List, List, Dict]:
        """
        이모지 맵을 파싱하여 walls, objects, floor_tiles 리스트 생성
        
        Returns:
            (walls, objects, floor_tiles): 벽 리스트, 객체 리스트, 바닥 타일 딕셔너리
        """
        walls = []
        objects = []
        floor_tiles = {}  # {(x, y): color}
        
        for y, row in enumerate(self.emoji_render):
            for x, emoji in enumerate(row):
                # 🤖와 🎯는 이미 _parse_map_data에서 ⬜️로 교체되었으므로 여기서는 처리하지 않음
                if emoji == '🤖' or emoji == '🎯':
                    # 이미 처리되었어야 하지만, 혹시 모를 경우를 대비해 빈 공간으로 처리
                    continue
                
                # 이모지 정의 확인
                if emoji not in self.emoji_objects:
                    # 정의되지 않은 이모지는 경고 후 무시
                    print(f"경고: 정의되지 않은 이모지 '{emoji}'가 맵에 있습니다 (위치: ({x}, {y})). 무시됩니다.")
                    continue
                
                emoji_def = self.emoji_objects[emoji]
                obj_type = emoji_def.get('type', 'wall')
                
                # 벽 추가 (외벽 포함, CustomRoomEnv가 자동으로 외벽을 생성하지만
                # emoji_render에 명시된 외벽도 처리하여 색상 등을 반영)
                if obj_type == 'wall':
                    color = emoji_def.get('color', 'grey')
                    # 모든 벽 추가 (외벽 포함)
                    walls.append((x, y, color))
                
                elif obj_type == 'emoji':
                    # 이모지 객체 생성
                    obj_config = {
                        'type': 'emoji',
                        'pos': (x, y),
                        'emoji_name': emoji_def.get('emoji_name', 'emoji'),
                        'color': emoji_def.get('color', 'yellow'),
                        'can_pickup': emoji_def.get('can_pickup', False),
                        'can_overlap': emoji_def.get('can_overlap', False),
                        'use_emoji_color': emoji_def.get('use_emoji_color', True)
                    }
                    # 외벽이 아닌 경우만 추가
                    if 0 < x < self.size - 1 and 0 < y < self.size - 1:
                        objects.append(obj_config)
                
                elif obj_type == 'floor':
                    # 바닥 타일: color 속성만 받음
                    color = emoji_def.get('color', 'grey')
                    floor_tiles[(x, y)] = color
                
                elif obj_type == 'goal':
                    # 목표 타일: goal_pos로 이미 설정되었으므로 빈 공간으로 처리
                    # (goal은 CustomRoomEnv에서 goal_pos로 자동 배치됨)
                    pass
                
                elif obj_type == 'empty' or obj_type == 'space':
                    # 빈 공간은 아무것도 하지 않음
                    pass
        
        return walls, objects, floor_tiles
    
    def create_room_config(self) -> Dict:
        """
        room_config 생성
        
        Returns:
            room_config 딕셔너리
        """
        walls, objects, floor_tiles = self._parse_emoji_map()
        
        # CustomRoomEnv는 자동으로 외벽을 생성하므로 외벽은 추가하지 않음
        # 하지만 emoji_render에 외벽이 명시적으로 표시되어 있으면 추가
        # (외벽 위치의 벽은 이미 _parse_emoji_map에서 처리됨)
        
        room_config = {
            'start_pos': self.start_pos,
            'goal_pos': self.goal_pos,
            'walls': walls,
            'objects': objects,
            **self.robot_config  # 로봇 설정 병합
        }
        
        # 바닥 타일이 있으면 추가
        if floor_tiles:
            room_config['floor_tiles'] = floor_tiles
        
        return room_config
    
    def create_wrapper(self) -> MiniGridEmojiWrapper:
        """
        MiniGridEmojiWrapper 생성 (절대 움직임 모드 활성화)
        
        Returns:
            MiniGridEmojiWrapper 인스턴스 (use_absolute_movement=True)
        """
        room_config = self.create_room_config()
        return MiniGridEmojiWrapper(size=self.size, room_config=room_config, use_absolute_movement=True)


def load_emoji_map_from_json(json_path: str) -> MiniGridEmojiWrapper:
    """
    JSON 파일에서 이모지 맵을 로드하여 환경 생성
    
    Args:
        json_path: JSON 파일 경로
    
    Returns:
        MiniGridEmojiWrapper: 생성된 환경 (절대 움직임 모드 활성화)
    """
    loader = EmojiMapLoader(json_path)
    return loader.create_wrapper()


if __name__ == "__main__":
    # 사용 예제
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python emoji_map_loader.py <json_file_path>")
        print("예제: python emoji_map_loader.py example_map.json")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    print(f"JSON 파일에서 맵 로드 중: {json_path}")
    wrapper = load_emoji_map_from_json(json_path)
    
    print("환경 초기화 중...")
    wrapper.reset()
    
    state = wrapper.get_state()
    print(f"에이전트 위치: {state['agent_pos']}")
    print(f"에이전트 방향: {state['agent_dir']}")
    
    print("\n맵이 성공적으로 로드되었습니다!")

