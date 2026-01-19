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
from .minigrid_customenv_emoji import MiniGridEmojiWrapper

# Import terminal formatting utils for important messages
from ..prompt_manager import terminal_formatting_utils as tfu
from ..prompt_manager.terminal_formatting_utils import (
    GREEN, YELLOW, CYAN
)


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
        
        # Emoji object definitions (JSON only, no defaults)
        if 'emoji_objects' not in self.map_data:
            raise ValueError("JSON file does not contain 'emoji_objects' key.")
        self.emoji_objects = self.map_data['emoji_objects']
        
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
                    # Replace marker with ⬜️ (treat as empty space)
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
            # Warning if marker found but JSON also has start_pos
            if 'start_pos' in self.map_data:
                tfu.cprint(
                    f"Warning: Found robot marker (🤖) and set start_pos to ({self.start_pos[0]}, {self.start_pos[1]}). "
                    f"JSON start_pos is ignored.",
                    YELLOW,
                    bold=True
                )
        
        # If 🎯 marker not found, use JSON goal_pos (or default)
        if not goal_marker_found:
            self.goal_pos = tuple(self.map_data.get('goal_pos', [self.size - 2, self.size - 2]))
        else:
            # Warning if 🎯 marker found but JSON also has goal_pos
            if 'goal_pos' in self.map_data:
                tfu.cprint(
                    f"Warning: Found 🎯 marker and set goal_pos to ({self.goal_pos[0]}, {self.goal_pos[1]}). "
                    f"JSON goal_pos is ignored.",
                    YELLOW,
                    bold=True
                )
        
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
        Parse emoji map to create walls, objects, floor_tiles lists
        
        Returns:
            (walls, objects, floor_tiles): wall list, object list, floor tile dictionary
        """
        walls = []
        objects = []
        floor_tiles = {}  # {(x, y): color}
        
        for y, row in enumerate(self.emoji_render):
            for x, emoji in enumerate(row):
                # 🤖 and 🎯 are already replaced with ⬜️ in _parse_map_data, so skip here
                if emoji == '🤖' or emoji == '🎯':
                    # Should already be processed, but treat as empty space just in case
                    continue
                
                # Check emoji definition
                # IMPORTANT: Always check emoji_objects first, even for ⬛
                # This ensures JSON definitions are respected, especially for outer walls
                if emoji not in self.emoji_objects:
                    # Undefined emoji: warn and ignore
                    tfu.cprint(
                        f"Warning: Undefined emoji '{emoji}' found in map (position: ({x}, {y})). Ignored.",
                        YELLOW,
                        bold=True
                    )
                    continue
                
                emoji_def = self.emoji_objects[emoji]
                obj_type = emoji_def.get('type', 'wall')
                
                # Add walls (including outer walls, CustomRoomEnv automatically creates outer walls
                # but we also process explicitly specified outer walls in emoji_render to reflect colors, etc.)
                if obj_type == 'wall':
                    color = emoji_def.get('color', 'grey')
                    # Add all walls (including outer walls)
                    walls.append((x, y, color))
                
                elif obj_type == 'emoji':
                    # Create emoji object
                    obj_config = {
                        'type': 'emoji',
                        'pos': (x, y),
                        'emoji_name': emoji_def.get('emoji_name', 'emoji'),
                        'color': emoji_def.get('color', 'yellow'),
                        'can_pickup': emoji_def.get('can_pickup', False),
                        'can_overlap': emoji_def.get('can_overlap', False),
                        'use_emoji_color': emoji_def.get('use_emoji_color', True)
                    }
                    # Only add if not outer wall
                    if 0 < x < self.size - 1 and 0 < y < self.size - 1:
                        objects.append(obj_config)
                
                elif obj_type == 'floor':
                    # Floor tile: only receives color attribute
                    color = emoji_def.get('color', 'grey')
                    floor_tiles[(x, y)] = color
                
                elif obj_type == 'goal':
                    # Goal tile: already set as goal_pos, so treat as empty space
                    # (goal is automatically placed at goal_pos by CustomRoomEnv)
                    pass
                
                elif obj_type == 'empty' or obj_type == 'space':
                    # Empty space: do nothing
                    pass
        
        return walls, objects, floor_tiles
    
    def create_room_config(self) -> Dict:
        """
        Create room_config
        
        Returns:
            room_config dictionary
        """
        walls, objects, floor_tiles = self._parse_emoji_map()
        
        # CustomRoomEnv automatically creates outer walls, so we don't add them
        # But if outer walls are explicitly specified in emoji_render, we add them
        # (outer wall positions are already processed in _parse_emoji_map)
        
        room_config = {
            'start_pos': self.start_pos,
            'goal_pos': self.goal_pos,
            'walls': walls,
            'objects': objects,
            **self.robot_config  # Merge robot configuration
        }
        
        # Add floor tiles if present
        if floor_tiles:
            room_config['floor_tiles'] = floor_tiles
        
        return room_config
    
    def create_wrapper(self) -> MiniGridEmojiWrapper:
        """
        Create MiniGridEmojiWrapper (absolute movement mode enabled)
        
        Returns:
            MiniGridEmojiWrapper instance (use_absolute_movement=True)
        """
        room_config = self.create_room_config()
        return MiniGridEmojiWrapper(size=self.size, room_config=room_config, use_absolute_movement=True)


def load_emoji_map_from_json(json_path: str) -> MiniGridEmojiWrapper:
    """Load an emoji-based map from a JSON file and create a MiniGrid environment.
    
    This is a convenience function that loads a map definition from a JSON file
    and creates a fully configured MiniGridEmojiWrapper environment. The JSON
    file should contain emoji-based map data with object definitions.
    
    Args:
        json_path: Path to the JSON file containing the map definition.
            The JSON file should have a structure like:
            {
                "map": {
                    "emoji_render": "⬛⬛⬛...\n⬛⬜️⬜️...\n...",
                    "emoji_objects": {
                        "⬛": {"type": "wall", "color": "grey", ...},
                        "🟩": {"type": "emoji", "emoji_name": "brick", ...},
                        ...
                    },
                    "start_pos": [1, 1],
                    "goal_pos": [8, 8],
                    "robot_config": {...}
                }
            }
    
    Returns:
        MiniGridEmojiWrapper: A configured environment instance with:
            - Absolute movement mode enabled (use_absolute_movement=True)
            - Map loaded from JSON file
            - Objects, walls, and floor tiles placed according to the map
            - Start and goal positions set
    
    Raises:
        FileNotFoundError: If the JSON file does not exist.
        ValueError: If the JSON file is malformed or missing required keys.
        ValueError: If the map is not square or has inconsistent row lengths.
    
    Examples:
        >>> # Load a map from JSON file
        >>> wrapper = load_emoji_map_from_json("config/example_map.json")
        >>> 
        >>> # Reset and use the environment
        >>> obs, info = wrapper.reset()
        >>> state = wrapper.get_state()
        >>> print(f"Agent starts at: {state['agent_pos']}")
        >>> 
        >>> # Execute actions
        >>> obs, reward, done, truncated, info = wrapper.step("move up")
    
    Note:
        The created environment will have absolute movement mode enabled by
        default, which is recommended for VLM control. The robot can be placed
        using the 🤖 emoji in the map, and the goal can be placed using the 🎯
        emoji. These markers will be automatically replaced with empty spaces.
    """
    loader = EmojiMapLoader(json_path)
    return loader.create_wrapper()


if __name__ == "__main__":
    # Usage example
    import sys
    
    if len(sys.argv) < 2:
        tfu.cprint("Usage: python emoji_map_loader.py <json_file_path>", CYAN)
        tfu.cprint("Example: python emoji_map_loader.py example_map.json", CYAN)
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    tfu.cprint(f"Loading map from JSON file: {json_path}", CYAN, bold=True)
    wrapper = load_emoji_map_from_json(json_path)
    
    tfu.cprint("Initializing environment...", CYAN)
    wrapper.reset()
    
    state = wrapper.get_state()
    tfu.cprint(f"Agent position: {state['agent_pos']}", GREEN)
    tfu.cprint(f"Agent direction: {state['agent_dir']}", GREEN)
    
    tfu.cprint("\nMap loaded successfully!", GREEN, bold=True)

