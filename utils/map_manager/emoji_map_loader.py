"""
Emoji Map JSON Loader and Conversion Module

Reads emoji maps from JSON files and converts them into a minigrid environment.

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
import utils.prompt_manager.terminal_formatting_utils as tfu


class EmojiMapLoader:
    """
    Emoji Map JSON Loader and Conversion Class
    """
    
    def __init__(self, json_path: str):
        """
        Args:
            json_path: JSON file path
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
        """
        Load JSON file
        """
        
        if not self.json_path.exists():
            raise FileNotFoundError(f"The JSON file cannot be found: {self.json_path}")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'map' not in data:
            raise ValueError("The JSON file does not contain a ‘map’ key.")
        
        self.map_data = data['map']
    
    def _parse_emoji_text(self, text: str) -> List[List[str]]:
        """
        Parsing text-based emoji maps into 2D arrays
        
        Args:
            text: Emoji map text separated by line breaks
        
        Returns:
            2D array (list of lists of rows)
        """
        
        lines = text.strip().split('\n')
        # Remove blank lines
        lines = [line.strip() for line in lines if line.strip()]
        
        if len(lines) == 0:
            raise ValueError("The emoji map text is empty.")
        
        result = []
        for line in lines:
            emojis = []
            i = 0
            while i < len(line):
                char = line[i]
                # Variation Selector (U+FE0F) or Zero Width Joiner (U+200D) is combined with the preceding character.
                if ord(char) in [0xFE0F, 0x200D]:
                    # Add to previous emojis (if already added)
                    if emojis:
                        emojis[-1] += char
                    i += 1
                    continue
                
                # Check whether the following character is a Variation Selector or a Zero Width Joiner
                if i + 1 < len(line):
                    next_char = line[i + 1]
                    if ord(next_char) in [0xFE0F, 0x200D]:
                        # If a Variation Selector or Zero Width Joiner is present, they are grouped together.
                        if i + 2 < len(line) and ord(line[i + 2]) in [0xFE0F, 0x200D]:
                            # In the rare but possible case of two combination characters
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
            
            # Remove items containing only empty strings or spaces
            emojis = [e for e in emojis if e.strip()]
            if emojis:
                result.append(emojis)
        
        return result
    
    def _parse_map_data(self):
        """
        Map Data Parsing
        """
        
        # Emoji Render Map
        if 'emoji_render' not in self.map_data:
            raise ValueError("The ‘emoji_render’ key is missing from the JSON file.")
        
        emoji_render_raw = self.map_data['emoji_render']
        
        # Verify text format (string or string array)
        if isinstance(emoji_render_raw, str):
            # Single string: separated by line breaks
            self.emoji_render = self._parse_emoji_text(emoji_render_raw)
        elif isinstance(emoji_render_raw, list) and len(emoji_render_raw) > 0:
            # If the first element is a string, it is in the form of a text array.
            if isinstance(emoji_render_raw[0], str):
                # String array: Each line is a string
                text = '\n'.join(emoji_render_raw)
                self.emoji_render = self._parse_emoji_text(text)
            else:
                # 2D array format (existing method)
                self.emoji_render = emoji_render_raw
        else:
            raise ValueError("'emoji_render' must be a string, an array of strings, or a 2D array.")
        
        # Check map size
        if not isinstance(self.emoji_render, list) or len(self.emoji_render) == 0:
            raise ValueError("The parsing result for ‘emoji_render’ is empty.")
        
        # Row and column numbers are taken from the emoji list.
        self.num_rows = len(self.emoji_render)
        row_lengths = [len(row) for row in self.emoji_render]
        
        # All rows must have the same length.
        if not all(length == row_lengths[0] for length in row_lengths):
            raise ValueError(
                f"All rows in the map must be the same length. "
                f"Number of rows: {self.num_rows}, The length of each row: {row_lengths}"
            )
        
        self.num_cols = row_lengths[0]
        
        # Since MiniGrid uses a square grid, it uses the larger value between the number of rows and the number of columns.
        self.size = max(self.num_rows, self.num_cols)
        
        # If the number of rows and columns differs, a warning is issued.
        if self.num_rows != self.num_cols:
            tfu.cprint(f"Warning: The map is not square. Number of rows: {self.num_rows}, Number of coloumns: {self.num_cols}, "
                  f"Grid size is set as: {self.size}x{self.size}.")
        
        # Emoji Object Definition
        if 'emoji_objects' not in self.map_data:
            raise ValueError("The JSON file does not contain the ‘emoji_objects’ key.")
        
        self.emoji_objects = self.map_data['emoji_objects']
        
        # Robot Settings
        self.robot_config = self.map_data.get('robot_config', {
            'use_robot_emoji': True,
            'robot_emoji_color': 'red',
            'use_robot_emoji_color': True
        })
        
        # Starting point and ending point
        self.start_pos = tuple(self.map_data.get('start_pos', [1, 1]))
        self.goal_pos = tuple(self.map_data.get('goal_pos', [self.size - 2, self.size - 2]))
    
    def _parse_emoji_map(self) -> Tuple[List, List, Dict]:
        """
        Parse the emoji map to generate lists of walls, objects, and floor_tiles
        """
        
        walls = []
        objects = []
        floor_tiles = {}  # {(x, y): color}
        
        for y, row in enumerate(self.emoji_render):
            for x, emoji in enumerate(row):
                
                # --- [Start of modified section] --------------------------------
                # 1. When you find a black square (⬛), always convert it to a ‘brick’ emoji object.
                if emoji == '⬛':
                    obj_config = {
                        'type': 'emoji',
                        'pos': (x, y),
                        'emoji_name': 'brick',   # Enter the desired emoji name or Unicode here (🧱)
                        'color': 'grey',         # Mini Grid Internal Color Codes (if needed)
                        'can_pickup': False,     # Set it so it can't be picked up like a wall
                        'can_overlap': False,    # Set to be impenetrable like a wall
                        'use_emoji_color': True
                    }
                    # If it's an interior space rather than an exterior wall, or if you want to apply it to the entire structure, append without conditions.
                    objects.append(obj_config)
                    continue  # The basic logic below is skipped.
                # -----------------------------------------------------

                # Check emoji definitions
                if emoji not in self.emoji_objects:
                    continue
                
                emoji_def = self.emoji_objects[emoji]
                obj_type = emoji_def.get('type', 'wall')
                
                # Verify robot position marker
                if emoji == '🟥' and obj_type == 'wall':
                    if self.start_pos == (1, 1) or self.start_pos == (x, y):
                        self.start_pos = (x, y)
                    continue
                
                # Standard Wall Treatment (Walls other than ⬛)
                if obj_type == 'wall':
                    color = emoji_def.get('color', 'grey')
                    walls.append((x, y, color))
                
                elif obj_type == 'emoji':
                    obj_config = {
                        'type': 'emoji',
                        'pos': (x, y),
                        'emoji_name': emoji_def.get('emoji_name', 'emoji'),
                        'color': emoji_def.get('color', 'yellow'),
                        'can_pickup': emoji_def.get('can_pickup', False),
                        'can_overlap': emoji_def.get('can_overlap', False),
                        'use_emoji_color': emoji_def.get('use_emoji_color', True)
                    }
                    if 0 < x < self.size - 1 and 0 < y < self.size - 1:
                        objects.append(obj_config)
                
                elif obj_type == 'floor':
                    color = emoji_def.get('color', 'grey')
                    floor_tiles[(x, y)] = color
                
                elif obj_type == 'empty' or obj_type == 'space':
                    pass
        
        return walls, objects, floor_tiles
    
    def create_room_config(self) -> Dict:
        """
        room_config creation
        
        Returns:
            room_config dictionary
        """
        
        walls, objects, floor_tiles = self._parse_emoji_map()
        
        # CustomRoomEnv automatically generates outer walls, so do not add them
        # However, if outer walls are explicitly specified in emoji_render, add them
        # (Walls at outer wall positions are already handled in _parse_emoji_map)
        
        room_config = {
            'start_pos': self.start_pos,
            'goal_pos': self.goal_pos,
            'walls': walls,
            'objects': objects,
            **self.robot_config  # Merge Robot Settings
        }
        
        # If there are floor tiles, add
        if floor_tiles:
            room_config['floor_tiles'] = floor_tiles
        
        return room_config
    
    def create_wrapper(self) -> MiniGridEmojiWrapper:
        """
        Create MiniGridEmojiWrapper (Absolute Movement Mode Enabled)
        
        Returns:
            MiniGridEmojiWrapper instance (use_absolute_movement=True)
        """
        
        room_config = self.create_room_config()
        
        return MiniGridEmojiWrapper(size=self.size, room_config=room_config, use_absolute_movement=True)


def load_emoji_map_from_json(json_path: str) -> MiniGridEmojiWrapper:
    """
    Load an emoji map from a JSON file to create an environment
    
    Args:
        json_path: JSON file path
    
    Returns:
        MiniGridEmojiWrapper: Created environment (absolute motion mode enabled)
    """
    
    loader = EmojiMapLoader(json_path)
    
    return loader.create_wrapper()


if __name__ == "__main__":
    # Usage Examples
    import sys
    
    if len(sys.argv) < 2:
        tfu.cprint("Usage Instructions: python emoji_map_loader.py <json_file_path>")
        tfu.cprint("Example: python emoji_map_loader.py example_map.json")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    tfu.cprint(f"Loading map from JSON file: {json_path}")
    wrapper = load_emoji_map_from_json(json_path)
    
    tfu.cprint("Initializing environment...")
    wrapper.reset()
    
    state = wrapper.get_state()
    tfu.cprint(f"Agent Location: {state['agent_pos']}")
    tfu.cprint(f"Agent Direction: {state['agent_dir']}")
    
    tfu.cprint("\nThe map has been successfully loaded!")



