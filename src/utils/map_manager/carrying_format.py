"""
Shared carrying object formatting for terminal display.

Provides format_carrying_object() and get_emoji_char_for_object() so that
scenario_runner, MiniGridEmojiWrapper, and other callers use a single
implementation and fallback emoji map.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, Union


# Fallback emoji map when JSON map or emoji_objects is not available.
# Includes entries from scenario_runner and minigrid_customenv_emoji.EMOJI_MAP.
DEFAULT_EMOJI_MAP = {
    'box': '📦',
    'apple': '🍎',
    'key': '🔑',
    'ball': '⚽',
    'chair': '🪑',
    'tree': '🌲',
    'mushroom': '🍄',
    'flower': '🌼',
    'cat': '🐈',
    'grass': '🌾',
    'rock': '🗿',
    'desktop': '🖥️',
    'workstation': '📱',
    'brick': '🧱',
    'TV': '📺',
    'sofa': '🛋️',
    'bed': '🛏️',
    'desk': '🖥️',
    'lamp': '💡',
    'wall': '🚧',
    'restroom': '🚻',
    'storage': '🗄️',
    'preperation': '🧑‍🍳',
    'kitchen': '🍳',
    'plating': '🍽️',
    'dining': '🍴',
    'water': '💦',
    'waterspill': '🫗',
}


def get_emoji_char_for_object(
    carrying_obj: Any,
    json_map_path: Optional[Union[str, Path]] = None,
    emoji_objects: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Resolve emoji character for a carrying object.
    Tries json_map_path (emoji_objects in map), then emoji_objects dict, then DEFAULT_EMOJI_MAP.
    """
    emoji_name = getattr(carrying_obj, 'emoji_name', None)
    if emoji_name is None:
        return '❓'
    if emoji_objects:
        for key, defn in emoji_objects.items():
            if isinstance(defn, dict) and defn.get('emoji_name') == emoji_name:
                return key
        return DEFAULT_EMOJI_MAP.get(emoji_name, '❓')
    if json_map_path:
        try:
            path = Path(json_map_path)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict) and 'map' in data and 'emoji_objects' in data.get('map', {}):
                    eo = data['map']['emoji_objects']
                    for key, defn in eo.items():
                        if isinstance(defn, dict) and defn.get('emoji_name') == emoji_name:
                            return key
        except Exception:
            pass
    return DEFAULT_EMOJI_MAP.get(emoji_name, '❓')


def format_single_carrying_object(
    carrying_obj: Any,
    json_map_path: Optional[Union[str, Path]] = None,
    emoji_objects: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Format a single carrying object for terminal display.
    Uses get_emoji_char_for_object for emoji resolution.
    """
    if carrying_obj is None:
        return "None"
    emoji_char = get_emoji_char_for_object(carrying_obj, json_map_path=json_map_path, emoji_objects=emoji_objects)
    if hasattr(carrying_obj, 'type'):
        obj_type = carrying_obj.type
        if obj_type == 'emoji' and hasattr(carrying_obj, 'emoji_name'):
            color = getattr(carrying_obj, 'color', 'N/A')
            return f"{emoji_char} {carrying_obj.emoji_name} (color: {color})"
        if obj_type == 'key':
            color = getattr(carrying_obj, 'color', 'N/A')
            return f"🔑 Key (color: {color})"
        if obj_type == 'ball':
            color = getattr(carrying_obj, 'color', 'N/A')
            return f"⚽ Ball (color: {color})"
        if obj_type == 'box':
            color = getattr(carrying_obj, 'color', 'N/A')
            return f"📦 Box (color: {color})"
        color = getattr(carrying_obj, 'color', None)
        if color:
            return f"{obj_type} (color: {color})"
        return f"{obj_type}"
    return str(carrying_obj)


def format_carrying_object(
    carrying_obj: Any,
    json_map_path: Optional[Union[str, Path]] = None,
    emoji_objects: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Format carrying object(s) for terminal display.
    Handles None, single object, or list of objects.
    """
    if carrying_obj is None:
        return "None"
    if isinstance(carrying_obj, list):
        if len(carrying_obj) == 0:
            return "None"
        parts = [format_single_carrying_object(obj, json_map_path=json_map_path, emoji_objects=emoji_objects) for obj in carrying_obj]
        return f"[{', '.join(parts)}]"
    return format_single_carrying_object(carrying_obj, json_map_path=json_map_path, emoji_objects=emoji_objects)
