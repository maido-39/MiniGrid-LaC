"""
Scenario 2 Full Path Subtask Experiment

Predicts the full path from a language instruction by:
1. Decomposing the instruction into subtasks (single VLM call)
2. For each subtask: semantic/spatial reasoning and path (list of coordinates)
3. Optionally executing the path in the environment
4. Rendering the predicted path on the map with start-to-end gradient
5. Logging everything to JSON for reproducibility and analysis.

Usage:
    cd src/
    python scenario2_test_fullpath_subtask.py [json_map_path]
    python scenario2_test_fullpath_subtask.py --no-execute config/example_map.json
    python scenario2_test_fullpath_subtask.py --fix-path config/scenario_2_4_map.json
    python scenario2_test_fullpath_subtask.py --debug config/scenario_2_4_map.json

    --no-execute: Only predict and visualize; do not execute path in env.
    --fix-path: If path endpoints mismatch start_goal, prepend/append to fix (literature-based path correction).
    --debug: Print debug info (raw response length, retries, timing, start_goal).

Logs are saved under: logs/fullpath_subtask_{map_name}_{timestamp}/
"""

import os
import sys
import json
import re
import csv
import argparse
import uuid
import time
import atexit
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import cv2

import utils.prompt_manager.terminal_formatting_utils as tfu
from utils.miscellaneous.scenario_runner import ScenarioExperiment
from utils.miscellaneous.safe_minigrid_registration import safe_minigrid_reg
from utils.miscellaneous.global_variables import (
    MAP_FILE_NAME,
    DEFAULT_INITIAL_MISSION,
    VLM_MODEL,
    DEBUG,
    PROMPT_DIR,
)
from utils.map_manager.emoji_map_loader import load_emoji_map_from_json

try:
    from matplotlib import cm as _mpl_cm
except ImportError:
    _mpl_cm = None

safe_minigrid_reg()

# OpenCV window for map and path visualization (single window, updated in place)
OPENCV_WINDOW_NAME = "FullPath Map"


def _close_opencv_windows() -> None:
    """Close OpenCV windows so they do not persist after script exit."""
    try:
        cv2.destroyWindow(OPENCV_WINDOW_NAME)
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


atexit.register(_close_opencv_windows)


def _sigint_close_opencv(signum, frame):
    """On Ctrl+C, close OpenCV windows first so they do not persist."""
    _close_opencv_windows()
    raise KeyboardInterrupt()


# ---------------------------------------------------------------------------
# Chess-style coordinate conversion (Alphabet + Number, e.g. A2, B7, G7)
# Chess format: column letter(s) + row number (e.g. G7, F11). Row 1 = bottom.
# Minigrid: (1,1) = top-left, x = column, y = row (increases downward).
# Rule:
#   Chess -> Minigrid: Num = row part, Alp = letter part.
#     minigrid_Y = grid_size - Num - 1  (invert, 1-based)
#     minigrid_X = alphabet to 1-based index (A=1, B=2, ...)
#   Minigrid -> Chess: Num = grid_size - minigrid_Y - 1, Letter = A + minigrid_X
# ---------------------------------------------------------------------------


def chess_to_minigrid(chess_str: str, grid_size: int) -> Tuple[int, int]:
    """
    Parse chess-style cell string (e.g. "G7", "F11") to minigrid (x, y).
    Format: column letter(s) + row number (AlphabetNum). Row 1 = bottom.
    minigrid_Y = grid_size - Num - 1; minigrid_X = A->1, B->2, ..., H->8, L->12 (1-based).
    """
    s = str(chess_str).strip()
    m = re.match(r"^([A-Za-z]+)(\d+)$", s)
    if not m:
        raise ValueError(f"Invalid chess coordinate: {chess_str!r}; expected format e.g. A2, G7, F11")
    col_str = m.group(1).upper()
    num = int(m.group(2))
    if num < 1 or num > grid_size:
        raise ValueError(f"Chess row {num} out of range [1, {grid_size}]")
    # Column: A=1, B=2, ..., H=8, L=12 (1-based minigrid X). Do NOT subtract 1.
    col_index = 0
    for c in col_str:
        col_index = col_index * 26 + (ord(c) - ord("A") + 1)
    if col_index < 1 or col_index > grid_size:
        raise ValueError(f"Chess column {col_str!r} out of range for grid_size={grid_size}")
    minigrid_x = col_index  # 1-based: L -> 12 (not 11)
    minigrid_y = grid_size - num - 1
    if minigrid_y < 0 or minigrid_y >= grid_size:
        raise ValueError(f"Chess row {num} out of range for grid_size={grid_size} (minigrid_y={minigrid_y})")
    return (minigrid_x, minigrid_y)


def minigrid_to_chess(x: int, y: int, grid_size: int) -> str:
    """Convert minigrid (x, y) to chess-style string e.g. 'A2', 'G7' (Alphabet + Num). x is 1-based (A=1..), y is 0-based."""
    if x < 1 or x > grid_size or y < 0 or y >= grid_size:
        raise ValueError(f"Minigrid ({x},{y}) out of range: x in [1, {grid_size}], y in [0, {grid_size})")
    num = grid_size - y - 1
    # x is 1-based: 1->A, 2->B, ..., 8->H, 12->L
    col_letter = chr(ord("A") + (x - 1)) if x <= 26 else ""
    if x > 26:
        t = x
        while t > 0:
            t, r = divmod(t - 1, 26)
            col_letter = chr(ord("A") + r) + col_letter
    return f"{col_letter}{num}"


def _cell_to_minigrid(cell: Any, grid_size: int) -> List[int]:
    """Convert one cell (chess notation only, e.g. F6, G7) to minigrid [x, y]."""
    if isinstance(cell, str):
        x, y = chess_to_minigrid(cell, grid_size)
        return [x, y]
    raise ValueError(f"Cell must be chess notation (e.g. F6, G7); got {type(cell)}: {cell}")


def _extract_start_goal_cell(item: Any) -> Any:
    """From start_goal entry: dict with 'coordinate' -> that value; else the item itself (chess string)."""
    if isinstance(item, dict) and "coordinate" in item:
        return item["coordinate"]
    return item


def _display_opencv(img: np.ndarray, window_name: str = OPENCV_WINDOW_NAME) -> None:
    """
    Show image in OpenCV window (RGB input). Resize for display like Visualizer (max 800px).
    """
    if img is None:
        return
    try:
        img_bgr = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
        if img_bgr.dtype != np.uint8:
            img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)
        h, w = img_bgr.shape[:2]
        max_size = 800
        if h < max_size and w < max_size:
            scale = min(max_size // h, max_size // w, 4)
            if scale > 1:
                new_w, new_h = w * scale, h * scale
                img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(window_name, img_bgr)
        cv2.waitKey(1)
    except Exception as e:
        tfu.cprint(f"[OpenCV] Display error: {e}", tfu.LIGHT_RED)


def _fix_path_endpoints_for_subtask(val: dict) -> None:
    """
    In-place fix: ensure path[0] == start_goal[0] and path[-1] == start_goal[1]
    by prepending/appending start_goal cells if needed (literature: path correction).
    """
    sg = val.get("start_goal")
    path = val.get("path")
    if not isinstance(sg, (list, tuple)) or len(sg) != 2 or not isinstance(path, (list, tuple)) or len(path) < 2:
        return
    try:
        start_cell = list(int(x) for x in sg[0])
        goal_cell = list(int(x) for x in sg[1])
        path_start = list(int(x) for x in path[0]) if path else []
        path_end = list(int(x) for x in path[-1]) if path else []
    except (TypeError, ValueError):
        return
    if start_cell != path_start:
        val["path"] = [start_cell] + list(path)
        path = val["path"]
    if goal_cell != list(path[-1]) if path else []:
        val["path"] = list(path) + [goal_cell]


def parse_fullpath_response(
    raw_response: str,
    grid_size: int,
    fix_path_endpoints: bool = False,
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Parse VLM raw response into fullpath action structure.
    Expects JSON with top-level "action" object containing subtask1, subtask2, ...
    each with "reasoning", "start_goal", "path".
    Prompt output is chess notation only: start_goal and path use chess strings (e.g. F6, B7).
    All are converted to minigrid [x,y] in parsed output; chess originals stored
    in start_goal_chess / path_chess per subtask for logging.

    If fix_path_endpoints is True, when path[0]!=start_goal[0] or path[-1]!=start_goal[1],
    prepend/append start_goal cells to path (literature-based path correction).

    Returns:
        (parsed_dict, parse_error). On success parsed_dict is the full parsed object and parse_error is None.
        On failure parsed_dict is None and parse_error is the error message.
    """
    if not raw_response or not raw_response.strip():
        return None, "Empty response"

    text = raw_response.strip()
    if "```json" in text:
        start_idx = text.find("```json") + 7
        end_idx = text.find("```", start_idx)
        if end_idx != -1:
            text = text[start_idx:end_idx].strip()
        else:
            text = text[start_idx:].strip()
    elif "```" in text:
        start_idx = text.find("```") + 3
        end_idx = text.find("```", start_idx)
        if end_idx != -1:
            text = text[start_idx:end_idx].strip()
        else:
            text = text[start_idx:].strip()

    if not text:
        return None, "No JSON content extracted"

    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return None, "Response is not a JSON object"
        if "action" not in parsed:
            return None, "Missing top-level 'action' key"
        action = parsed["action"]
        if not isinstance(action, dict):
            return None, "'action' is not an object"
        for key in sorted(action.keys()):
            if not re.match(r"subtask\d+", key):
                continue
            val = action[key]
            if not isinstance(val, dict):
                return None, f"'{key}' value is not an object"
            if "reasoning" not in val or "start_goal" not in val or "path" not in val:
                return None, f"'{key}' missing reasoning, start_goal, or path"
            sg = val.get("start_goal")
            path_raw = val.get("path")
            # Accept start_goal as object {"start": ..., "goal": ...} or list [start, goal]; values must be chess notation
            if isinstance(sg, dict) and "start" in sg and "goal" in sg:
                start_item, goal_item = sg["start"], sg["goal"]
            elif isinstance(sg, (list, tuple)) and len(sg) == 2:
                start_item, goal_item = sg[0], sg[1]
            else:
                return None, f"'{key}.start_goal' must be {{ \"start\": ..., \"goal\": ... }} (each value chess notation or dict with 'coordinate')"
            if not isinstance(path_raw, (list, tuple)):
                return None, f"'{key}.path' must be a list of cells in chess notation"
            if len(path_raw) < 2:
                return None, f"'{key}.path' must have at least 2 cells (start and goal)"
            start_cell_raw = _extract_start_goal_cell(start_item)
            goal_cell_raw = _extract_start_goal_cell(goal_item)
            try:
                start_minigrid = _cell_to_minigrid(start_cell_raw, grid_size)
                goal_minigrid = _cell_to_minigrid(goal_cell_raw, grid_size)
                path_minigrid = [_cell_to_minigrid(c, grid_size) for c in path_raw]
            except (ValueError, TypeError) as e:
                return None, f"'{key}' coordinate conversion: {e}"
            # Store chess notation for logging (prompt output is chess only)
            def to_chess_str(c: Any) -> str:
                return c if isinstance(c, str) else str(c)
            val["start_goal_chess"] = [to_chess_str(start_cell_raw), to_chess_str(goal_cell_raw)]
            val["path_chess"] = [to_chess_str(c) for c in path_raw]
            if isinstance(start_item, dict) or isinstance(goal_item, dict):
                val["start_goal_semantic"] = [
                    start_item.get("semantic_position") if isinstance(start_item, dict) else None,
                    goal_item.get("semantic_position") if isinstance(goal_item, dict) else None,
                ]
            val["start_goal"] = [start_minigrid, goal_minigrid]
            val["path"] = path_minigrid
            path_start = tuple(path_minigrid[0])
            path_end = tuple(path_minigrid[-1])
            start_cell = tuple(start_minigrid)
            goal_cell = tuple(goal_minigrid)
            if fix_path_endpoints:
                _fix_path_endpoints_for_subtask(val)
                path_minigrid = val.get("path", [])
                if len(path_minigrid) >= 2:
                    path_start = tuple(path_minigrid[0])
                    path_end = tuple(path_minigrid[-1])
            if path_start != start_cell:
                return None, f"'{key}.path[0]' must equal start_goal[0]; got path[0]={path_start}, start_goal[0]={start_cell}"
            if path_end != goal_cell:
                return None, f"'{key}.path[-1]' must equal start_goal[1]; got path[-1]={path_end}, start_goal[1]={goal_cell}"
        return parsed, None
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {e}"


def path_to_actions(
    path: List[List[int]], current_pos: Tuple[int, int], current_dir: int
) -> List[Tuple[int, Any]]:
    """
    Convert a path (list of [x,y]) into a sequence of (action_index, (nx,ny))
    for absolute movement. Grid: x=col, y=row (y increases downward).
    Absolute action: 0=north(up), 1=south(down), 2=west(left), 3=east(right).
    (dx,dy) = (nx-px, ny-py): north=(0,-1), south=(0,1), west=(-1,0), east=(1,0).
    """
    direction_to_absolute = {
        (0, -1): 0,   # north / up
        (0, 1): 1,    # south / down
        (-1, 0): 2,   # west / left
        (1, 0): 3,    # east / right
    }
    result = []
    pos = list(current_pos)
    for pt in path:
        nx, ny = int(pt[0]), int(pt[1])
        dx = nx - pos[0]
        dy = ny - pos[1]
        if dx == 0 and dy == 0:
            continue
        if abs(dx) + abs(dy) != 1:
            continue
        key = (dx, dy)
        if key not in direction_to_absolute:
            key = (-dx, -dy)
            if key in direction_to_absolute:
                key = (dx, dy)
        act = direction_to_absolute.get(key)
        if act is not None:
            result.append((act, (nx, ny)))
        pos = [nx, ny]
    return result


def render_path_on_map(
    image: np.ndarray,
    parsed_action: Dict[str, Any],
    grid_size: int,
    tile_size: int = 32,
    color_start: Tuple[int, int, int] = (0, 200, 0),
    color_end: Tuple[int, int, int] = (200, 50, 50),
    inset: int = 1,
    colormap: Optional[str] = "viridis",
) -> np.ndarray:
    """
    Draw each subtask path on the map image. Gradient is applied globally over all segments
    (subtask1 start -> last subtask end) so sequence order is visible.
    Path is minigrid (x,y): x is 1-based (A=1..), y is 0-based. Chess: Alphabet+Num, row 1 = bottom.
    Outermost 1 row/column is not map; inset=1 shifts drawing inward by one tile.

    colormap: If set (e.g. 'viridis', 'plasma', 'inferno', 'cividis'), use matplotlib's
    perceptual colormap; None or '' uses linear RGB between color_start and color_end.
    """
    out = image.copy()
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    h, w = out.shape[:2]

    def cell_to_pixel(cx: int, cy: int) -> Tuple[int, int]:
        # cx 1-based, cy 0-based. Inset: first map cell at (tile_size, tile_size).
        px = int(inset * tile_size + (cx - 1) * tile_size + tile_size // 2)
        py = int(inset * tile_size + cy * tile_size + tile_size // 2)
        return (min(max(px, 0), w - 1), min(max(py, 0), h - 1))

    # Academic-style gradient: matplotlib viridis/plasma/inferno/cividis (perceptually uniform)
    if colormap and _mpl_cm is not None:
        try:
            cmap = _mpl_cm.get_cmap(colormap)
        except (ValueError, AttributeError):
            cmap = _mpl_cm.get_cmap("viridis")

        def interp(t: float) -> Tuple[int, int, int]:
            rgba = cmap(t)
            r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
            return (b, g, r)
    else:
        def interp(t: float) -> Tuple[int, int, int]:
            r = int(color_start[0] * (1 - t) + color_end[0] * t)
            g = int(color_start[1] * (1 - t) + color_end[1] * t)
            b = int(color_start[2] * (1 - t) + color_end[2] * t)
            return (b, g, r)

    subtask_keys = sorted([k for k in parsed_action.get("action", {}).keys() if re.match(r"subtask\d+", k)])
    # Count total segments (all subtasks) for global gradient
    total_segments = 0
    for sk in subtask_keys:
        path = parsed_action["action"][sk].get("path", [])
        total_segments += max(0, len(path) - 1)
    total_segments = max(1, total_segments)

    global_seg_idx = 0
    thickness = max(2, tile_size // 8)
    for idx, sk in enumerate(subtask_keys):
        val = parsed_action["action"][sk]
        path = val.get("path", [])
        if not path:
            continue
        pts = [cell_to_pixel(int(p[0]), int(p[1])) for p in path]
        n = len(pts)
        if n < 2:
            if n == 1:
                t = global_seg_idx / total_segments if total_segments else 0.0
                cx, cy = int(path[0][0]), int(path[0][1])
                px, py = cell_to_pixel(cx, cy)
                c = interp(t)
                cv2.circle(out, (px, py), max(2, tile_size // 4), c, -1)
            continue
        for i in range(n - 1):
            t = global_seg_idx / (total_segments - 1) if total_segments > 1 else 0.0
            c = interp(t)
            cv2.line(out, pts[i], pts[i + 1], c, thickness)
            global_seg_idx += 1
        # Start circle: t at beginning of this path
        t_start = (global_seg_idx - (n - 1)) / (total_segments - 1) if total_segments > 1 else 0.0
        t_end = (global_seg_idx - 1) / (total_segments - 1) if total_segments > 1 else 1.0
        if total_segments == 1:
            t_start, t_end = 0.0, 1.0
        cv2.circle(out, pts[0], max(2, tile_size // 4), interp(t_start), -1)
        cv2.circle(out, pts[-1], max(2, tile_size // 4), interp(t_end), -1)

    return out


class FullPathSubtaskExperiment(ScenarioExperiment):
    """
    Full-path subtask experiment: one VLM call for subtask decomposition and
    per-subtask paths; optional execution; path visualization with gradient; JSON log.
    """

    def __init__(self, *args, no_execute: bool = False, fix_path: bool = False, **kwargs):
        kwargs_no_exec = {k: v for k, v in kwargs.items() if k not in ("no_execute", "fix_path")}
        json_map_path = kwargs_no_exec.get("json_map_path") or (args[1] if len(args) > 1 else None)
        if json_map_path is None:
            json_map_path = f"config/{MAP_FILE_NAME}"
        map_name = Path(json_map_path).stem
        log_dir = Path("logs") / f"fullpath_subtask_{map_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        kwargs_no_exec["log_dir"] = log_dir
        super().__init__(*args, **kwargs_no_exec)
        self.no_execute = no_execute
        self.fix_path = fix_path
        self.vlm_response_raw = ""
        self.vlm_response_parsed = {}
        self.parse_error = None
        self.execution_steps = []
        self.execution_success = False
        self.path_visualization_path = None
        if hasattr(self.vlm_processor, "vlm"):
            w = self.vlm_processor.vlm
            if hasattr(w, "max_tokens"):
                w.max_tokens = 32768
            if hasattr(w, "_handler") and hasattr(w._handler, "max_tokens"):
                w._handler.max_tokens = 32768
        if self.debug:
            tfu.cprint(f"[Debug] log_dir={self.log_dir}", tfu.LIGHT_BLACK)
            tfu.cprint(f"[Debug] max_tokens=32768 (fullpath)", tfu.LIGHT_BLACK)

    def _init_csv_logging(self):
        """Minimal CSV: one row per run (optional)."""
        csv_path = self.log_dir / "experiment_log.csv"
        file_exists = csv_path.exists()
        self.csv_file = open(csv_path, "a", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file, quoting=csv.QUOTE_ALL)
        if not file_exists:
            self.csv_writer.writerow([
                "run_id", "timestamp", "instruction", "parse_ok", "executed", "success", "path_image"
            ])

    def _load_fullpath_prompts(
        self, mission: str, agent_x: int, agent_y: int, grid_size: int
    ) -> Tuple[str, str]:
        """Load and substitute system and user prompts for fullpath subtask."""
        base_dir = Path(__file__).parent / PROMPT_DIR
        sys_path = base_dir / "system_prompt_fullpath_subtask.txt"
        user_path = base_dir / "user_prompt_fullpath_subtask.txt"
        if not sys_path.exists():
            raise FileNotFoundError(f"System prompt not found: {sys_path}")
        if not user_path.exists():
            raise FileNotFoundError(f"User prompt not found: {user_path}")
        system_prompt = sys_path.read_text(encoding="utf-8")
        user_text = user_path.read_text(encoding="utf-8")
        # Env agent_pos is 0-based; minigrid_to_chess expects 1-based x (A=1..).
        agent_chess = minigrid_to_chess(agent_x + 1, agent_y, grid_size)
        user_prompt = user_text.replace("$mission", mission)
        user_prompt = user_prompt.replace("$agent_x", str(agent_x))
        user_prompt = user_prompt.replace("$agent_y", str(agent_y))
        user_prompt = user_prompt.replace("$agent_chess", agent_chess)
        return system_prompt, user_prompt

    def vlm_gen_fullpath_subtask(
        self, image: np.ndarray, system_prompt: str, user_prompt: str
    ) -> str:
        """Single VLM call for fullpath; returns raw string."""
        tfu.cprint("\n[VLM] Sending full-path subtask request...", tfu.LIGHT_CYAN)
        raw = self.vlm_processor.requester(
            image=image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            grounding_file=None,
            debug=self.debug,
        )
        return raw or ""

    def run(self):
        """One-shot: initialize, show map in OpenCV, get User Prompt, VLM call, parse, optional execute, visualize in OpenCV, log JSON."""
        self.initialize()
        self.image = self.wrapper.get_image()
        self.state = self.wrapper.get_state()
        agent_pos = self.state["agent_pos"]
        if isinstance(agent_pos, np.ndarray):
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        else:
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])

        # Show map in OpenCV window, then get User Prompt from terminal
        _display_opencv(self.image)
        tfu.cprint("\n[OpenCV] Map is displayed. Enter your mission (User Prompt) in the terminal and press Enter.", tfu.LIGHT_CYAN)
        raw_input = input("User Prompt (mission): ").strip()
        instruction = raw_input if raw_input else DEFAULT_INITIAL_MISSION
        self.user_prompt = instruction

        tfu.cprint("\n" + "=" * 80, bold=True)
        tfu.cprint("Instruction:", tfu.YELLOW, bold=True)
        tfu.cprint(instruction, tfu.LIGHT_BLACK)
        tfu.cprint("=" * 80 + "\n", bold=True)

        grid_size = getattr(self.wrapper, "size", 12)
        self._run_grid_size = grid_size
        self._run_agent_start = (agent_x, agent_y)
        system_prompt, user_prompt = self._load_fullpath_prompts(
            instruction, agent_x, agent_y, grid_size
        )
        max_parse_retries = 3
        self.vlm_response_raw = ""
        self.vlm_response_parsed = {}
        self.parse_error = None
        for attempt in range(1, max_parse_retries + 1):
            if self.debug:
                tfu.cprint(f"[Debug] VLM+parse attempt {attempt}/{max_parse_retries}", tfu.LIGHT_BLACK)
            t0 = datetime.now()
            self.vlm_response_raw = self.vlm_gen_fullpath_subtask(
                self.image, system_prompt, user_prompt
            )
            elapsed = (datetime.now() - t0).total_seconds()
            if self.debug:
                tfu.cprint(f"[Debug] VLM elapsed={elapsed:.2f}s, raw_response len={len(self.vlm_response_raw or '')}", tfu.LIGHT_BLACK)
            if not self.vlm_response_raw:
                tfu.cprint(f"[Attempt {attempt}/{max_parse_retries}] VLM returned empty response.", tfu.LIGHT_RED)
                if attempt < max_parse_retries:
                    tfu.cprint("[Retry] Will retry VLM call...", tfu.LIGHT_YELLOW)
                    time.sleep(2)
                continue
            grid_size = getattr(self.wrapper, "size", 12)
            self.vlm_response_parsed, self.parse_error = parse_fullpath_response(
                self.vlm_response_raw,
                grid_size=grid_size,
                fix_path_endpoints=getattr(self, "fix_path", False),
            )
            if self.parse_error:
                tfu.cprint(f"[Attempt {attempt}/{max_parse_retries}] Parse failed: {self.parse_error}", tfu.LIGHT_RED)
                if self.debug:
                    tail = (self.vlm_response_raw or "")[-200:]
                    tfu.cprint(f"[Debug] raw_response tail (200 chars): {repr(tail)}", tfu.LIGHT_BLACK)
                if attempt < max_parse_retries:
                    tfu.cprint("[Retry] Will retry VLM call (no forced parsing).", tfu.LIGHT_YELLOW)
                    time.sleep(2)
                continue
            break
        if self.parse_error or not self.vlm_response_parsed:
            tfu.cprint(f"[Error] All {max_parse_retries} attempts failed. Last error: {self.parse_error}", tfu.LIGHT_RED)
            self._write_json_log(instruction, executed=False)
            return

        tfu.cprint("[Parse OK] Full-path response parsed.", tfu.LIGHT_GREEN)
        if self.debug:
            tfu.cprint(f"[Debug] raw_response length={len(self.vlm_response_raw or '')} chars", tfu.LIGHT_BLACK)
        action = self.vlm_response_parsed.get("action", {})
        for k in sorted(action.keys()):
            if re.match(r"subtask\d+", k):
                path_len = len(action[k].get("path", []))
                tfu.cprint(f"  {k}: reasoning + start_goal + path (len={path_len})", tfu.LIGHT_BLACK)
                if self.debug:
                    sg = action[k].get("start_goal", [])
                    tfu.cprint(f"       start_goal={sg}", tfu.LIGHT_BLACK)

        self.visualizer.visualize_grid_cli(self.wrapper, self.state)

        grid_size = getattr(self.wrapper, "size", 12)
        tile_size = getattr(self.wrapper.env, "tile_size", 32) if self.wrapper else 32
        vis_img = render_path_on_map(
            self.image.copy(),
            self.vlm_response_parsed,
            grid_size,
            tile_size,
        )
        self.path_visualization_path = "path_visualization.png"
        out_path = self.log_dir / self.path_visualization_path
        cv2.imwrite(str(out_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        tfu.cprint(f"[Visualization] Saved: {out_path}", tfu.LIGHT_GREEN)
        # Update OpenCV window with path visualization result
        _display_opencv(vis_img)
        tfu.cprint("[OpenCV] Path visualization updated. Press any key in the OpenCV window to close.", tfu.LIGHT_CYAN)

        if not self.no_execute and self.vlm_response_parsed:
            self._execute_paths()
        else:
            self.execution_success = False
            if self.no_execute:
                tfu.cprint("[No Execute] Skipping path execution.", tfu.LIGHT_YELLOW)

        self._write_json_log(instruction, executed=not self.no_execute and len(self.execution_steps) > 0)
        tfu.cprint("\nExperiment complete. Logs saved to:", self.log_dir, tfu.LIGHT_GREEN)
        # Wait for key in OpenCV window so user can see the result before closing
        cv2.waitKey(0)

    def _execute_paths(self):
        """Execute each subtask path in order via wrapper.step."""
        action = self.vlm_response_parsed.get("action", {})
        subtask_keys = sorted([k for k in action.keys() if re.match(r"subtask\d+", k)])
        self.execution_steps = []
        self.execution_success = False
        grid_size = getattr(self.wrapper, "size", 12)
        state = self.wrapper.get_state()
        agent_pos = state["agent_pos"]
        if isinstance(agent_pos, np.ndarray):
            pos = (int(agent_pos[0]), int(agent_pos[1]))
        else:
            pos = (int(agent_pos[0]), int(agent_pos[1]))
        agent_dir = int(state["agent_dir"])

        for sk in subtask_keys:
            val = action[sk]
            path = val.get("path", [])
            if not path:
                continue
            # Path has 1-based x (from chess); wrapper uses 0-based. Convert for execution.
            path_0based = [[int(p[0]) - 1, int(p[1])] for p in path]
            seq = path_to_actions(path_0based, pos, agent_dir)
            for act_idx, (nx, ny) in seq:
                state_before = {"agent_pos": list(pos), "agent_dir": agent_dir}
                state_before_chess = {"agent_pos_chess": minigrid_to_chess(pos[0] + 1, pos[1], grid_size)}
                action_name = ["move up", "move down", "move left", "move right"][act_idx]
                try:
                    _, reward, term, trunc, _ = self.wrapper.step(act_idx)
                    done = term or trunc
                except Exception as e:
                    tfu.cprint(f"Step failed: {e}", tfu.LIGHT_RED)
                    done = True
                    reward = 0.0
                state_after = self.wrapper.get_state()
                pos = (int(state_after["agent_pos"][0]), int(state_after["agent_pos"][1]))
                agent_dir = int(state_after["agent_dir"])
                state_after_chess = {"agent_pos_chess": minigrid_to_chess(pos[0] + 1, pos[1], grid_size)}
                self.execution_steps.append({
                    "subtask_id": sk,
                    "state_before": state_before,
                    "state_before_chess": state_before_chess,
                    "action_taken": action_name,
                    "state_after": {"agent_pos": list(pos), "agent_dir": agent_dir},
                    "state_after_chess": state_after_chess,
                    "reward": float(reward),
                    "done": bool(done),
                })
                if done:
                    self.execution_success = True
                    break
            if self.execution_steps and self.execution_steps[-1].get("done"):
                break

        if self.execution_steps:
            self.execution_success = self.execution_steps[-1].get("done", False)
        tfu.cprint(f"[Execution] Steps: {len(self.execution_steps)}, success: {self.execution_success}", tfu.LIGHT_CYAN)

    def _write_json_log(self, instruction: str, executed: bool):
        """Write one JSON log file per run with meta, config, instruction, vlm, execution, visualization."""
        run_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        map_name = Path(self.json_map_path).stem if self.json_map_path else "unknown"

        grid_size = getattr(self, "_run_grid_size", None)
        agent_start = getattr(self, "_run_agent_start", None)
        coordinates_section = {}
        if grid_size is not None and agent_start is not None:
            # agent_start from env is 0-based; log minigrid with 1-based x (A=1..).
            ax_0, ay = agent_start[0], agent_start[1]
            coordinates_section = {
                "grid_size": grid_size,
                "agent_start": {
                    "minigrid": [ax_0 + 1, ay],
                    "chess": minigrid_to_chess(ax_0 + 1, ay, grid_size),
                },
            }

        log_data = {
            "meta": {
                "run_id": run_id,
                "timestamp": timestamp,
                "script": "scenario2_test_fullpath_subtask.py",
                "map_file": self.json_map_path,
                "map_name": map_name,
                "vlm_model": getattr(self, "vlm_model", VLM_MODEL),
            },
            "config": {
                "json_map_path": self.json_map_path,
                "vlm_model": getattr(self, "vlm_model", VLM_MODEL),
                "no_execute": getattr(self, "no_execute", False),
                "fix_path": getattr(self, "fix_path", False),
            },
            "coordinates": coordinates_section,
            "instruction": instruction,
            "vlm": {
                "raw_response": self.vlm_response_raw,
                "parsed": self.vlm_response_parsed,
                "parse_error": self.parse_error,
            },
            "execution": {
                "executed": executed,
                "steps": self.execution_steps,
                "success": self.execution_success,
            },
            "visualization": {
                "path_image": self.path_visualization_path,
            },
        }

        from utils.miscellaneous.episode_manager import convert_numpy_types
        log_data = convert_numpy_types(log_data)

        log_path = self.log_dir / "experiment_log_fullpath.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        # Collapse [num, num] arrays to one line to avoid noisy indent (path/start_goal)
        with open(log_path, "r", encoding="utf-8") as f:
            text = f.read()
        text = re.sub(r"\n(\s+)\[\s*\n\s+(-?\d+)\s*,\s*\n\s+(-?\d+)\s*\n\s+\]", r"\n\1[\2, \3]", text)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(text)
        tfu.cprint(f"[Log] Wrote {log_path}", tfu.LIGHT_GREEN)

        if self.csv_file and self.csv_writer:
            try:
                self.csv_writer.writerow([
                    run_id,
                    timestamp,
                    instruction[:200],
                    self.parse_error is None,
                    executed,
                    self.execution_success,
                    self.path_visualization_path or "",
                ])
                self.csv_file.flush()
            except Exception as e:
                tfu.cprint(f"[Warning] CSV write failed: {e}", tfu.LIGHT_YELLOW)

    def cleanup(self):
        """Close OpenCV window first, then run parent cleanup."""
        _close_opencv_windows()
        super().cleanup()


def main():
    parser = argparse.ArgumentParser(description="Full-path subtask experiment")
    parser.add_argument("json_map_path", nargs="?", default=None, help="Path to map JSON")
    parser.add_argument("--no-execute", action="store_true", help="Only predict and visualize; do not execute path")
    parser.add_argument("--fix-path", action="store_true", help="If path endpoints mismatch start_goal, prepend/append to fix (literature-based path correction)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output (raw response length, retries, timing)")
    args = parser.parse_args()
    json_map_path = args.json_map_path
    if json_map_path is None:
        json_map_path = f"config/{MAP_FILE_NAME}"
    debug = args.debug or DEBUG

    # Ctrl+C 시 OpenCV 창이 즉시 닫히도록 SIGINT 핸들러 등록
    try:
        signal.signal(signal.SIGINT, _sigint_close_opencv)
    except (ValueError, OSError):
        pass  # main thread가 아닐 때 등록 실패 가능

    try:
        experiment = FullPathSubtaskExperiment(
            json_map_path=json_map_path,
            use_logprobs=False,
            debug=debug,
            no_execute=args.no_execute,
            fix_path=args.fix_path,
        )
        experiment.run()
        experiment.cleanup()
    except KeyboardInterrupt:
        tfu.cprint("\nInterrupted by user.", tfu.LIGHT_BLUE, bold=True)
        if "experiment" in dir():
            experiment.cleanup()
    except Exception as e:
        tfu.cprint(f"\nError: {e}", tfu.LIGHT_RED, bold=True)
        import traceback
        traceback.print_exc()
        if "experiment" in dir():
            experiment.cleanup()


if __name__ == "__main__":
    main()
