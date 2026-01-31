"""
Scenario 4 Full Path Experiment (dev-scenario4)

FullPathSubtaskExperiment를 상속하여 다음만 오버라이드:
- Grounding: 지정 JSON 파일의 grounding_content 또는 vlm_output 영역을 JSON 블록으로 불러와
  system prompt의 $grounding_content에 치환.
- 이미지: dev-scenario4/sc4.png 사용 (VLM/시각화용).
- System prompt: dev-scenario4/scenario4_fullpath_oneshot_systemprompt.txt.
- Logging: dev-scenario4/logs/ 에 저장.

utils/ 라이브러리는 수정하지 않음.

Usage:
    cd src/
    python dev-scenario4/scenario4_fullpath_experiment.py [json_map_path] [grounding_json_path]
    python dev-scenario4/scenario4_fullpath_experiment.py config/scenario4_map.json dev-scenario4/scenario4_map_vlm_output.json

    --no-execute, --fix-path, --debug 는 scenario2_test_fullpath_subtask.py 와 동일.
"""

import json
import re
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Union

# src/ 를 path에 추가
_DEV_DIR = Path(__file__).resolve().parent
_SRC_DIR = _DEV_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import cv2
import numpy as np
import utils.prompt_manager.terminal_formatting_utils as tfu
from utils.miscellaneous.global_variables import PROMPT_DIR, DEFAULT_INITIAL_MISSION, DEBUG

# scenario2_test_fullpath_subtask 에서 클래스·함수 import (utils 수정 없음)
from scenario2_test_fullpath_subtask import (
    FullPathSubtaskExperiment,
    parse_fullpath_response,
    path_to_actions,
    render_path_on_map,
    minigrid_to_chess,
    _display_opencv,
    _close_opencv_windows,
    OPENCV_WINDOW_NAME,
)


# ---------- 설정 (경로) ----------
DEFAULT_SYSTEM_PROMPT_PATH = _DEV_DIR / "scenario4_fullpath_oneshot_systemprompt.txt"
DEFAULT_IMAGE_PATH = _DEV_DIR / "sc4.png"
DEFAULT_GROUNDING_JSON_PATH = _DEV_DIR / "scenario4_map_vlm_output.json"
DEFAULT_LOGS_BASE = _DEV_DIR / "logs"
# OpenCV 창 표시: True = 맵/경로 시각화 창 띄움, False = 창 없이 터미널만
SHOW_OPENCV_WINDOW = False
# VLM 출력 길이: subtask 10개 이상 등 긴 경로 시 65535 권장 (Gemini 2.5 Flash max output)
FULLPATH_MAX_TOKENS = 65535


def _load_image_rgb(image_path: Path) -> np.ndarray:
    """이미지 파일을 RGB numpy (H,W,3) uint8 로 로드 (env.get_image() 형식과 동일)."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_grounding_from_json(json_path: Path) -> str:
    """
    지정 JSON 파일에서 grounding 블록 문자열 반환.
    키 'grounding_content' 또는 'vlm_output' 중 존재하는 것을 사용.
    JSON 블록으로 넣을 수 있도록 문자열 그대로 반환 (이미 JSON 문자열이면 그대로).
    """
    if not json_path.is_file():
        raise FileNotFoundError(f"Grounding JSON not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("grounding_content") or data.get("vlm_output") or ""
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False, indent=2)
    return raw.strip() if isinstance(raw, str) else str(raw)


class Scenario4FullpathExperiment(FullPathSubtaskExperiment):
    """
    Scenario 4 전용 fullpath 실험:
    - Grounding: JSON 파일의 grounding_content / vlm_output → system prompt $grounding_content
    - 이미지: sc4.png
    - System prompt: scenario4_fullpath_oneshot_systemprompt.txt
    - Log: dev-scenario4/logs/
    """

    def __init__(
        self,
        json_map_path: str = None,
        grounding_json_path: Union[str, Path] = None,
        image_path: Union[str, Path] = None,
        system_prompt_path: Union[str, Path] = None,
        log_dir: Optional[Union[str, Path]] = None,
        no_execute: bool = False,
        fix_path: bool = False,
        debug: bool = None,
        **kwargs,
    ):
        if json_map_path is None:
            json_map_path = str(_SRC_DIR / "config" / "scenario4_map.json")
        map_name = Path(json_map_path).stem
        if log_dir is not None:
            log_dir = Path(log_dir).resolve()
        else:
            log_dir = DEFAULT_LOGS_BASE / f"fullpath_sc4_{map_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(
            json_map_path=json_map_path,
            log_dir=log_dir,
            no_execute=no_execute,
            fix_path=fix_path,
            **kwargs,
        )
        self.log_dir = log_dir

        # VLM 출력 길이 확대 (subtask 10개 이상 등 긴 경로 완성용)
        # thinking_budget=0 으로 끄면 max_output_tokens 전부가 응답 JSON에 사용됨 (Gemini 2.5 Flash)
        if hasattr(self, "vlm_processor") and self.vlm_processor is not None:
            w = getattr(self.vlm_processor, "vlm", None)
            if w is not None:
                if hasattr(w, "max_tokens"):
                    w.max_tokens = FULLPATH_MAX_TOKENS
                if hasattr(w, "_handler") and w._handler is not None:
                    w._handler.max_tokens = FULLPATH_MAX_TOKENS
                    if hasattr(w._handler, "thinking_budget"):
                        w._handler.thinking_budget = 0
        if self.debug:
            tfu.cprint(f"[Debug] max_tokens={FULLPATH_MAX_TOKENS}, thinking_budget=0 (scenario4 fullpath)", tfu.LIGHT_BLACK)

        self._grounding_json_path = Path(grounding_json_path) if grounding_json_path else DEFAULT_GROUNDING_JSON_PATH
        self._image_path = Path(image_path) if image_path else DEFAULT_IMAGE_PATH
        self._system_prompt_path = Path(system_prompt_path) if system_prompt_path else DEFAULT_SYSTEM_PROMPT_PATH
        if debug is not None:
            self.debug = debug

    def _load_fullpath_prompts(
        self, mission: str, agent_x: int, agent_y: int, grid_size: int
    ):
        """System prompt: scenario4_fullpath_oneshot_systemprompt.txt + $grounding_content 치환. User prompt: utils/prompts 동일."""
        base_dir = _SRC_DIR / PROMPT_DIR
        user_path = base_dir / "user_prompt_fullpath_subtask.txt"
        if not user_path.exists():
            raise FileNotFoundError(f"User prompt not found: {user_path}")
        user_text = user_path.read_text(encoding="utf-8")
        # Env agent_pos is 0-based; minigrid_to_chess expects 1-based x (A=1..).
        agent_chess = minigrid_to_chess(agent_x + 1, agent_y, grid_size)
        user_prompt = user_text.replace("$mission", mission)
        user_prompt = user_prompt.replace("$agent_x", str(agent_x))
        user_prompt = user_prompt.replace("$agent_y", str(agent_y))
        user_prompt = user_prompt.replace("$agent_chess", agent_chess)

        if not self._system_prompt_path.exists():
            raise FileNotFoundError(f"System prompt not found: {self._system_prompt_path}")
        system_prompt = self._system_prompt_path.read_text(encoding="utf-8")
        grounding_block = _load_grounding_from_json(self._grounding_json_path)
        system_prompt = system_prompt.replace("$grounding_content", grounding_block)
        return system_prompt, user_prompt

    def run(self, mission: Optional[str] = None):
        """initialize 후 이미지만 sc4.png 로 교체, 나머지는 부모 run() 로직 유지. mission이 주어지면 입력 프롬프트 없이 사용(배치용)."""
        self.initialize()
        self.image = _load_image_rgb(self._image_path)
        self.state = self.wrapper.get_state()
        agent_pos = self.state["agent_pos"]
        if isinstance(agent_pos, np.ndarray):
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        else:
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])

        if SHOW_OPENCV_WINDOW:
            _display_opencv(self.image)
        if mission is None:
            if SHOW_OPENCV_WINDOW:
                tfu.cprint("\n[OpenCV] Map (sc4.png) is displayed. Enter your mission (User Prompt) in the terminal and press Enter.", tfu.LIGHT_CYAN)
            else:
                tfu.cprint("\nEnter your mission (User Prompt) and press Enter.", tfu.LIGHT_CYAN)
            raw_input = input("User Prompt (mission): ").strip()
            instruction = raw_input if raw_input else DEFAULT_INITIAL_MISSION
        else:
            instruction = mission
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
        # VLM 호출 직전에 max_tokens·thinking_budget 재설정 (API에 실제 전달되는 값 보장)
        w = getattr(self.vlm_processor, "vlm", None)
        if w is not None and hasattr(w, "_handler") and w._handler is not None:
            w.max_tokens = FULLPATH_MAX_TOKENS
            w._handler.max_tokens = FULLPATH_MAX_TOKENS
            if hasattr(w._handler, "thinking_budget"):
                w._handler.thinking_budget = 0
        tfu.cprint(f"[VLM] max_output_tokens={FULLPATH_MAX_TOKENS}, thinking_budget=0 (fullpath)", tfu.LIGHT_CYAN)
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
        if SHOW_OPENCV_WINDOW:
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
        if SHOW_OPENCV_WINDOW:
            cv2.waitKey(0)


def main():
    import argparse
    import signal
    import atexit
    from utils.miscellaneous.global_variables import MAP_FILE_NAME

    def _sigint_close_opencv(signum, frame):
        _close_opencv_windows()
        raise KeyboardInterrupt()

    parser = argparse.ArgumentParser(description="Scenario 4 fullpath experiment (grounding from JSON, sc4.png, dev-scenario4/logs)")
    parser.add_argument("json_map_path", nargs="?", default=None, help="Path to map JSON (default: config/scenario4_map.json)")
    parser.add_argument("grounding_json_path", nargs="?", default=None, help="Path to grounding JSON (default: dev-scenario4/scenario4_map_vlm_output.json)")
    parser.add_argument("--no-execute", action="store_true", help="Only predict and visualize; do not execute path")
    parser.add_argument("--fix-path", action="store_true", help="Fix path endpoints if mismatch with start_goal")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory (default: dev-scenario4/logs/fullpath_sc4_...)")
    parser.add_argument("--mission", type=str, default=None, help="Mission string for batch mode (skip interactive input)")
    args = parser.parse_args()

    json_map_path = args.json_map_path
    if json_map_path is None:
        json_map_path = str(_SRC_DIR / "config" / "scenario4_map.json")
    grounding_json_path = args.grounding_json_path
    if grounding_json_path is None:
        grounding_json_path = str(DEFAULT_GROUNDING_JSON_PATH)
    debug = args.debug or DEBUG

    try:
        signal.signal(signal.SIGINT, _sigint_close_opencv)
    except (ValueError, OSError):
        pass
    atexit.register(_close_opencv_windows)

    try:
        experiment = Scenario4FullpathExperiment(
            json_map_path=json_map_path,
            grounding_json_path=grounding_json_path,
            log_dir=args.log_dir,
            use_logprobs=False,
            debug=debug,
            no_execute=args.no_execute,
            fix_path=args.fix_path,
        )
        experiment.run(mission=args.mission)
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
