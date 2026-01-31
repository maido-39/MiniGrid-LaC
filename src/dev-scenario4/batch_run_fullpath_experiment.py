"""
Scenario 4 Full Path 배치 실험 (dev-scenario4)

sc4-batchexpr-groundings/ 안의 grounding JSON을 사용해 scenario4_fullpath_experiment.py 를
각각 실행하고, 로그는 logs/batch_expr/ 아래에 가중치 접두사({8,1,1}, {1,1,8} 등)로 저장합니다.

Usage:
    cd src/
    python dev-scenario4/batch_run_fullpath_experiment.py [--no-execute] [--mission "..."]

- sc4-batchexpr-groundings/*.json 파일명이 {a,b,c}_... 형태면 가중치 접두사로 사용.
- 로그: dev-scenario4/logs/batch_expr/{8,1,1}_fullpath_sc4_scenario4_map_YYYYMMDD_HHMMSS/
"""

import sys
from pathlib import Path
from datetime import datetime

DEV_DIR = Path(__file__).resolve().parent
SRC_DIR = DEV_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

GROUNDINGS_DIR = DEV_DIR / "sc4-batchexpr-groundings"
LOGS_BATCH_BASE = DEV_DIR / "logs" / "batch_expr"
MAP_JSON = SRC_DIR / "config" / "scenario4_map.json"
INSTRUCTION_TXT = DEV_DIR / "instruction.txt"

# 배치용 기본 mission (instruction.txt [Instruction] 본문, 없으면 고정 문자열)
DEFAULT_MISSION = (
    "1. gather green pants and put into brown basket\n"
    "2. gather purple t-shirt inside bedroom\n"
    "3. get water from kitchen\n"
    "4. gather soap and do laundary inside shower room"
)


def _instruction_body_from_file() -> str:
    lines = []
    if not INSTRUCTION_TXT.is_file():
        return DEFAULT_MISSION
    with open(INSTRUCTION_TXT, "r", encoding="utf-8") as f:
        in_instruction = False
        for line in f:
            if line.strip() == "[Instruction]":
                in_instruction = True
                continue
            if in_instruction and line.strip():
                lines.append(line.rstrip())
    return "\n".join(lines) if lines else DEFAULT_MISSION


def _weight_prefix_from_filename(filename: str) -> str:
    """파일명에서 가중치 접두사 추출. 예: {8,1,1}_scenario4_map_vlm_output.json -> {8,1,1}"""
    if filename.startswith("{"):
        end = filename.find("}")
        if end != -1:
            return filename[: end + 1]
    return Path(filename).stem


def main():
    import argparse
    import importlib.util
    import utils.prompt_manager.terminal_formatting_utils as tfu

    # dev-scenario4는 패키지가 아니므로 파일 경로로 로드
    exp_script = DEV_DIR / "scenario4_fullpath_experiment.py"
    spec = importlib.util.spec_from_file_location("scenario4_fullpath_experiment", exp_script)
    exp_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp_mod)
    Scenario4FullpathExperiment = exp_mod.Scenario4FullpathExperiment

    parser = argparse.ArgumentParser(
        description="Run scenario4 fullpath experiment for each grounding JSON in sc4-batchexpr-groundings/"
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Only predict and visualize; do not execute path",
    )
    parser.add_argument(
        "--mission",
        type=str,
        default=None,
        help=f"Mission string for all runs (default: from {INSTRUCTION_TXT.name} [Instruction] or fixed)",
    )
    parser.add_argument("--fix-path", action="store_true", help="Fix path endpoints if mismatch with start_goal")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    if not GROUNDINGS_DIR.is_dir():
        print(f"Error: Groundings directory not found: {GROUNDINGS_DIR}", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(GROUNDINGS_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files in {GROUNDINGS_DIR}", file=sys.stderr)
        sys.exit(1)

    mission = args.mission if args.mission is not None else _instruction_body_from_file()
    LOGS_BATCH_BASE.mkdir(parents=True, exist_ok=True)

    for jpath in json_files:
        weight = _weight_prefix_from_filename(jpath.name)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = LOGS_BATCH_BASE / f"{weight}_fullpath_sc4_scenario4_map_{ts}"

        tfu.cprint(f"[{weight}] Running: {jpath.name} -> {log_dir.name}", tfu.LIGHT_CYAN)
        try:
            experiment = Scenario4FullpathExperiment(
                json_map_path=str(MAP_JSON),
                grounding_json_path=jpath,
                log_dir=log_dir,
                no_execute=args.no_execute,
                fix_path=args.fix_path,
                debug=args.debug,
            )
            experiment.run(mission=mission)
            experiment.cleanup()
            tfu.cprint(f"  [OK] Logs: {log_dir}", tfu.LIGHT_GREEN)
        except Exception as e:
            tfu.cprint(f"  [FAIL] {e}", tfu.LIGHT_RED, bold=True)
            import traceback
            traceback.print_exc()

    tfu.cprint("Batch experiment done.", tfu.LIGHT_GREEN)


if __name__ == "__main__":
    main()
