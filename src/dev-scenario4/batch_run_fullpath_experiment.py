"""
Scenario 4 Full Path 배치 실험 (dev-scenario4)

sc4-batchexpr-groundings/ 안의 grounding JSON을 사용해 scenario4_fullpath_experiment.py 를
병렬로 실행하고, 로그는 logs/batch_expr/ 아래에 가중치 접두사({8,1,1}, {1,1,8} 등)로 저장합니다.
실패 시 exponential backoff 후 재시도.

Usage:
    cd src/
    python dev-scenario4/batch_run_fullpath_experiment.py [--no-execute] [--mission "..."]
    python dev-scenario4/batch_run_fullpath_experiment.py --workers 4 --max-retries 3

- sc4-batchexpr-groundings/*.json 파일명이 {a,b,c}_... 형태면 가중치 접두사로 사용.
- 로그: dev-scenario4/logs/batch_expr/{8,1,1}_fullpath_sc4_scenario4_map_YYYYMMDD_HHMMSS/
"""

import sys
import time
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

# 병렬·재시도 설정
DEFAULT_WORKERS = 4
DEFAULT_MAX_RETRIES = 3
BACKOFF_BASE = 2.0

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


def _run_one_experiment(args_tuple):
    """
    단일 실험 실행 (프로세스 풀 워커용).
    실패 시 exponential backoff 후 재시도.
    Returns: (weight: str, log_dir: str, success: bool, error_msg: str | None)
    """
    import importlib.util

    (
        jpath_str,
        mission,
        log_dir_str,
        map_json_str,
        no_execute,
        fix_path,
        debug,
        dev_dir_str,
        src_dir_str,
        max_retries,
        backoff_base,
    ) = args_tuple

    if str(src_dir_str) not in sys.path:
        sys.path.insert(0, str(src_dir_str))

    jpath = Path(jpath_str)
    log_dir = Path(log_dir_str)
    dev_dir = Path(dev_dir_str)
    weight = _weight_prefix_from_filename(jpath.name)

    exp_script = dev_dir / "scenario4_fullpath_experiment.py"
    spec = importlib.util.spec_from_file_location("scenario4_fullpath_experiment", exp_script)
    exp_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp_mod)
    Scenario4FullpathExperiment = exp_mod.Scenario4FullpathExperiment
    # 배치 시 OpenCV 창 끄기
    if hasattr(exp_mod, "SHOW_OPENCV_WINDOW"):
        exp_mod.SHOW_OPENCV_WINDOW = False

    last_error = None
    for attempt in range(max_retries):
        try:
            experiment = Scenario4FullpathExperiment(
                json_map_path=map_json_str,
                grounding_json_path=jpath,
                log_dir=log_dir,
                no_execute=no_execute,
                fix_path=fix_path,
                debug=debug,
            )
            experiment.run(mission=mission)
            experiment.cleanup()
            return (weight, log_dir_str, True, None)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_sec = backoff_base ** attempt
                time.sleep(wait_sec)
    return (weight, log_dir_str, False, str(last_error))


def main():
    import argparse
    import utils.prompt_manager.terminal_formatting_utils as tfu

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
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_WORKERS}). 1 = sequential.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Max retries per run on failure with exponential backoff (default: {DEFAULT_MAX_RETRIES})",
    )
    parser.add_argument(
        "--backoff-base",
        type=float,
        default=BACKOFF_BASE,
        help=f"Base seconds for exponential backoff (default: {BACKOFF_BASE})",
    )
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

    ts_batch = datetime.now().strftime("%Y%m%d_%H%M%S")
    tasks = []
    for i, jpath in enumerate(json_files):
        weight = _weight_prefix_from_filename(jpath.name)
        log_dir = LOGS_BATCH_BASE / f"{weight}_fullpath_sc4_scenario4_map_{ts_batch}_{i}"
        tasks.append(
            (
                str(jpath),
                mission,
                str(log_dir),
                str(MAP_JSON),
                args.no_execute,
                args.fix_path,
                args.debug,
                str(DEV_DIR),
                str(SRC_DIR),
                args.max_retries,
                args.backoff_base,
            )
        )

    workers = max(1, args.workers)
    tfu.cprint(f"Running {len(tasks)} experiments with {workers} worker(s).", tfu.LIGHT_CYAN)
    if workers > 1:
        tfu.cprint(f"On failure: max_retries={args.max_retries}, backoff_base={args.backoff_base}s", tfu.LIGHT_BLACK)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    ok_count = 0
    fail_count = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_one_experiment, t): t for t in tasks}
        for future in as_completed(futures):
            task = futures[future]
            weight = _weight_prefix_from_filename(Path(task[0]).name)
            try:
                weight_out, log_dir_out, success, err_msg = future.result()
                if success:
                    ok_count += 1
                    tfu.cprint(f"  [OK] {weight_out} -> {Path(log_dir_out).name}", tfu.LIGHT_GREEN)
                else:
                    fail_count += 1
                    tfu.cprint(f"  [FAIL] {weight_out}: {err_msg}", tfu.LIGHT_RED, bold=True)
            except Exception as e:
                fail_count += 1
                tfu.cprint(f"  [FAIL] {weight}: {e}", tfu.LIGHT_RED, bold=True)
                import traceback
                traceback.print_exc()

    tfu.cprint(f"Batch done. OK: {ok_count}, FAIL: {fail_count}", tfu.LIGHT_GREEN)


if __name__ == "__main__":
    main()
