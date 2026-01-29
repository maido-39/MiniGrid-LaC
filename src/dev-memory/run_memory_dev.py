#!/usr/bin/env python3
"""
dev-memory: Prompt + 이미지로 VLM(gemini-2.5-flash, GCP key) 실행 후
  - 반환 JSON
  - memory 내용
  - memory로 렌더된 프롬프트
를 출력하여 프롬프트 개발을 빠르게 할 수 있도록 하는 스크립트.

사용법 (src/ 에서 실행):
  python dev-memory/run_memory_dev.py --prompt system_prompt_start.txt --image path/to/image.png
  python dev-memory/run_memory_dev.py -p system_prompt_start.txt -i path/to/image.png --user-prompt "Go to restroom."
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 프로젝트 루트(src)를 path에 넣어 utils 임포트 가능하게
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.prompt_manager.prompt_interp import system_prompt_interp
from utils.vlm.vlm_processor import VLMProcessor
from utils.miscellaneous.global_variables import (
    USE_GCP_KEY,
    VLM_MAX_TOKENS,
    VLM_TEMPERATURE,
    VLM_THINKING_BUDGET,
)
import utils.prompt_manager.terminal_formatting_utils as tfu


def _create_vlm_processor():
    """gemini-2.5-flash, GCP key(Vertex AI 또는 credentials)로 VLMProcessor 생성."""
    model = "gemini-2.5-flash"
    vertexai = False
    credentials = None
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    if USE_GCP_KEY and project_id:
        vertexai = True
    return VLMProcessor(
        model=model,
        temperature=VLM_TEMPERATURE,
        max_tokens=VLM_MAX_TOKENS,
        thinking_budget=VLM_THINKING_BUDGET,
        debug=False,
        vertexai=vertexai,
        credentials=credentials,
        project_id=project_id,
        location=location,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prompt 파일 + 이미지로 VLM 실행 후 JSON, memory, 렌더된 프롬프트 출력"
    )
    parser.add_argument(
        "-p", "--prompt",
        required=True,
        help="프롬프트 파일 이름 (utils/prompts/ 아래, 예: system_prompt_start.txt)",
    )
    parser.add_argument(
        "-i", "--image",
        required=True,
        help="입력 이미지 경로 (예: logs_good/.../step_0001.png)",
    )
    parser.add_argument(
        "--user-prompt",
        default="Go to the target. Select one action.",
        help="사용자(미션) 프롬프트 (기본: Go to the target. Select one action.)",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="파싱된 JSON을 저장할 파일 경로 (선택)",
    )
    parser.add_argument(
        "--out-rendered",
        default=None,
        help="memory로 렌더된 프롬프트를 저장할 파일 경로 (선택)",
    )
    args = parser.parse_args()

    prompt_file = args.prompt.strip()
    image_path = Path(args.image.strip())
    if not image_path.is_file():
        tfu.cprint(f"[Error] Image not found: {image_path}", tfu.RED, bold=True)
        sys.exit(1)

    # 1) 빈 memory로 시스템 프롬프트 생성 (VLM 호출용)
    try:
        system_prompt_empty = system_prompt_interp(
            file_name=prompt_file,
            strict=False,
            last_action_str="None",
            grounding_content="",
            memory={},
        )
    except Exception as e:
        tfu.cprint(f"[Error] Failed to load prompt '{prompt_file}': {e}", tfu.RED, bold=True)
        sys.exit(1)

    # 2) VLM 호출 (gemini-2.5-flash, GCP key)
    tfu.cprint("\n[1] Calling VLM (gemini-2.5-flash, GCP key)...", tfu.CYAN, bold=True)
    processor = _create_vlm_processor()
    raw_response = processor.requester(
        image=str(image_path.resolve()),
        system_prompt=system_prompt_empty,
        user_prompt=args.user_prompt,
        debug=False,
    )
    if not raw_response:
        tfu.cprint("[Error] VLM returned empty response.", tfu.RED, bold=True)
        sys.exit(1)

    # 3) JSON 파싱 (memory 포함)
    try:
        parsed = processor.parser_action(raw_response)
    except Exception as e:
        tfu.cprint(f"[Warning] Parse failed, using raw response: {e}", tfu.LIGHT_YELLOW)
        parsed = {"action": [], "reasoning": "", "grounding": "", "memory": {}}
        try:
            parsed = json.loads(raw_response)
        except Exception:
            pass

    memory = parsed.get("memory", {})
    if isinstance(memory, str):
        try:
            memory = json.loads(memory)
        except Exception:
            memory = {}
    if not isinstance(memory, dict):
        memory = {}

    # 4) memory로 프롬프트 렌더
    try:
        rendered_prompt = system_prompt_interp(
            file_name=prompt_file,
            strict=False,
            last_action_str="None",
            grounding_content="",
            memory=memory,
        )
    except Exception as e:
        tfu.cprint(f"[Warning] Render with memory failed: {e}", tfu.LIGHT_YELLOW)
        rendered_prompt = system_prompt_empty

    # 5) 출력
    tfu.cprint("\n" + "=" * 80, tfu.CYAN)
    tfu.cprint("[2] Parsed JSON (content)", tfu.CYAN, bold=True)
    tfu.cprint("=" * 80)
    print(json.dumps(parsed, ensure_ascii=False, indent=2))

    tfu.cprint("\n" + "=" * 80, tfu.CYAN)
    tfu.cprint("[3] Memory (memory block only)", tfu.CYAN, bold=True)
    tfu.cprint("=" * 80)
    print(json.dumps(memory, ensure_ascii=False, indent=2))

    tfu.cprint("\n" + "=" * 80, tfu.CYAN)
    tfu.cprint("[4] Rendered prompt (with memory substituted)", tfu.CYAN, bold=True)
    tfu.cprint("=" * 80)
    print(rendered_prompt)

    if args.out_json:
        out_json_path = Path(args.out_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        tfu.cprint(f"\n[JSON saved] {out_json_path}", tfu.LIGHT_GREEN)

    if args.out_rendered:
        out_rendered_path = Path(args.out_rendered)
        out_rendered_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_rendered_path, "w", encoding="utf-8") as f:
            f.write(rendered_prompt)
        tfu.cprint(f"[Rendered prompt saved] {out_rendered_path}", tfu.LIGHT_GREEN)


if __name__ == "__main__":
    main()
