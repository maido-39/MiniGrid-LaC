#!/usr/bin/env python3
"""
dev-memory: Prompt + 이미지로 VLM(gemini-2.5-flash, GCP key) 실행 후
  - 반환 JSON
  - memory 내용
  - memory로 렌더된 프롬프트
를 출력하여 프롬프트 개발을 빠르게 할 수 있도록 하는 스크립트.

Minigrid 환경/step 전혀 없음. 이미 렌더해 둔 예시 이미지(dummy_img)를 VLM에 넘기는 용도.

사용법 (src/ 에서 실행):
  python dev-memory/run_memory_dev.py              # user prompt 터미널 입력 대기
  python dev-memory/run_memory_dev.py "Pick up the key."
  (기본: -p dev_prompt.txt, -i dummy_img.png)
"""

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from string import Template

from utils.prompt_manager.prompt_interp import system_prompt_interp, _substitute_memory_brackets
from utils.vlm.vlm_processor import VLMProcessor
from utils.miscellaneous.global_variables import (
    USE_GCP_KEY,
    VLM_MAX_TOKENS,
    VLM_TEMPERATURE,
    VLM_THINKING_BUDGET,
)
import utils.prompt_manager.terminal_formatting_utils as tfu

_DEFAULT_SYSTEM_PROMPT_FILE = "dev_prompt.txt"
_DEFAULT_IMAGE = "dummy_img.png"


def _render_system_prompt(prompt_file: str, memory: dict, strict: bool = False) -> str:
    """dev-memory 내부에 해당 파일이 있으면 내용 읽어 _substitute_memory_brackets + Template으로 로컬 렌더, 없으면 system_prompt_interp 사용."""
    candidate = _SCRIPT_DIR / prompt_file.strip()
    if candidate.is_file():
        template_text = candidate.read_text(encoding="utf-8")
        memory_dict = memory if isinstance(memory, dict) else {}
        template_text = _substitute_memory_brackets(
            template_text, memory_dict, strict=strict, file_name=prompt_file
        )
        vars_for_template = {"last_action_str": "None", "grounding_content": ""}
        return Template(template_text).safe_substitute(**vars_for_template)
    return system_prompt_interp(
        file_name=prompt_file,
        strict=strict,
        last_action_str="None",
        grounding_content="",
        memory=memory,
    )


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
        description="Prompt + 이미지로 VLM 실행 후 JSON/memory/렌더된 프롬프트 출력 (Minigrid·step 없음, 예시 이미지 사용)"
    )
    parser.add_argument(
        "user_prompt",
        nargs="?",
        default=None,
        help="사용자(미션) 프롬프트. 생략하면 터미널에서 입력 대기.",
    )
    parser.add_argument(
        "-p", "--prompt",
        default=_DEFAULT_SYSTEM_PROMPT_FILE,
        help=f"시스템 프롬프트 파일 (기본: {_DEFAULT_SYSTEM_PROMPT_FILE})",
    )
    parser.add_argument(
        "-i", "--image",
        default=str(_SCRIPT_DIR / _DEFAULT_IMAGE),
        help=f"입력 이미지 경로 (기본: dev-memory/{_DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="파싱된 JSON 저장 경로 (선택)",
    )
    parser.add_argument(
        "--out-rendered",
        default=None,
        help="렌더된 프롬프트 저장 경로 (선택)",
    )
    args = parser.parse_args()

    prompt_file = args.prompt.strip()
    user_prompt_text = (args.user_prompt or "").strip()
    if not user_prompt_text:
        try:
            user_prompt_text = input("User prompt (미션): ").strip() or "Go to the target. Select one action."
        except (EOFError, KeyboardInterrupt):
            tfu.cprint("\n[Aborted]", tfu.LIGHT_YELLOW)
            sys.exit(0)
        if not user_prompt_text:
            user_prompt_text = "Go to the target. Select one action."
    image_path = Path(args.image.strip())
    if not image_path.is_file():
        tfu.cprint(f"[Error] Image not found: {image_path}", tfu.RED, bold=True)
        sys.exit(1)

    # 1) 빈 memory로 시스템 프롬프트 생성 (VLM 호출용). dev-memory 내 파일이면 문자열 렌더 오버라이드.
    try:
        system_prompt_empty = _render_system_prompt(prompt_file, memory={}, strict=False)
    except Exception as e:
        tfu.cprint(f"[Error] Failed to load prompt '{prompt_file}': {e}", tfu.RED, bold=True)
        sys.exit(1)

    # 2) VLM 호출 (gemini-2.5-flash, GCP key)
    tfu.cprint("\n[1] Calling VLM (gemini-2.5-flash, GCP key)...", tfu.CYAN, bold=True)
    processor = _create_vlm_processor()
    raw_response = processor.requester(
        image=str(image_path.resolve()),
        system_prompt=system_prompt_empty,
        user_prompt=user_prompt_text,
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

    # 4) memory로 프롬프트 렌더 (동일하게 _render_system_prompt 사용)
    try:
        rendered_prompt = _render_system_prompt(prompt_file, memory=memory, strict=False)
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
