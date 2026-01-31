"""
Prompt 파일과 (맵 이미지 또는 Minigrid 환경)을 입력받아 VLM 추론 후 출력을 TXT로 저장하는 스크립트.

기본(인터랙티브): 맵 OpenCV 창 + user prompt는 instruction.txt 기본. VLM 호출 후 출력·JSON 저장.
                  시스템 프롬프트는 prompt.txt. 사용 맵 JSON은 스크립트 내 DEFAULT_MAP_JSON 으로 지정 가능.
대안: --batch (-b) 로 프롬프트 파일 + 맵으로 비대화형 실행 후 TXT 저장.
이미지: Minigrid 환경(JSON 맵) 또는 --map-image 로 이미지 파일 직접 지정.

utils/ 아래 코드는 수정하지 않고, VLMWrapper·load_emoji_map_from_json 등 기존 유틸만 사용합니다.

사용법:
    # 기본(인터랙티브): 맵 OpenCV + user prompt는 instruction.txt → VLM 결과 출력 및 JSON 저장
    python vlm_prompt_map_to_txt.py

    # 배치: 프롬프트 파일 + 맵으로 비대화형 실행 후 TXT 저장
    python vlm_prompt_map_to_txt.py --batch --prompt user_prompt.txt

    # 맵 JSON 지정
    python vlm_prompt_map_to_txt.py --map-json config/scenario4_example_map.json

    # 맵 이미지 파일 직접 지정
    python vlm_prompt_map_to_txt.py --map-image path/to/map.png

옵션:
    --batch, -b   비인터랙티브: 프롬프트 파일(-p)을 user prompt로 사용, 출력 TXT 저장 (기본은 인터랙티브)
    --prompt, -p  (배치 모드) user 프롬프트 텍스트 파일 경로 (기본: dev-scenario4/prompt.txt)
    --map-json     맵 JSON 경로 (미지정 시 스크립트 내 DEFAULT_MAP_JSON 사용)
    --instruction, -I  (인터랙티브) user prompt 파일 (기본: dev-scenario4/instruction.txt)
    --map-image, -m  맵 이미지 파일 경로. 지정 시 Minigrid 대신 이 이미지 사용
    --output, -o  출력 경로 (기본: 인터랙티브는 logs/{timestamp}_{맵stem}_vlm_output.json, 배치는 _vlm_output.txt)
    --system, -s  시스템 프롬프트 파일 (기본: dev-scenario4/prompt.txt)
    --debug       VLM 디버그 출력
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

# src/ 를 path에 추가 (다른 dev-* 와 동일)
_SRC_DIR = Path(__file__).resolve().parent.parent
_DEV_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ---------- 스크립트 내부에서 기본값 할당 (필요 시 수정) ----------
DEFAULT_MAP_JSON = "scenario4_map.json"  # config/ 기준 맵 JSON 파일명. 바꿔서 사용.
# ---------------------------------------------------------------

from utils.vlm.vlm_wrapper import VLMWrapper
from utils.miscellaneous.global_variables import (
    VLM_MODEL,
    VLM_TEMPERATURE,
    VLM_MAX_TOKENS,
    VLM_THINKING_BUDGET,
    DEBUG,
    USE_GCP_KEY,
)


def _gemini_auth_kwargs() -> dict:
    """USE_GCP_KEY=True일 때 Vertex AI(GCP 키) 사용을 위한 VLMWrapper 인자 반환."""
    if not USE_GCP_KEY or not (VLM_MODEL or "").lower().startswith("gemini"):
        return {}
    return {
        "vertexai": True,
        "project_id": os.getenv("GOOGLE_CLOUD_PROJECT"),
        "location": os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
    }


def load_prompt(path: Path) -> str:
    """프롬프트 파일 내용을 UTF-8로 읽어 반환."""
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def get_image_from_minigrid(map_json_path: Path) -> np.ndarray:
    """맵 JSON으로 Minigrid 환경을 만들고, reset 후 현재 화면 이미지를 반환."""
    from minigrid import register_minigrid_envs
    from utils.map_manager.emoji_map_loader import load_emoji_map_from_json

    register_minigrid_envs()
    env = load_emoji_map_from_json(str(map_json_path))
    env.reset()
    return env.get_image()


def image_for_display(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    """VLM용 이미지(경로 또는 RGB 배열)를 OpenCV 표시용 BGR 배열로 반환."""
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise FileNotFoundError(f"Cannot load image: {image}")
        return bgr
    # numpy RGB (H, W, 3)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def run_vlm_interactive(
    image: Union[str, Path, np.ndarray],
    user_prompt: str,
    system_prompt_path: Optional[Path] = None,
    debug: bool = False,
) -> str:
    """이미지와 터미널에서 받은 user prompt로 VLM 호출 후 리턴 문자열 반환."""
    system_prompt = ""
    if system_prompt_path is not None and system_prompt_path.is_file():
        system_prompt = load_prompt(system_prompt_path)

    wrapper = VLMWrapper(
        model=VLM_MODEL,
        temperature=VLM_TEMPERATURE,
        max_tokens=VLM_MAX_TOKENS,
        thinking_budget=VLM_THINKING_BUDGET,
        **_gemini_auth_kwargs(),
    )
    return wrapper.generate(
        image=image,
        system_prompt=system_prompt,
        user_prompt=user_prompt.strip(),
        debug=debug,
    )


def run_vlm_and_save(
    prompt_path: Path,
    image: Union[str, Path, np.ndarray],
    output_path: Path,
    system_prompt_path: Optional[Path] = None,
    debug: bool = False,
) -> str:
    """
    Prompt와 이미지(경로 또는 numpy 배열)로 VLM을 호출하고, 응답을 output_path에 TXT로 저장.
    """
    prompt_text = load_prompt(prompt_path)
    system_prompt = ""
    if system_prompt_path is not None and system_prompt_path.is_file():
        system_prompt = load_prompt(system_prompt_path)

    wrapper = VLMWrapper(
        model=VLM_MODEL,
        temperature=VLM_TEMPERATURE,
        max_tokens=VLM_MAX_TOKENS,
        thinking_budget=VLM_THINKING_BUDGET,
        **_gemini_auth_kwargs(),
    )
    response = wrapper.generate(
        image=image,
        system_prompt=system_prompt,
        user_prompt=prompt_text,
        debug=debug,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(response, encoding="utf-8")
    return response


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prompt + (Minigrid 환경 또는 맵 이미지)로 VLM을 실행하고 출력을 TXT로 저장합니다."
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="비인터랙티브: 프롬프트 파일(-p)을 user prompt로 사용, 출력 TXT 저장 (기본은 인터랙티브)",
    )
    parser.add_argument(
        "--prompt", "-p",
        type=Path,
        default=_DEV_DIR / "prompt.txt",
        help="(배치 모드) user 프롬프트 텍스트 파일 경로 (기본: dev-scenario4/prompt.txt)",
    )
    parser.add_argument(
        "--instruction", "-I",
        type=Path,
        default=_DEV_DIR / "instruction.txt",
        help="(인터랙티브) user prompt로 쓸 지시 파일 (기본: dev-scenario4/instruction.txt)",
    )
    parser.add_argument(
        "--map-json",
        type=Path,
        default=None,
        help="맵 JSON 경로 (미지정 시 스크립트 내 DEFAULT_MAP_JSON 사용)",
    )
    parser.add_argument(
        "--map-image", "-m",
        type=Path,
        default=None,
        help="(대안) 맵 이미지 파일 경로. 지정 시 Minigrid 대신 이 이미지 사용",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="출력 경로 (기본: 인터랙티브는 dev-scenario4/logs/{timestamp}_{맵stem}_vlm_output.json, 배치는 _vlm_output.txt)",
    )
    parser.add_argument(
        "--system", "-s",
        type=Path,
        default=None,
        help="시스템 프롬프트 파일 경로 (선택)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="VLM 디버그 출력 (미지정 시 global_variables.DEBUG 사용)",
    )
    args = parser.parse_args()

    prompt_path = args.prompt.resolve()
    if not prompt_path.is_file():
        prompt_path = (_DEV_DIR / args.prompt).resolve()

    debug = args.debug or DEBUG

    # 이미지 소스: --map-image 우선, 없으면 Minigrid 환경
    if args.map_image is not None:
        map_image_path = args.map_image.resolve()
        if not map_image_path.is_file():
            print(f"Error: Map image not found: {map_image_path}", file=sys.stderr)
            sys.exit(1)
        image: Union[str, Path, np.ndarray] = str(map_image_path)
        output_stem = map_image_path.stem
    else:
        # Minigrid 환경 사용
        if args.map_json is not None:
            map_json_path = args.map_json.resolve()
            if not map_json_path.is_file():
                map_json_path = (_SRC_DIR / args.map_json).resolve()
        else:
            map_json_path = (_SRC_DIR / "config" / DEFAULT_MAP_JSON).resolve()
        if not map_json_path.is_file():
            print(f"Error: Map JSON not found: {map_json_path}", file=sys.stderr)
            sys.exit(1)
        try:
            image = get_image_from_minigrid(map_json_path)
        except Exception as e:
            print(f"Error: Failed to create Minigrid env from {map_json_path}: {e}", file=sys.stderr)
            sys.exit(1)
        output_stem = map_json_path.stem

    # 기본: 인터랙티브 모드 (OpenCV 창 → CLI user prompt → VLM → JSON 저장)
    if not args.batch:
        try:
            display_img = image_for_display(image)
            cv2.imshow("Map (press any key to run VLM)", display_img)
            cv2.waitKey(1)  # 창이 먼저 뜨도록 한 프레임 처리
            instruction_path = args.instruction.resolve()
            if not instruction_path.is_file():
                print(f"Error: Instruction file not found: {instruction_path}", file=sys.stderr)
                cv2.destroyAllWindows()
                sys.exit(1)
            user_prompt = load_prompt(instruction_path)
            if not user_prompt.strip():
                print("Empty instruction. Exiting.")
                cv2.destroyAllWindows()
                sys.exit(0)
            print(f"\n[Map window opened. User prompt from: {instruction_path.name}]")
            print("Calling VLM...")
            # 시스템 프롬프트: --system 지정 시 해당 파일, 미지정 시 prompt.txt
            system_path = (args.system.resolve() if args.system else _DEV_DIR / "prompt.txt")
            response = run_vlm_interactive(
                image=image,
                user_prompt=user_prompt,
                system_prompt_path=system_path,
                debug=debug,
            )
            print("\n--- VLM return ---")
            print(response)
            print("--- end ---")
            # 출력 JSON 파일 저장: dev-scenario4/logs/ 에 timestamp_맵stem_vlm_output.json
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = args.output.resolve() if args.output else (_DEV_DIR / "logs" / f"{ts}_{output_stem}_vlm_output.json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_payload = {
                "instruction_ref": user_prompt.strip(),
                "vlm_output": response,
            }
            out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nOutput saved to: {out_path}")
            print("\nPress any key in the map window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            cv2.destroyAllWindows()
            print(f"VLM or display failed: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # 배치 모드: 프롬프트 파일을 user prompt로 사용, 출력 TXT 저장
    if args.output is not None:
        output_path = args.output.resolve()
    else:
        output_path = _DEV_DIR / f"{output_stem}_vlm_output.txt"

    try:
        response = run_vlm_and_save(
            prompt_path=prompt_path,
            image=image,
            output_path=output_path,
            system_prompt_path=args.system.resolve() if args.system else None,
            debug=debug,
        )
        print(f"VLM output saved to: {output_path}")
        print(f"Response length: {len(response)} characters")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"VLM or save failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
