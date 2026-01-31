"""
Prompt 파일과 (맵 이미지 또는 Minigrid 환경)을 입력받아 VLM 추론 후 출력을 TXT로 저장하는 스크립트.

기본: 이 폴더(dev-scenario4)의 prompt.txt 를 사용하고, Minigrid 환경을 JSON 맵에서 로드해
      현재 화면(env.get_image())을 VLM에 넣습니다.
대안: --map-image 로 이미지 파일을 직접 줄 수 있습니다.

--interactive (-i): 맵을 OpenCV 창으로 띄우고, 터미널에서 user prompt를 입력받아 VLM 호출 후
                    리턴값을 터미널에 출력합니다. (파일 저장 없음)

utils/ 아래 코드는 수정하지 않고, VLMWrapper·load_emoji_map_from_json 등 기존 유틸만 사용합니다.

사용법:
    # 기본: dev-scenario4/prompt.txt + Minigrid 환경(global MAP_FILE_NAME 맵)
    python vlm_prompt_map_to_txt.py

    # 인터랙티브: 맵 OpenCV로 띄우고, 터미널에서 prompt 입력 → VLM 결과 출력
    python vlm_prompt_map_to_txt.py --interactive

    # 맵 JSON 지정
    python vlm_prompt_map_to_txt.py --map-json config/scenario4_example_map.json

    # 대안: 맵 이미지 파일 직접 지정
    python vlm_prompt_map_to_txt.py --map-image path/to/map.png

옵션:
    --interactive, -i  맵 OpenCV 표시 + 터미널에서 user prompt 입력, VLM 리턴값 출력
    --prompt, -p   프롬프트 텍스트 파일 (기본: 이 폴더의 prompt.txt)
    --map-json     맵 JSON 경로 (Minigrid 환경 사용 시, 기본: config/MAP_FILE_NAME)
    --map-image, -m  (대안) 맵 이미지 파일 경로. 지정하면 Minigrid 대신 이 이미지 사용
    --output, -o  출력 TXT 경로 (기본: dev-scenario4/vlm_output.txt 또는 맵 기준)
    --system, -s  (선택) 시스템 프롬프트 파일
    --debug       VLM 디버그 출력
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

# src/ 를 path에 추가 (다른 dev-* 와 동일)
_SRC_DIR = Path(__file__).resolve().parent.parent
_DEV_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.vlm.vlm_wrapper import VLMWrapper
from utils.miscellaneous.global_variables import (
    VLM_MODEL,
    VLM_TEMPERATURE,
    VLM_MAX_TOKENS,
    VLM_THINKING_BUDGET,
    DEBUG,
    MAP_FILE_NAME,
)


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
        "--interactive", "-i",
        action="store_true",
        help="맵 OpenCV 표시 + 터미널에서 user prompt 입력, VLM 리턴값 출력",
    )
    parser.add_argument(
        "--prompt", "-p",
        type=Path,
        default=_DEV_DIR / "prompt.txt",
        help="프롬프트 텍스트 파일 경로 (기본: dev-scenario4/prompt.txt)",
    )
    parser.add_argument(
        "--map-json",
        type=Path,
        default=None,
        help="맵 JSON 경로 (Minigrid 사용 시, 기본: config/MAP_FILE_NAME)",
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
        help="출력 TXT 경로 (기본: dev-scenario4/vlm_output.txt 또는 맵 기준)",
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
            map_json_path = (_SRC_DIR / "config" / MAP_FILE_NAME).resolve()
        if not map_json_path.is_file():
            print(f"Error: Map JSON not found: {map_json_path}", file=sys.stderr)
            sys.exit(1)
        try:
            image = get_image_from_minigrid(map_json_path)
        except Exception as e:
            print(f"Error: Failed to create Minigrid env from {map_json_path}: {e}", file=sys.stderr)
            sys.exit(1)
        output_stem = map_json_path.stem

    # 인터랙티브 모드: OpenCV로 맵 띄우고, 터미널에서 prompt 입력 → VLM 결과 출력
    if args.interactive:
        try:
            display_img = image_for_display(image)
            cv2.imshow("Map (press any key after entering prompt)", display_img)
            print("\n[Map window opened. Enter your user prompt below, then press Enter.]")
            user_prompt = input("User prompt: ")
            if not user_prompt.strip():
                print("Empty prompt. Exiting.")
                cv2.destroyAllWindows()
                sys.exit(0)
            print("\nCalling VLM...")
            response = run_vlm_interactive(
                image=image,
                user_prompt=user_prompt,
                system_prompt_path=args.system.resolve() if args.system else None,
                debug=debug,
            )
            print("\n--- VLM return ---")
            print(response)
            print("--- end ---")
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
