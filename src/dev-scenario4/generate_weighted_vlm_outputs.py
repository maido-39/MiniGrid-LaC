"""
Generate scenario4_map_vlm_output.json files with varying weight ratios.

각 비율 조합에 대해 vlm_prompt_map_to_txt.py 를 배치로 실행하여 VLM을 호출하고,
instruction_ref + vlm_output 을 담은 grounding JSON 파일을 생성합니다.

Weights: (time_efficiency, Physical_Safety, Social_Compliance) in 10-unit steps, sum=100.
Filenames: {a,b,c}_scenario4_map_vlm_output.json (a,b,c in tens: 8,1,1 ... 1,6,3).

주의: WEIGHT_SEQUENCE 개수만큼 VLM API 호출이 발생하므로 비용·시간이 소요됩니다.
"""

import json
import subprocess
import tempfile
from pathlib import Path

DEV_DIR = Path(__file__).resolve().parent
SRC_DIR = DEV_DIR.parent
INSTRUCTION_TXT = DEV_DIR / "instruction.txt"
PROMPT_TXT = DEV_DIR / "prompt.txt"  # 시스템 프롬프트 (vlm_prompt_map_to_txt.py 직접 실행 시와 동일)
VLM_SCRIPT = DEV_DIR / "vlm_prompt_map_to_txt.py"
MAP_JSON = SRC_DIR / "config" / "scenario4_map.json"

# (time_efficiency, Physical_Safety, Social_Compliance) in 10-unit steps, sum=100
WEIGHT_SEQUENCE = [
    (80, 10, 10),
    (70, 20, 10),
    (60, 30, 10),
    (50, 30, 20),
    (40, 40, 20),
    (30, 50, 20),
    (20, 50, 30),
    (10, 60, 30),
    (10, 80, 10),
    (10, 10, 80),
]


def _instruction_body_from_file() -> str:
    instruction_lines = []
    with open(INSTRUCTION_TXT, "r", encoding="utf-8") as f:
        in_instruction = False
        for line in f:
            if line.strip() == "[Instruction]":
                in_instruction = True
                continue
            if in_instruction and line.strip():
                instruction_lines.append(line.rstrip())
    if instruction_lines:
        return "\n".join(instruction_lines)
    return (
        "1. gather green pants and put into brown basket\n"
        "2. gather purple t-shirt inside bedroom\n"
        "3. get water from kitchen\n"
        "4. gather soap and do laundary inside shower room"
    )


def main():
    instruction_body = _instruction_body_from_file()

    for te, ps, sc in WEIGHT_SEQUENCE:
        weights_block = (
            "{\n"
            f'  "weights": {{\n'
            f'    "time_efficiency": {te},\n'
            f'    "Physical_Safety": {ps},\n'
            f'    "Social_Compliance": {sc}\n'
            "  }\n"
            "}"
        )
        instruction_ref = (
            "[Navigation_Context]\n"
            f"{weights_block}\n\n"
            "[Instruction]\n"
            f"{instruction_body}"
        )

        # 임시 user prompt 파일로 instruction_ref 저장
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            dir=str(DEV_DIR),
            encoding="utf-8",
        ) as f:
            f.write(instruction_ref)
            temp_prompt_path = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, dir=str(DEV_DIR)) as tf:
            temp_out_path = Path(tf.name)

        try:
            # vlm_prompt_map_to_txt.py 배치 실행 (시스템 프롬프트=prompt.txt 지정, 직접 실행과 동일)
            cmd = [
                "conda", "run", "-n", "minigrid",
                "python",
                str(VLM_SCRIPT),
                "--batch",
                "--prompt", str(temp_prompt_path),
                "--system", str(PROMPT_TXT),
                "--map-json", str(MAP_JSON),
                "--output", str(temp_out_path),
            ]
            print(f"[{te},{ps},{sc}] Calling VLM (conda minigrid)...")
            result = subprocess.run(cmd, cwd=str(SRC_DIR), capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"  stderr: {result.stderr}")
                raise RuntimeError(f"vlm_prompt_map_to_txt.py failed with code {result.returncode}")

            vlm_output = temp_out_path.read_text(encoding="utf-8").strip()
            out = {
                "instruction_ref": instruction_ref,
                "vlm_output": vlm_output,
            }
            name = f"{{{te//10},{ps//10},{sc//10}}}_scenario4_map_vlm_output.json"
            out_path = DEV_DIR / name
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  Wrote {out_path.name}")
        finally:
            temp_prompt_path.unlink(missing_ok=True)
            temp_out_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
