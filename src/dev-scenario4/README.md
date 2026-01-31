## Code structure of : vlm_prompt_map_to_txt.py

## Usage

- **기본(인터랙티브):** 맵 OpenCV 창 + `instruction.txt`를 user prompt로 VLM 호출 → 결과 출력 및 JSON 저장.
  ```bash
  python vlm_prompt_map_to_txt.py
  ```
- **배치:** 프롬프트 파일을 user prompt로 사용, 출력 TXT 저장.
  ```bash
  python vlm_prompt_map_to_txt.py --batch -p user_prompt.txt
  ```
- **맵 지정:** `vlm_prompt_map_to_txt.py` 상단 `DEFAULT_MAP_JSON` 수정 또는 `--map-json config/맵이름.json` 사용.
- **출력 경로:** `-o 경로` 로 지정. 미지정 시 인터랙티브는 `{맵stem}_vlm_output.json`, 배치는 `{맵stem}_vlm_output.txt`.
