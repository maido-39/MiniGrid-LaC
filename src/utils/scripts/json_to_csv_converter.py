######################################################
#                                                    #
#                  JSON TO CSV                      #
#                   CONVERTER                       #
#                                                    #
######################################################


""""""




######################################################
#                                                    #
#                      LIBRARIES                     #
#                                                    #
######################################################


import json
import csv
from pathlib import Path
from typing import Optional, Dict, List, Any
import argparse


######################################################
#                                                    #
#                    FUNCTIONS                       #
#                                                    #
######################################################


def convert_episode_json_to_csv(episode_json_path: Path, output_csv_path: Optional[Path] = None):
    """
    Episode JSON 파일을 CSV로 변환
    
    Args:
        episode_json_path: Episode JSON 파일 경로
        output_csv_path: 출력 CSV 파일 경로 (None이면 자동 생성)
    """
    # JSON 파일 읽기
    with open(episode_json_path, 'r', encoding='utf-8') as f:
        episode_data = json.load(f)
    
    # CSV 출력 경로 결정
    if output_csv_path is None:
        output_csv_path = episode_json_path.parent / f"{episode_json_path.stem}.csv"
    
    # CSV 헤더 정의
    headers = [
        "step_id",
        "instruction",
        "status",
        "action_index",
        "action_name",
        "agent_pos_x",
        "agent_pos_y",
        "agent_dir",
        "feedback_user_preference",
        "feedback_spatial",
        "feedback_procedural",
        "feedback_general",
        "image_path"
    ]
    
    # CSV 데이터 생성
    rows = []
    for step in episode_data.get("steps", []):
        row = [
            step.get("step_id", ""),
            step.get("instruction", ""),
            step.get("status", ""),
            step.get("action", {}).get("index", ""),
            step.get("action", {}).get("name", ""),
            step.get("state", {}).get("agent_pos", [None, None])[0],
            step.get("state", {}).get("agent_pos", [None, None])[1],
            step.get("state", {}).get("agent_dir", ""),
            step.get("feedback", {}).get("user_preference", ""),
            step.get("feedback", {}).get("spatial", ""),
            step.get("feedback", {}).get("procedural", ""),
            step.get("feedback", {}).get("general", ""),
            step.get("image_path", "")
        ]
        rows.append(row)
    
    # CSV 파일 쓰기
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    
    print(f"CSV 파일이 생성되었습니다: {output_csv_path}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Episode JSON을 CSV로 변환")
    parser.add_argument("json_file", type=str, help="Episode JSON 파일 경로")
    parser.add_argument("-o", "--output", type=str, default=None, help="출력 CSV 파일 경로 (선택)")
    
    args = parser.parse_args()
    
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"오류: 파일을 찾을 수 없습니다: {json_path}")
        return
    
    output_path = Path(args.output) if args.output else None
    convert_episode_json_to_csv(json_path, output_path)


if __name__ == "__main__":
    main()
