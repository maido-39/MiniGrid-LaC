"""
시나리오 2 실험 환경 테스트 스크립트 (VLM 제어 버전 - Spatial Logic Enhanced)
시나리오 2: 파란 기둥으로 가서 오른쪽으로 돌고, 테이블 옆에 멈추시오
"""

from minigrid import register_minigrid_envs
from custom_environment import CustomRoomWrapper
from vlm_wrapper import ChatGPT4oVLMWrapper
from vlm_postprocessor import VLMResponsePostProcessor
import numpy as np
import cv2
import json
import csv
import math
from datetime import datetime
from pathlib import Path
from PIL import Image

# MiniGrid 환경 등록
register_minigrid_envs()

# VLM 설정
VLM_MODEL = "gpt-4o"
VLM_TEMPERATURE = 0.0
VLM_MAX_TOKENS = 1000

# Mission/Task 설정
DEFAULT_MISSION = "Go to the blue pillar, turn right, then stop next to the table."

# --- [추가된 로직] 상대 위치 계산 함수 ---
def get_relative_position_hint(agent_pos, agent_dir, target_pos):
    """
    로봇의 현재 위치와 방향을 기준으로 목표물이 어디에 있는지 힌트 텍스트 생성
    """
    if target_pos is None:
        return "No specific target identified."

    ax, ay = agent_pos
    tx, ty = target_pos # 기둥의 대표 좌표

    # 글로벌 좌표계 차이
    dx = tx - ax
    dy = ty - ay
    
    direction_str = ["Right (East)", "Down (South)", "Left (West)", "Up (North)"]
    curr_dir = direction_str[agent_dir]

    # 절대 방위 계산
    ns = "South" if dy > 0 else "North" if dy < 0 else "Same Latitude"
    ew = "East" if dx > 0 else "West" if dx < 0 else "Same Longitude"
    
    # 상대 각도 계산
    target_angle = math.degrees(math.atan2(dy, dx))
    robot_angle_map = {0: 0, 1: 90, 2: 180, 3: 270}
    robot_angle = robot_angle_map[agent_dir]
    
    rel_angle = (target_angle - robot_angle) % 360
    
    # 상대 위치 설명
    relative_desc = ""
    if 315 <= rel_angle or rel_angle < 45:
        relative_desc = "IN FRONT OF you"
    elif 45 <= rel_angle < 135:
        relative_desc = "to your RIGHT"
    elif 135 <= rel_angle < 225:
        relative_desc = "BEHIND you"
    else:
        relative_desc = "to your LEFT"

    hint = f"""
    [Spatial Analysis (Calculated)]
    - Your Position: ({ax}, {ay}) facing {curr_dir}
    """
    
    # 결정적인 힌트 추가: 남쪽(Down)을 보고 있는데 목표가 북쪽(Up)에 있는 경우
    if relative_desc == "BEHIND you" and agent_dir == 1: 
        hint += "\n    - WARNING: You are facing DOWN, but target is UP (North). DO NOT move forward. You must TURN AROUND first."
        
    return hint


class PromptOrganizer:
    """프롬프트 관리 클래스"""
    
    def __init__(self):
        self.grounding = ""
        self.previous_action = ""
        self.task_process = {"goal": "", "status": ""}
    
    def get_system_prompt(self, wrapper=None, spatial_hint="") -> str:
        """전체 System Prompt 생성"""
        heading_info = ""
        if wrapper is not None:
            heading = wrapper.get_heading()
            heading_short = wrapper.get_heading_short()
            heading_info = f"{heading} ({heading_short})"
        else:
            heading_info = "provided by the environment"
        
        grounding_content = self.grounding if self.grounding else ""
        previous_action = self.previous_action if self.previous_action else "None"
        task_goal = self.task_process.get("goal", "") if self.task_process.get("goal") else "None"
        task_status = self.task_process.get("status", "") if self.task_process.get("status") else "None"
        task_process_str = f"Goal: {task_goal}, Status: {task_status}"
        
        return f"""You are a robot operating in a grid-based environment.

## Coordinate System & Orientation (CRITICAL)
- The grid origin (0,0) is at the TOP-LEFT corner.
- **Y-axis increases DOWNWARD.** (Moving DOWN increases Y, Moving UP decreases Y).
- **Direction:**
  - 0: Right (East, +X)
  - 1: Down (South, +Y)
  - 2: Left (West, -X)
  - 3: Up (North, -Y)

## Robot State
- Current Heading: {heading_info}
- This heading is ground-truth.

## Spatial Reality Check
{spatial_hint}

## Action Rules
- "move forward": Moves 1 step in your CURRENT facing direction.
- "turn left/right": Rotates 90 degrees.
- **CRITICAL**: If the Spatial Reality Check says the target is BEHIND you, you MUST turn before moving.

## Completion Criteria (Definition of Done)
- **1-Cell Radius Rule**: You are considered "at" the target if you are **ADJACENT** (1 cell away) to it (north, south, east, west).
- when any cell of the target is 1-cell around the robot, you are considered "at" the target.
- **Do NOT Collide**: Do not try to move *into* the object. Being next to it is sufficient.
- **Auto-Transition**: If you are adjacent to the current goal object (e.g., Blue Pillar), immediately mark the current subtask as "completed" in memory and set the "goal" to the NEXT step of the mission (e.g., "turn right").

## Task Process Semantics
- task_process is a record of task state.
- Subtasks with status "completed" must NOT be used as action targets.

## Grounding Knowledge
{grounding_content}

## Memory
- Previous Action: {previous_action}
- Task Process: {task_process_str}

## Response Format (Strict JSON)
Respond in valid JSON:
```json
{{
  "action": ["<action1>", "<action2>", "<action3>"],
  "reasoning": "<Step-by-step logic based on Spatial Reality Check>",
  "grounding": "<new rule if needed>",
  "memory": {{
    "spatial_description": "<relative position of objects>",
    "task_process": {{
      "goal": "<current subtask>",
      "status": "<pending|in_progress|completed>"
    }},
    "previous_action": "<action1>"
  }}
}}
Important: Exactly 3 actions in the list. Only the first is executed immediately.
"""

    def get_feedback_system_prompt(self) -> str:
        return """You are a feedback-to-knowledge converter.
Convert user feedback into a single-line behavioral heuristic (grounding rule). Response JSON: {"grounding_rule": "<rule>"} """

    def update_grounding(self, new_grounding: str):
        if new_grounding and new_grounding.strip():
            if self.grounding:
                self.grounding = f"{self.grounding}\n\n{new_grounding.strip()}"
            else:
                self.grounding = new_grounding.strip()

    def get_user_prompt(self, default_prompt: str = None) -> str:
        actual_default = default_prompt if default_prompt else DEFAULT_MISSION
        if default_prompt:
            print(f"Task Hint: {default_prompt}")
        print(f"명령을 입력하세요 (Enter: {actual_default}):")
        user_input = input("> ").strip()
        # print(user_input)
        
        if not user_input:
            return f"Mission: {DEFAULT_MISSION}"
        return user_input


class VLMProcessor: 
    def __init__(self, model: str = VLM_MODEL, temperature: float = VLM_TEMPERATURE, max_tokens: int = VLM_MAX_TOKENS): 
        self.vlm = ChatGPT4oVLMWrapper(model=model, temperature=temperature, max_tokens=max_tokens) 
        self.postprocessor_action = VLMResponsePostProcessor(required_fields=["action", "reasoning", "grounding", "memory"]) 
        self.postprocessor_feedback = VLMResponsePostProcessor(required_fields=["knowledge"])

    def requester(self, image: np.ndarray, system_prompt: str, user_prompt: str) -> str:
        try:
            return self.vlm.generate(image=image, system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as e:
            print(f"VLM API Error: {e}")
            return ""

    def parser_action(self, raw_response: str) -> dict:
        try:
            return self.postprocessor_action.process(raw_response, strict=True)
        except ValueError:
            return {"action": ["2"], "reasoning": "Parse Fail", "memory": {}}

    def parser_feedback(self, raw_response: str) -> dict:
        try:
            return self.postprocessor_feedback.process(raw_response, strict=True)
        except ValueError:
            return {"knowledge": ""}


class Visualizer:
    def __init__(self, window_name: str = "Scenario 2"):
        self.window_name = window_name

    def visualize_grid_cli(self, wrapper: CustomRoomWrapper, state: dict):
        env = wrapper.env
        size = wrapper.size
        agent_pos = state['agent_pos']
        agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        agent_dir = state['agent_dir']
        direction_symbols = {0: '> ', 1: 'v ', 2: '< ', 3: '^ '}
        agent_symbol = direction_symbols.get(agent_dir, 'A')
        
        print("\n" + "=" * 60)
        print("Current Grid State:")
        print("=" * 60)
        for y in range(size):
            row = []
            for x in range(size):
                cell = env.grid.get(x, y)
                if x == agent_x and y == agent_y:
                    row.append(agent_symbol)
                elif cell is not None and cell.type == 'wall':
                    if hasattr(cell, 'color'):
                        if cell.color == 'blue': row.append('🟦')
                        elif cell.color == 'purple': row.append('🟪')
                        else: row.append('⬛')
                    else: row.append('⬛')
                elif cell is not None and cell.type == 'goal':
                    row.append('🟩')
                else:
                    row.append('⬜️')
            print(''.join(row))
        print("=" * 60)
        print(f"Agent: ({agent_x}, {agent_y}) facing {agent_symbol}")
        print("=" * 60 + "\n")

    def display_image(self, img: np.ndarray):
        if img is not None:
            try:
                img_bgr = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
                h, w = img_bgr.shape[:2]
                scale = min(800//h, 800//w, 4)
                if scale > 1:
                    img_bgr = cv2.resize(img_bgr, (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
                cv2.imshow(self.window_name, img_bgr)
                cv2.waitKey(1)
            except Exception: pass

    def cleanup(self):
        cv2.destroyAllWindows()


class UserInteraction:
    def get_input(self, prompt: str = "> ") -> str:
        return input(prompt).strip()


# [수정됨] 클래스 외부로 분리된 환경 생성 함수
def create_scenario2_environment() -> CustomRoomWrapper:
    size = 10 
    walls = [] 
    for i in range(size): 
        walls.append((i, 0))
        walls.append((i, size-1))
        walls.append((0, i))
        walls.append((size-1, i))

    # Blue Pillar
    for pos in [(3, 4), (4, 4), (3, 5), (4, 5)]:
        walls.append((pos[0], pos[1], 'blue'))
    # Table
    for pos in [(5, 1), (6, 1), (7, 1)]:
        walls.append((pos[0], pos[1], 'purple'))
        
    return CustomRoomWrapper(size=size, room_config={
        'start_pos': (1, 8), 'goal_pos': (8, 1), 'walls': walls, 'objects': []
    })


class Scenario2Experiment:
    # [수정됨] __init__으로 수정 (언더바 2개)
    def __init__(self, log_dir: Path = None): 
        self.wrapper = None
        self.prompt_organizer = PromptOrganizer()
        self.vlm_processor = VLMProcessor()
        self.visualizer = Visualizer()

        if log_dir is None:
            log_dir = Path("logs") / f"scenario2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.step = 0
        self.done = False
        self.state = None
        self.target_obj_pos = (3, 4) # Blue Pillar 대표 좌표
        
        self.csv_file = None
        self.csv_writer = None
        self._init_csv_logging()

    def _init_csv_logging(self):
        csv_path = self.log_dir / "experiment_log.csv"
        self.csv_file = open(csv_path, 'a', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["step", "timestamp", "agent_pos", "action", "reasoning", "reward", "done"])

    def _log_step(self, action_name, reasoning, reward, done):
        self.csv_writer.writerow([
            self.step, datetime.now().isoformat(), 
            str(self.state['agent_pos']), action_name, reasoning, reward, done
        ])
        self.csv_file.flush()
        
        # 이미지 저장
        img_path = self.log_dir / f"step_{self.step:04d}.png"
        if self.wrapper and self.wrapper.get_image() is not None:
             Image.fromarray(self.wrapper.get_image()).save(img_path)

    def initialize(self):
        print("=" * 60); print("Scenario 2: VLM Control (Spatial Enhanced)"); print("=" * 60)
        self.wrapper = create_scenario2_environment()
        self.wrapper.reset()
        self.state = self.wrapper.get_state()
        print(f"Start Pos: {self.state['agent_pos']}, Dir: {self.state['agent_dir']}")

    def run_step(self):
        self.step += 1
        print(f"\n[STEP {self.step}]")
        
        self.state = self.wrapper.get_state()
        image = self.wrapper.get_image()
        self.visualizer.visualize_grid_cli(self.wrapper, self.state)
        self.visualizer.display_image(image)
        
        # 1. Spatial Hint 계산
        spatial_hint = get_relative_position_hint(
            self.state['agent_pos'], 
            self.state['agent_dir'], 
            self.target_obj_pos
        )
        
        # 2. User Input
        self.user_prompt = self.prompt_organizer.get_user_prompt(f"Mission: {DEFAULT_MISSION}")
        
        # 3. Prompt 생성 (Hint 주입)
        system_prompt = self.prompt_organizer.get_system_prompt(self.wrapper, spatial_hint)
        
        # 4. VLM 요청
        print("[VLM Requesting...]")
        parsed = self.vlm_processor.requester(image, system_prompt, self.user_prompt)
        response = self.vlm_processor.parser_action(parsed)
        
        # 5. 응답 처리
        action_chunk = response.get('action', ['2'])
        if isinstance(action_chunk, str): action_chunk = [action_chunk]
        action_str = str(action_chunk[0]) if action_chunk else '2'
        reasoning = response.get('reasoning', '')
        
        print(f"\n[AI Thought]\nReasoning: {reasoning}")
        print(f"Action: {action_str} (Plan: {action_chunk})")
        print(f"Spatial Hint Used:\n{spatial_hint}")
        
        # Memory 업데이트
        mem = response.get('memory', {})
        if isinstance(mem, dict):
            self.prompt_organizer.previous_action = mem.get('previous_action', action_str)
            self.prompt_organizer.task_process = mem.get('task_process', {})
        
        # 6. 실행
        try:
            idx = self.wrapper.parse_action(action_str)
            name = self.wrapper.ACTION_NAMES.get(idx, f"act_{idx}")
            _, reward, term, trunc, _ = self.wrapper.step(idx)
            self.done = term or trunc
            print(f"Executed: {name} | Reward: {reward} | Done: {self.done}")
            self._log_step(name, reasoning, reward, self.done)
        except Exception as e:
            print(f"Execution Error: {e}")
            if self.wrapper:
                self.wrapper.step(2)

        return True

    def run(self):
        self.initialize()
        while not self.done and self.step < 100:
            self.run_step()
        self.cleanup()

    def cleanup(self):
        self.visualizer.cleanup()
        if self.wrapper:
            self.wrapper.close()
        if self.csv_file: self.csv_file.close()
        print("\nExperiment Finished.")


def main():
    try:
        Scenario2Experiment().run()
    except KeyboardInterrupt:
        print("\nAborted by user.")

if __name__ == "__main__":
    main()