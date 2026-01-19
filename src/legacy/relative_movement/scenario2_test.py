"""
시나리오 2 실험 환경 테스트 스크립트 (VLM 제어 버전 - 클래스 기반)

시나리오 2: 파란 기둥으로 가서 오른쪽으로 돌고, 테이블 옆에 멈추시오

환경 구성:
- 벽: 검은색 (외벽)
- 파란 기둥: 파란색 2x2 Grid (통과불가, 색상이 있는 벽)
- 테이블: 보라색 1x3 Grid (통과불가, 색상이 있는 벽)
- 시작점: (1, 8)
- 종료점: (8, 1)

레이아웃 (10x10):
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬜️⬜️⬜️⬜️🟪🟪🟪🟩⬛
⬛⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬛
⬛⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬛
⬛⬜️⬜️🟦🟦⬜️⬜️⬜️⬜️⬛ 
⬛⬜️⬜️🟦🟦⬜️⬜️⬜️⬜️⬛
⬛⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬛
⬛⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬛
⬛⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛

사용법:
    python scenario2_test.py
"""

from minigrid import register_minigrid_envs
# Actual path: legacy.relative_movement.custom_environment
from legacy import CustomRoomWrapper
# Actual paths: utils.vlm.vlm_wrapper, utils.vlm.vlm_postprocessor
from utils import ChatGPT4oVLMWrapper, VLMResponsePostProcessor
import numpy as np
import cv2
import json
import csv
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


class PromptOrganizer:
    """프롬프트 관리 클래스"""
    
    def __init__(self):
        self.grounding = ""
        self.previous_action = ""
        self.task_process = {"goal": "", "status": ""}  # status: pending | in_progress | completed | blocked
    
    def get_system_prompt(self, wrapper=None) -> str:
        """전체 System Prompt 생성"""
        # Heading 정보 가져오기 (변수화 필요한 부분만)
        heading_info = ""
        if wrapper is not None:
            heading = wrapper.get_heading()
            heading_short = wrapper.get_heading_short()
            heading_info = f"{heading} ({heading_short})"
        else:
            heading_info = "provided by the environment"
        
        ## Prompt 오류 핸들링용임
        # Grounding 내용 (항상 표시, 비어있으면 빈 문자열)
        grounding_content = self.grounding if self.grounding else ""
        
        # Previous Action (항상 표시, 비어있으면 "None")
        previous_action = self.previous_action if self.previous_action else "None"
        
        # Task Process (항상 표시, 비어있으면 기본값)
        task_goal = self.task_process.get("goal", "") if self.task_process.get("goal") else "None"
        task_status = self.task_process.get("status", "") if self.task_process.get("status") else "None"
        task_process_str = f"Goal: {task_goal}, Status: {task_status}"
        
        
        ## 실제 적용 Prompt 시작
        return f"""You are a robot operating in a grid-based environment.

## Robot State (Authoritative)
- The robot's current heading is {heading_info}.
- Heading indicates the robot's forward-facing direction.
- This heading is ground-truth and MUST be used as-is.
- Do NOT infer or reinterpret the robot's heading from the image.

## Coordinate Convention
- Top of the image: North
- Bottom of the image: South
- Left of the image: West
- Right of the image: East

## Environment
Grid world with:
- Walls (black, impassable)
- Blue pillar (impassable)
- Purple table (impassable)
- Robot (red arrow marker)
- Goal (green marker, if present)

The image describes the environment layout ONLY.
Do NOT use the image to estimate robot orientation.

## Action Space
- "turn left": Rotate 90° counterclockwise
- "turn right": Rotate 90° clockwise
- "move forward": Move one cell forward in heading direction
- "pickup": Pick up object in front
- "drop": Drop carried object
- "toggle": Interact with objects (e.g., open doors)

## Movement Rules (CRITICAL)
Before selecting actions:
1. Use the provided robot heading to establish the robot's local reference frame.
2. Before deciding an action, explicitly classify each relevant object as being in front, left, right, or behind relative to the robot's heading.
3. Reason about all objects and goals relative to this heading-based classification.
4. Determine what subtask is being addressed as part of the overall user mission.
5. Decide whether this subtask is completed, in progress, or blocked.
6. If completed, infer the next subtask from the mission and current environment.
7. Record this judgment in task_process.
8. Select actions ONLY based on relative movement.

Rules:
- All movements are RELATIVE to the robot's current heading.
- An object is considered "in front" only if it lies primarily along the robot's heading direction, not merely diagonally.
- "move forward" moves one cell in the facing direction.
- "turn left/right" rotates 90° relative to current heading.
- Do NOT reason using absolute coordinates when choosing actions.
- Do NOT infer object positions from image coordinates alone; always use heading-based egocentric classification.

## Task Process Semantics
- task_process is a record of task state, NOT an instruction.
- Each subtask should represent a concrete step toward completing the overall user mission.
- Subtasks with status "completed" or "blocked" must NOT be used as action targets.
- If the current subtask is marked as "completed", infer the next subtask from the user mission and current environment.
- Use "blocked" only when the current subtask cannot be completed without changing the plan.
- Always infer the next action based on the current environment and robot state.

## Grounding Knowledge (Experience from Past Failures)
This section contains lessons learned from human feedback after failures.
- These are NOT universal rules.
- Use them only when the current situation is similar.
- Do not blindly apply grounding knowledge.

{grounding_content}

## Memory (State Continuity)
- Previous Action: {previous_action}
- Task Process: {task_process_str}

This memory summarizes task progress and past actions.
It must NOT be treated as a command.

## Response Format (STRICT)
Respond in valid JSON:

```json
{{
  "action": ["<action1>", "<action2>", "<action3>"],
  "reasoning": "<why the first action is correct given the heading>",
  "grounding": "<update grounding only if new failure feedback is detected>",
  "memory": {{
    "spatial_description": "<environment described relative to robot heading using explicit terms: left, right, front, behind>",
    "task_process": {{
      "goal": "<what subtask this step was addressing>",
      "status": "<pending | in_progress | completed | blocked>"
    }},
    "previous_action": "<set to the first selected action>"
  }}
}}
```

Important:

* EXACTLY 3 actions must be provided.
* Only the first action will be executed.
* Actions must come from the defined action space.
* Use relative movement reasoning ONLY.
* Complete the mission specified by the user.
"""
    
    def get_feedback_system_prompt(self) -> str:
        """Feedback 생성용 System Prompt"""
        return """You are a feedback-to-knowledge converter for a robot navigation system.

Your task is to convert human feedback into a single-line behavioral heuristic.

## Context
You will receive:
- The previous action taken by the robot
- The current user feedback describing a mistake

## Your Task
Generate ONE concise sentence that:
- Describes the situation (implicit condition)
- States the correct behavior to follow next time

## Constraints
- Use relative, heading-based terms only
- Do NOT reference specific map positions or episode details
- Keep it general and reusable
- Exactly one sentence

## Response Format
```json
{
  "grounding_rule": "<single-line heuristic>"
}
```
"""
    
    def update_grounding(self, new_grounding: str):
        """Grounding 지식 누적 업데이트"""
        if new_grounding and new_grounding.strip():
            if self.grounding:
                self.grounding = f"{self.grounding}\n\n{new_grounding.strip()}"
            else:
                self.grounding = new_grounding.strip()
    
    def get_user_prompt(self, default_prompt: str = None) -> str:
        """사용자 프롬프트 입력 받기"""
        # 기본 프롬프트 결정 (default_prompt가 없으면 DEFAULT_MISSION 사용)
        actual_default = default_prompt if default_prompt else DEFAULT_MISSION
        
        if default_prompt:
            print(f"Task Hint: {default_prompt}")
        print(f"Enter command (Enter: {actual_default}):")
        user_input = input("> ").strip()
        
        if not user_input:
            if default_prompt:
                return f"Task: {default_prompt}\n\nBased on the current image, choose the next action to complete this task."
            return f"Based on the current image, choose the next action to complete the mission: {DEFAULT_MISSION}"
        
        return user_input


class VLMProcessor:
    """VLM 요청 및 파싱 처리 클래스"""
    
    def __init__(self, model: str = VLM_MODEL, temperature: float = VLM_TEMPERATURE, max_tokens: int = VLM_MAX_TOKENS):
        self.vlm = ChatGPT4oVLMWrapper(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.postprocessor_action = VLMResponsePostProcessor(required_fields=["action", "reasoning", "grounding", "memory"])
        self.postprocessor_feedback = VLMResponsePostProcessor(required_fields=["knowledge"])
    
    def requester(self, image: np.ndarray, system_prompt: str, user_prompt: str) -> str:
        """VLM에 요청 전송 (기본 메서드)"""
        try:
            response = self.vlm.generate(
                image=image,
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            return response
        except Exception as e:
            print(f"VLM API call failed: {e}")
            return ""
    
    def parser_action(self, raw_response: str) -> dict:
        """Action 생성 응답 파싱"""
        try:
            parsed = self.postprocessor_action.process(raw_response, strict=True)
            return parsed
        except ValueError as e:
            print(f"Response parsing failed: {e}")
            return {
                "action": ["2"],
                "reasoning": "Parsing failed",
                "grounding": "",
                "memory": {
                    "spatial_description": "",
                    "task_process": {"goal": "", "status": ""},
                    "previous_action": ""
                }
            }
    
    def parser_feedback(self, raw_response: str) -> dict:
        """Feedback 생성 응답 파싱"""
        try:
            parsed = self.postprocessor_feedback.process(raw_response, strict=True)
            return parsed
        except ValueError as e:
            print(f"Feedback response parsing failed: {e}")
            return {"knowledge": ""}


class Visualizer:
    """시각화 클래스"""
    
    def __init__(self, window_name: str = "Scenario 2: VLM Control"):
        self.window_name = window_name
    
    def visualize_grid_cli(self, wrapper: CustomRoomWrapper, state: dict):
        """CLI에서 그리드를 텍스트로 시각화"""
        env = wrapper.env
        size = wrapper.size
        
        agent_pos = state['agent_pos']
        if isinstance(agent_pos, np.ndarray):
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        else:
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        
        agent_dir = state['agent_dir']
        direction_symbols = {0: '>', 1: 'v', 2: '<', 3: '^'}
        agent_symbol = direction_symbols.get(agent_dir, 'A')
        
        grid_chars = []
        for y in range(size):
            row = []
            for x in range(size):
                cell = env.grid.get(x, y)
                
                if x == agent_x and y == agent_y:
                    row.append(agent_symbol)
                elif cell is not None and cell.type == 'wall':
                    if hasattr(cell, 'color'):
                        if cell.color == 'blue':
                            row.append('🟦')
                        elif cell.color == 'purple':
                            row.append('🟪')
                        elif cell.color == 'red':
                            row.append('🟥')
                        elif cell.color == 'green':
                            row.append('🟩')
                        elif cell.color == 'yellow':
                            row.append('🟨')
                        else:
                            row.append('⬛')
                    else:
                        row.append('⬛')
                elif cell is not None and cell.type == 'goal':
                    row.append('🟩')
                elif cell is not None:
                    if hasattr(cell, 'color'):
                        if cell.color == 'blue':
                            row.append('🟦')
                        elif cell.color == 'purple':
                            row.append('🟪')
                        else:
                            row.append('🟨')
                    else:
                        row.append('🟨')
                else:
                    row.append('⬜️')
            grid_chars.append(row)
        
        print("\n" + "=" * 60)
        print("Current Grid State:")
        print("=" * 60)
        for y in range(size):
            print(''.join(grid_chars[y]))
        print("=" * 60)
        print(f"Agent Position: ({agent_x}, {agent_y}), Direction: {agent_dir} ({agent_symbol})")
        print("=" * 60 + "\n")

    def display_image(self, img: np.ndarray):
        """OpenCV를 사용하여 이미지 표시"""
        if img is not None:
            try:
                img_bgr = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
                
                height, width = img_bgr.shape[:2]
                max_size = 800
                if height < max_size and width < max_size:
                    scale = min(max_size // height, max_size // width, 4)
                    if scale > 1:
                        new_width = width * scale
                        new_height = height * scale
                        img_bgr = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                
                cv2.imshow(self.window_name, img_bgr)
                cv2.waitKey(1)
            except Exception as e:
                print(f"Image display error: {e}")
    
    def cleanup(self):
        """리소스 정리"""
        cv2.destroyAllWindows()


class UserInteraction:
    """사용자 상호작용 클래스"""
    
    def get_input(self, prompt: str = "> ") -> str:
        """사용자 입력 받기"""
        return input(prompt).strip()


def create_scenario2_environment() -> CustomRoomWrapper:
    """시나리오 2 환경 생성"""
    size = 10
    
    walls = []
    for i in range(size):
        walls.append((i, 0))
        walls.append((i, size-1))
        walls.append((0, i))
        walls.append((size-1, i))
    
    blue_pillar_positions = [(3, 4), (4, 4), (3, 5), (4, 5)]
    for pos in blue_pillar_positions:
        walls.append((pos[0], pos[1], 'blue'))
    
    table_positions = [(5, 1), (6, 1), (7, 1)]
    for pos in table_positions:
        walls.append((pos[0], pos[1], 'purple'))
    
    start_pos = (1, 8)
    goal_pos = (8, 1)
    
    room_config = {
        'start_pos': start_pos,
        'goal_pos': goal_pos,
        'walls': walls,
        'objects': []
    }
    
    return CustomRoomWrapper(size=size, room_config=room_config)


class Scenario2Experiment:
    """시나리오 2 실험 메인 클래스 (Runner)"""
    
    def __init__(self, log_dir: Path = None):
        self.wrapper = None
        self.prompt_organizer = PromptOrganizer()
        self.vlm_processor = VLMProcessor()
        self.visualizer = Visualizer()
        self.user_interaction = UserInteraction()
        
        if log_dir is None:
            log_dir = Path("logs") / f"scenario2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.step = 0
        self.done = False
        self.state = None
        self.image = None
        self.user_prompt = ""
        self.vlm_response_raw = ""
        self.vlm_response_parsed = {}
        self.action_index = 2
        self.action_name = "move forward"
        self.reward = 0.0
        
        self.csv_file = None
        self.csv_writer = None
        self._init_csv_logging()
    
    def _evaluate_feedback(self, user_prompt: str) -> bool:
        """피드백 평가 (내부 메서드)"""
        feedback_keywords = [
            "wrong", "incorrect", "that's wrong", "no", "not that", "don't", 
            "shouldn't", "error", "mistake", "why did you", "why didn't you",
            "what are you doing", "where are you going", "not feasible", 
            "cannot", "should not",
            "feedback :"
        ]
        
        user_lower = user_prompt.lower()
        for keyword in feedback_keywords:
            if keyword in user_lower:
                return True
        
        return False
    
    def vlm_gen_action(self, image: np.ndarray, system_prompt: str, user_prompt: str) -> dict:
        """Action 생성용 VLM 호출"""
        print("\n[3] Sending action generation request to VLM...")
        raw_response = self.vlm_processor.requester(
            image=image,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        if not raw_response:
            print("VLM response is empty.")
            return {}
        
        print("VLM response received")
        print("[4] Parsing response...")
        parsed = self.vlm_processor.parser_action(raw_response)
        return parsed
    
    def vlm_gen_feedback(self, system_prompt: str, user_feedback: str) -> str:
        """Feedback 생성용 VLM 호출"""
        print("\n[3-F] Sending feedback analysis request to VLM...")
        
        feedback_system_prompt = self.prompt_organizer.get_feedback_system_prompt()
        
        feedback_user_prompt = f"""## System Prompt Used
{system_prompt}

## User Feedback
feedback : {user_feedback}

Please analyze the feedback and generate concise knowledge to improve future actions.
"""
        
        raw_response = self.vlm_processor.requester(
            image=None,
            system_prompt=feedback_system_prompt,
            user_prompt=feedback_user_prompt
        )
        
        if not raw_response:
            print("Feedback VLM response is empty.")
            return ""
        
        print("Feedback VLM response received")
        print("[4-F] Parsing feedback response...")
        parsed = self.vlm_processor.parser_feedback(raw_response)
        knowledge = parsed.get('knowledge', '')
        
        if knowledge:
            print(f"\n[4-F-1] Generated Knowledge: {knowledge}")
            self.prompt_organizer.update_grounding(knowledge)
            print("\n[4-F-2] Grounding update completed")
            print("=" * 80)
            print("Updated Grounding content:")
            print("-" * 80)
            print(knowledge)
            print("-" * 80)
            print("\nFull Grounding information:")
            print("=" * 80)
            if self.prompt_organizer.grounding:
                print(self.prompt_organizer.grounding)
            else:
                print("(None)")
            print("=" * 80)
        
        return knowledge
    
    def _init_csv_logging(self):
        """CSV 로깅 초기화"""
        csv_path = self.log_dir / "experiment_log.csv"
        file_exists = csv_path.exists()
        
        self.csv_file = open(csv_path, 'a', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        
        if not file_exists:
            self.csv_writer.writerow([
                "step", "timestamp", "agent_x", "agent_y", "agent_dir",
                "action_index", "action_name", "user_prompt",
                "vlm_action_chunk", "vlm_reasoning", "vlm_grounding",
                "memory_spatial_description", "memory_task_goal", "memory_task_status", "memory_previous_action",
                "reward", "done", "image_path"
            ])
    
    def _log_step(self):
        """현재 스텝 로깅"""
        timestamp = datetime.now().isoformat()
        
        agent_pos = self.state['agent_pos']
        if isinstance(agent_pos, np.ndarray):
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        else:
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        
        image_path = f"step_{self.step:04d}.png"
        
        # Memory 파싱
        memory = self.vlm_response_parsed.get('memory', {})
        if isinstance(memory, str):
            try:
                memory = json.loads(memory)
            except Exception:
                memory = {"spatial_description": "", "task_process": {"goal": "", "status": ""}, "previous_action": ""}
        elif not isinstance(memory, dict):
            memory = {"spatial_description": "", "task_process": {"goal": "", "status": ""}, "previous_action": ""}
        
        # task_process 파싱
        task_process = memory.get('task_process', {})
        if not isinstance(task_process, dict):
            task_process = {"goal": "", "status": ""}
        
        action_chunk = self.vlm_response_parsed.get('action', [])
        if isinstance(action_chunk, str):
            try:
                action_chunk = json.loads(action_chunk)
            except Exception:
                action_chunk = [action_chunk] if action_chunk else []
        if not isinstance(action_chunk, list):
            action_chunk = [str(action_chunk)]
        
        self.csv_writer.writerow([
            self.step,
            timestamp,
            agent_x,
            agent_y,
            int(self.state['agent_dir']),
            self.action_index,
            self.action_name,
            self.user_prompt,
            json.dumps(action_chunk, ensure_ascii=False),
            self.vlm_response_parsed.get('reasoning', ''),
            self.vlm_response_parsed.get('grounding', ''),
            memory.get('spatial_description', ''),
            task_process.get('goal', ''),
            task_process.get('status', ''),
            memory.get('previous_action', ''),
            float(self.reward),
            bool(self.done),
            image_path
        ])
        self.csv_file.flush()
        
        json_path = self.log_dir / "experiment_log.json"
        json_data = {
            "step": self.step,
            "timestamp": timestamp,
            "state": {
                "agent_pos": [agent_x, agent_y],
                "agent_dir": int(self.state['agent_dir']),
                "mission": str(self.state.get('mission', ''))
            },
            "action": {
                "index": self.action_index,
                "name": self.action_name
            },
            "user_prompt": self.user_prompt,
            "vlm_response": self.vlm_response_parsed,
            "memory": memory,
            "grounding": self.prompt_organizer.grounding,
            "reward": float(self.reward),
            "done": bool(self.done),
            "image_path": image_path
        }
        
        all_data = []
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    all_data = json.load(f)
                    if not isinstance(all_data, list):
                        all_data = [all_data]
                except json.JSONDecodeError:
                    all_data = []
        
        all_data.append(json_data)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        image_path_full = self.log_dir / image_path
        img_pil = Image.fromarray(self.image)
        img_pil.save(image_path_full)
    
    def initialize(self):
        """실험 초기화"""
        print("=" * 60)
        print("Scenario 2: VLM Control Experiment")
        print("=" * 60)
        print("\nEnvironment Configuration:")
        print("  - Blue Pillar: 2x2 Grid (colored wall)")
        print("  - Table: Purple 1x3 Grid (colored wall)")
        print("  - Start Point: (1, 8)")
        print("  - End Point: (8, 1)")
        print(f"\nMission: {DEFAULT_MISSION}")
        print(f"\nLog directory: {self.log_dir}")
        
        print("\n[1] Creating environment...")
        self.wrapper = create_scenario2_environment()
        self.wrapper.reset()
        
        self.state = self.wrapper.get_state()
        print(f"Agent start position: {self.state['agent_pos']}")
        print(f"Agent direction: {self.state['agent_dir']}")
        
        print("\n[2] VLM initialization completed")
        print("\n" + "=" * 60)
        print("Experiment Started")
        print("=" * 60)
    
    def run_step(self):
        """한 스텝 실행"""
        self.step += 1
        print("\n" + "=" * 80)
        print(f"STEP {self.step}")
        print("=" * 80)
        
        self.image = self.wrapper.get_image()
        self.state = self.wrapper.get_state()
        heading = self.wrapper.get_heading()
        heading_desc = self.wrapper.get_heading_description()
        print(f"Position: {self.state['agent_pos']}, Direction: {self.state['agent_dir']} ({heading})")
        print(f"Current Heading: {heading_desc}")
        
        self.visualizer.visualize_grid_cli(self.wrapper, self.state)
        self.visualizer.display_image(self.image)
        
        default_prompt = f"Mission: {DEFAULT_MISSION}"
        self.user_prompt = self.prompt_organizer.get_user_prompt(default_prompt)
        
        # Feedback 평가
        has_feedback = self._evaluate_feedback(self.user_prompt)
        
        if has_feedback:
            # Feedback 처리: "feedback : "으로 시작하는 경우
            if self.user_prompt.lower().startswith("feedback :"):
                feedback_text = self.user_prompt[10:].strip()  # "feedback : " 제거
            else:
                feedback_text = self.user_prompt
            
            # Feedback 생성 VLM 호출
            system_prompt = self.prompt_organizer.get_system_prompt(self.wrapper)
            self.vlm_gen_feedback(system_prompt, feedback_text)
            
            # Feedback 처리 후 일반 action 생성으로 진행하지 않고 스킵
            print("\n[4-1] 피드백 처리 완료. 다음 스텝으로 진행합니다.")
            return True
        
        # 일반 Action 생성
        system_prompt = self.prompt_organizer.get_system_prompt(self.wrapper)
        self.vlm_response_parsed = self.vlm_gen_action(
            image=self.image,
            system_prompt=system_prompt,
            user_prompt=self.user_prompt
        )
        
        if not self.vlm_response_parsed:
            return False
        
        # Action chunk에서 첫 번째 액션만 추출
        action_chunk = self.vlm_response_parsed.get('action', [])
        if isinstance(action_chunk, str):
            try:
                action_chunk = json.loads(action_chunk)
            except Exception:
                action_chunk = [action_chunk] if action_chunk else []
        if not isinstance(action_chunk, list):
            action_chunk = [str(action_chunk)]
        
        if len(action_chunk) == 0:
            action_str = '2'
        else:
            action_str = str(action_chunk[0])
        
        # Memory 파싱
        memory = self.vlm_response_parsed.get('memory', {})
        if isinstance(memory, str):
            try:
                memory = json.loads(memory)
            except Exception:
                memory = {}
        if not isinstance(memory, dict):
            memory = {}
        
        # task_process 파싱
        task_process = memory.get('task_process', {})
        if not isinstance(task_process, dict):
            task_process = {"goal": "", "status": ""}
        
        # Memory 업데이트
        if isinstance(memory, dict):
            self.prompt_organizer.previous_action = memory.get('previous_action', action_str)
            self.prompt_organizer.task_process = {
                "goal": task_process.get('goal', ''),
                "status": task_process.get('status', '')
            }
        
        # Grounding 업데이트 (응답에서 온 경우)
        grounding_update = self.vlm_response_parsed.get('grounding', '')
        grounding_updated = False
        if grounding_update and grounding_update.strip():
            self.prompt_organizer.update_grounding(grounding_update)
            grounding_updated = True
        
        # CLI 출력: Action, Reasoning, Memory, Grounding
        print("\n" + "=" * 80)
        print("[VLM 응답 정보]")
        print("=" * 80)
        
        # Action Chunk 출력
        print("\n[Action Chunk]")
        print("-" * 80)
        if len(action_chunk) > 0:
            for i, action in enumerate(action_chunk, 1):
                marker = "→ 실행" if i == 1 else "  예측"
                print(f"  {marker} [{i}] {action}")
        else:
            print("  (액션 없음)")
        
        # Reasoning 출력
        reasoning = self.vlm_response_parsed.get('reasoning', '')
        print("\n[Reasoning]")
        print("-" * 80)
        if reasoning:
            print(f"  {reasoning}")
        else:
            print("  (없음)")
        
        # Memory 출력
        print("\n[Memory]")
        print("-" * 80)
        spatial_desc = memory.get('spatial_description', '')
        task_goal = task_process.get('goal', '')
        task_status = task_process.get('status', '')
        prev_action = memory.get('previous_action', '')
        
        print("  Spatial Description:")
        if spatial_desc:
            print(f"    {spatial_desc}")
        else:
            print("    (없음)")
        
        print("  Task Process:")
        if task_goal or task_status:
            print(f"    Goal: {task_goal if task_goal else '(없음)'}")
            print(f"    Status: {task_status if task_status else '(없음)'}")
        else:
            print("    (없음)")
        
        print("  Previous Action:")
        if prev_action:
            print(f"    {prev_action}")
        else:
            print("    (없음)")
        
        # Grounding 출력 (업데이트된 경우만)
        if grounding_updated:
            print("\n[Grounding Update]")
            print("-" * 80)
            print(f"  {grounding_update}")
        
        print("=" * 80)
        
        print("\n[5] 액션 실행 중...")
        try:
            self.action_index = self.wrapper.parse_action(action_str)
            self.action_name = self.wrapper.ACTION_NAMES.get(self.action_index, f"action_{self.action_index}")
            print(f"실행할 액션: {self.action_name} (인덱스: {self.action_index})")
            
            _, self.reward, terminated, truncated, _ = self.wrapper.step(self.action_index)
            self.done = terminated or truncated
            
            print(f"보상: {self.reward}, 종료: {self.done}")
        except Exception as e:
            print(f"액션 실행 실패: {e}")
            self.action_index = 2
            self.action_name = "move forward"
            _, self.reward, terminated, truncated, _ = self.wrapper.step(2)
            self.done = terminated or truncated
        
        # Previous action 업데이트 (실제 실행된 액션)
        self.prompt_organizer.previous_action = self.action_name
        
        new_state = self.wrapper.get_state()
        self.visualizer.visualize_grid_cli(self.wrapper, new_state)
        updated_image = self.wrapper.get_image()
        self.visualizer.display_image(updated_image)
        
        self._log_step()
        
        return True
    
    def run(self):
        """메인 루프 실행"""
        self.initialize()
        
        while not self.done:
            if not self.run_step():
                break
            
            if self.done:
                print("\n" + "=" * 80)
                print("Goal 도착! 종료")
                print("=" * 80)
                break
            
            if self.step >= 100:
                print("\n최대 스텝 수(100)에 도달했습니다.")
                break
    
        self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        self.visualizer.cleanup()
        if self.wrapper:
            self.wrapper.close()
        if self.csv_file:
            self.csv_file.close()
        print(f"\n실험 완료. 로그는 {self.log_dir}에 저장되었습니다.")


def main():
    """메인 함수"""
    try:
        experiment = Scenario2Experiment()
        experiment.run()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
