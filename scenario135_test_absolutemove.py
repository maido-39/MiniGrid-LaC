"""
시나리오 2 실험 환경 테스트 스크립트 (절대 좌표 이동 버전 - 클래스 기반)

시나리오 2: 파란 기둥으로 가서 오른쪽으로 돌고, 테이블 옆에 멈추시오


> 여기 이제 json 에서 불러와져서, 맵 바꾸려면 json 파일만 바꾸면 됨 !!!
환경 구성: (업데이트필요)
- 벽: 검은색 (외벽)
- 파란 기둥: 파란색 2x2 Grid (통과불가, 색상이 있는 벽)
- 테이블: 보라색 1x3 Grid (통과불가, 색상이 있는 벽)
- 시작점: (1, 8)
- 종료점: (8, 1)

레이아웃 (14x14): `example_map.json` 에서 정의, emoji_map_loader.py 에서 로드됨
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬜️⬜️⬜️🟩🟩🟩⬜️🟦🟦🟦⬜️⬜️⬛
⬛⬜️⬜️⬜️🟩🟩🟩⬜️🟦🟦🟦⬜️⬜️⬛
⬛⬜️⬜️⬜️🟩🟩🟩⬜🟦🟦🟦⬜️⬜️⬛
⬛⬜️⬜️⬜️⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟩🟩🟩⬜️⬜️🟩🟩🟩⬜️⬜️⬜️⬜️⬛
⬛🟩🟩🟩⬜️⬜️🟩🟩🟩⬜️⬜️⬜️⬜️⬛
⬛🟩🟩🟩⬜️⬜️🟩🟩🟩⬜️🟩🟩🟩⬛
⬛⬜️⬜️⬜️⬜️⬛⬜️⬜️⬜️⬜️🟩🟩🟩⬛
⬛⬜️⬜️⬜️⬜️⬛⬜️⬜️⬜️⬜️🟩🟩🟩⬛
⬛⬛⬛⬛⬛⬛⬜️⬜️⬜️⬛⬛⬛⬛⬛
⬛⬜️⬜️⬜️⬜️🟩🟩🟩⬜️⬜️⬜️⬜️⬜️⬛
⬛⬜️🟥⬜️⬜️🟩🟩🟩⬜️⬜️⬜️⬜️⬜️⬛
⬛⬜️⬜️⬜️⬜️🟩🟩🟩⬜️⬜️⬜️⬜️⬜️⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛

사용법:
    python scenario2_test_absolutemove.py [json_map_path]
    예: python scenario2_test_absolutemove.py example_map.json
"""
## Import common libraries
from typing import Union  # Union은 visualize_grid_cli에서 사용
import numpy as np
import cv2
import json
import csv
from datetime import datetime
from pathlib import Path
from PIL import Image

# Import MiniGrid and VLM related classes
from minigrid import register_minigrid_envs
from minigrid_customenv_emoji import MiniGridEmojiWrapper
from emoji_map_loader import load_emoji_map_from_json
from vlm_wrapper import ChatGPT4oVLMWrapper
from vlm_postprocessor import VLMResponsePostProcessor

# MiniGrid 환경 등록
register_minigrid_envs()

# VLM 설정
VLM_MODEL = "gpt-4o"
VLM_TEMPERATURE = 0.0
VLM_MAX_TOKENS = 1000

# Mission/Task 설정
DEFAULT_MISSION = "Go to the blue pillar, turn right, then stop next to the table."


class PromptOrganizer:
    """프롬프트 관리 클래스 (절대 좌표 버전)"""
    
    def __init__(self):
        self.grounding = ""
        self.previous_action = ""
        self.task_process = {"goal": "", "status": ""}  # status: pending | in_progress | completed | blocked
    
    def get_system_prompt(self, wrapper=None, last_action_result=None) -> str:
        """전체 System Prompt 생성 (절대 좌표 버전)"""
        ## Prompt 오류 핸들링용임
        # Grounding 내용 (항상 표시, 비어있으면 빈 문자열)
        grounding_content = self.grounding if self.grounding else ""
        
        # Previous Action (항상 표시, 비어있으면 "None")
        previous_action = self.previous_action if self.previous_action else "None"
        
        # Task Process (항상 표시, 비어있으면 기본값)
        task_goal = self.task_process.get("goal", "") if self.task_process.get("goal") else "None"
        task_status = self.task_process.get("status", "") if self.task_process.get("status") else "None"
        task_process_str = f"Goal: {task_goal}, Status: {task_status}"
        
        # Last Action Result (실패 정보)
        if last_action_result and last_action_result.get("action"):
            action_result = last_action_result.get("action", "None")
            result_status = "success" if last_action_result.get("success", True) else "failed"
            failure_reason = last_action_result.get("failure_reason", "")
            position_changed = "yes" if last_action_result.get("position_changed", True) else "no"
            last_action_str = f"Action: {action_result}, Result: {result_status}"
            if not last_action_result.get("success", True):
                last_action_str += f", Failure Reason: {failure_reason}"
            last_action_str += f", Position Changed: {position_changed}"
        else:
            last_action_str = "None"
        
        
        ## 실제 적용 Prompt 시작 (절대 좌표 버전)
        return f"""You are a robot operating in a grid-based environment.

## Coordinate System (ABSOLUTE)
- Top=North, Bottom=South, Left=West, Right=East
- Use absolute directions: up/down/left/right (or north/south/east/west)

## Action Space
- "move up"/"up"/"north"/"n": Move North
- "move down"/"down"/"south"/"s": Move South
- "move left"/"left"/"west"/"w": Move West
- "move right"/"right"/"east"/"e": Move East
- "pickup", "drop", "toggle"

## CRITICAL Movement Constraints
- The robot CANNOT enter or move into colored blocks (blue, purple, green) or walls(brick) .
- Attempting to move into a colored block or wall ALWAYS fails.
- Do NOT propose actions that move into colored blocks or walls.
- Check the image to identify impassable cells before selecting actions.

## Loop Prevention (CRITICAL)
- If the same action is attempted twice consecutively AND the robot's position does not change, that action becomes INVALID and must not be selected again.
- Always check the "Last Action Result" section below to avoid repeating failed actions.

Before selecting actions:
1. Check "Last Action Result" - if previous action failed, do NOT repeat it.
2. Check if current subtask is completed (robot adjacent to target = completed).
3. Check action feasibility (target cell must be passable, NOT a colored block or wall).
4. Apply applicable grounding knowledge if situation matches.
5. Select feasible action using absolute directions.



## Grounding Knowledge (Experience from Past Failures) - CRITICAL
This section contains lessons learned from human feedback after failures.
**IMPORTANT**: When the current situation matches the conditions described in a grounding rule, you MUST apply that rule when selecting actions.
- Review each grounding rule before selecting actions.
- If a grounding rule applies to the current situation, prioritize actions that follow the rule.
- These rules help avoid repeating past mistakes.
- Match the situation carefully: only apply rules when the conditions are similar.
{grounding_content}

## Last Action Result (Authoritative - Ground Truth)
This information is FACT and MUST be trusted. Do NOT infer or reinterpret.
- Last Action: {last_action_str}
- If result is "failed", the action did not execute successfully and position did not change.
- If position_changed is "no", the robot is blocked and that direction is INVALID.

## Memory (State Continuity)
- Previous Action: {previous_action}
- Task Process: {task_process_str}

## Response Format (STRICT)
Respond in valid JSON:

```json
{{
  "action": ["<action1>", "<action2>", "<action3>"],
  "reasoning": "<why the first action is correct. MUST include: (1) last action result check (if failed, explain why not repeating), (2) task completion check, (3) action feasibility (target cell is passable), (4) loop prevention check, (5) grounding rule applied if any.>",
  "grounding": "<update grounding only if new failure feedback is detected>",
  "memory": {{
    "spatial_description": "<environment described using absolute coordinates (North/South/East/West)>",
    "task_process": {{
      "goal": "<what subtask this step was addressing>",
      "status": "<pending | in_progress | completed | blocked>",
      "blocked_reason": "<optional: reason if status is blocked>"
    }},
    "previous_action": "<set to the first selected action>",
    "last_action_result": {{
      "action": "<last attempted action>",
      "success": true | false,
      "failure_reason": "<if failed: blocked_by_obstacle | wall | unknown>",
      "position_changed": true | false
    }}
  }}
}}
```

Important:
* EXACTLY 3 actions must be provided. Only the first action will be executed.
* Actions must come from the defined action space (absolute directions: up/down/left/right/pickup/drop/toggle).
* Check task completion and action feasibility before selecting actions.
* Apply applicable grounding knowledge.
* Complete the mission specified by the user.
"""
    
    def get_feedback_system_prompt(self) -> str:
        """Feedback 생성용 System Prompt (절대 좌표 버전)"""
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
- Use absolute direction terms (North/South/East/West or up/down/left/right)
- Do NOT reference specific map positions or episode details
- Keep it general and reusable
- Exactly one sentence

## Response Format
```json
{
  "knowledge": "<single-line heuristic>"
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
        print(f"명령을 입력하세요 (Enter: {actual_default}):")
        user_input = input("> ").strip()
        
        if not user_input:
            if default_prompt:
                return f"Task: {default_prompt}\n\nBased on the current image, choose the next action to complete this task. Use absolute directions (up/down/left/right)."
            return f"Based on the current image, choose the next action to complete the mission: {DEFAULT_MISSION}. Use absolute directions (up/down/left/right)."
        
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
            print(f"VLM API 호출 실패: {e}")
            return ""
    
    def parser_action(self, raw_response: str) -> dict:
        """Action 생성 응답 파싱"""
        try:
            parsed = self.postprocessor_action.process(raw_response, strict=True)
            return parsed
        except ValueError as e:
            print(f"응답 파싱 실패: {e}")
            return {
                "action": ["0"],  # 기본값: move up
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
            print(f"Feedback 응답 파싱 실패: {e}")
            return {"knowledge": ""}


class Visualizer:
    """시각화 클래스"""
    
    def __init__(self, window_name: str = "Scenario 2: VLM Control (Absolute)"):
        self.window_name = window_name
    
    def visualize_grid_cli(self, wrapper: MiniGridEmojiWrapper, state: dict):
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
                print(f"이미지 표시 오류: {e}")
    
    def cleanup(self):
        """리소스 정리"""
        cv2.destroyAllWindows()


class UserInteraction:
    """사용자 상호작용 클래스"""
    
    def get_input(self, prompt: str = "> ") -> str:
        """사용자 입력 받기"""
        return input(prompt).strip()




class Scenario2Experiment:
    """시나리오 2 실험 메인 클래스 (Runner) - 절대 좌표 버전"""
    
    def __init__(self, log_dir: Path = None, json_map_path: str = "scenario135_example_map.json"):
        """
        Args:
            log_dir: 로그 디렉토리 경로
            json_map_path: JSON 맵 파일 경로
        """
        self.wrapper = None
        self.json_map_path = json_map_path
        self.prompt_organizer = PromptOrganizer()
        self.vlm_processor = VLMProcessor()
        self.visualizer = Visualizer()
        self.user_interaction = UserInteraction()
        
        if log_dir is None:
            map_name = Path(json_map_path).stem
            log_dir = Path("logs") / f"scenario2_absolute_{map_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.step = 0
        self.done = False
        self.state = None
        self.image = None
        self.user_prompt = ""
        self.vlm_response_raw = ""
        self.vlm_response_parsed = {}
        self.action_index = 0  # 기본값: move up
        self.action_name = "move up"
        self.reward = 0.0
        
        # Last action result tracking
        self.last_action_result = {
            "action": "",
            "success": True,
            "failure_reason": "",
            "position_changed": True
        }
        self.previous_position = None
        
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
        print("\n[3] VLM에 Action 생성 요청 전송 중...")
        raw_response = self.vlm_processor.requester(
            image=image,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        if not raw_response:
            print("VLM 응답이 비어있습니다.")
            return {}
        
        print("VLM 응답 수신 완료")
        print("[4] 응답 파싱 중...")
        parsed = self.vlm_processor.parser_action(raw_response)
        return parsed
    
    def vlm_gen_feedback(self, system_prompt: str, user_feedback: str) -> str:
        """Feedback 생성용 VLM 호출"""
        print("\n[3-F] VLM에 Feedback 분석 요청 전송 중...")
        
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
            print("Feedback VLM 응답이 비어있습니다.")
            return ""
        
        print("Feedback VLM 응답 수신 완료")
        print("[4-F] Feedback 응답 파싱 중...")
        parsed = self.vlm_processor.parser_feedback(raw_response)
        knowledge = parsed.get('knowledge', '')
        
        if knowledge:
            print(f"\n[4-F-1] 생성된 Knowledge: {knowledge}")
            self.prompt_organizer.update_grounding(knowledge)
            print("\n[4-F-2] Grounding 업데이트 완료")
            print("=" * 80)
            print("업데이트된 Grounding 내용:")
            print("-" * 80)
            print(knowledge)
            print("-" * 80)
            print("\n전체 Grounding 정보:")
            print("=" * 80)
            if self.prompt_organizer.grounding:
                print(self.prompt_organizer.grounding)
            else:
                print("(없음)")
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
                "memory_spatial_description", "memory_task_goal", "memory_task_status", "memory_task_blocked_reason", "memory_previous_action",
                "last_action_result_action", "last_action_result_success", "last_action_result_failure_reason", "last_action_result_position_changed",
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
            task_process = {"goal": "", "status": "", "blocked_reason": ""}
        
        action_chunk = self.vlm_response_parsed.get('action', [])
        if isinstance(action_chunk, str):
            try:
                action_chunk = json.loads(action_chunk)
            except Exception:
                action_chunk = [action_chunk] if action_chunk else []
        if not isinstance(action_chunk, list):
            action_chunk = [str(action_chunk)]
        
        # last_action_result 가져오기
        last_action_result = self.last_action_result if hasattr(self, 'last_action_result') else {
            "action": "",
            "success": True,
            "failure_reason": "",
            "position_changed": True
        }
        
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
            task_process.get('blocked_reason', ''),
            memory.get('previous_action', ''),
            last_action_result.get('action', ''),
            bool(last_action_result.get('success', True)),
            last_action_result.get('failure_reason', ''),
            bool(last_action_result.get('position_changed', True)),
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
            "last_action_result": last_action_result,
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
        print("시나리오 2: VLM 제어 실험 (절대 좌표 이동 버전)")
        print("=" * 60)
        print(f"\nMission: {DEFAULT_MISSION}")
        print("\n액션 공간: 상/하/좌/우로 직접 이동 가능 (절대 좌표)")
        print(f"\n로그 디렉토리: {self.log_dir}")
        
        print("\n[1] 환경 생성 중...")
        print(f"  맵 파일: {self.json_map_path}")
        self.wrapper = load_emoji_map_from_json(self.json_map_path)
        self.wrapper.reset()
        
        self.state = self.wrapper.get_state()
        print(f"에이전트 시작 위치: {self.state['agent_pos']}")
        print(f"에이전트 방향: {self.state['agent_dir']}")
        
        # 초기 위치 저장
        initial_pos = tuple(self.state['agent_pos'])
        if isinstance(initial_pos, np.ndarray):
            initial_pos = (int(initial_pos[0]), int(initial_pos[1]))
        self.previous_position = initial_pos
        
        # 초기 last_action_result 설정
        self.last_action_result = {
            "action": "",
            "success": True,
            "failure_reason": "",
            "position_changed": True
        }
        
        # 액션 공간 정보 출력
        action_space = self.wrapper.get_absolute_action_space()
        print(f"\n절대 방향 액션 공간:")
        print(f"  - 사용 가능한 액션: {action_space['actions']}")
        
        print("\n[2] VLM 초기화 완료")
        print("\n" + "=" * 60)
        print("실험 시작")
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
        print(f"위치: {self.state['agent_pos']}, 방향: {self.state['agent_dir']} ({heading})")
        print(f"현재 Heading: {heading_desc}")
        
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
        system_prompt = self.prompt_organizer.get_system_prompt(self.wrapper, self.last_action_result)
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
            action_str = '0'  # 기본값: move up
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
            task_process = {"goal": "", "status": "", "blocked_reason": ""}
        
        # last_action_result 파싱 (VLM 응답에서)
        vlm_last_action_result = memory.get('last_action_result', {})
        if not isinstance(vlm_last_action_result, dict):
            vlm_last_action_result = {}
        
        # Memory 업데이트
        if isinstance(memory, dict):
            self.prompt_organizer.previous_action = memory.get('previous_action', action_str)
            self.prompt_organizer.task_process = {
                "goal": task_process.get('goal', ''),
                "status": task_process.get('status', ''),
                "blocked_reason": task_process.get('blocked_reason', '')
            }
            
            # VLM이 blocked 상태로 설정한 경우 반영
            if task_process.get('status') == 'blocked':
                blocked_reason = task_process.get('blocked_reason', '')
                if blocked_reason:
                    print(f"\n[Memory] Task marked as blocked: {blocked_reason}")
        
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
        
        # 현재 위치 저장 (액션 실행 전)
        current_pos_before = tuple(self.state['agent_pos'])
        if isinstance(current_pos_before, np.ndarray):
            current_pos_before = (int(current_pos_before[0]), int(current_pos_before[1]))
        
        try:
            self.action_index = self.wrapper.parse_absolute_action(action_str)
            action_space = self.wrapper.get_absolute_action_space()
            self.action_name = action_space['action_mapping'].get(self.action_index, f"action_{self.action_index}")
            print(f"실행할 액션: {self.action_name} (인덱스: {self.action_index})")
            
            # use_absolute_movement=True이므로 step()이 절대 움직임을 처리
            _, self.reward, terminated, truncated, _ = self.wrapper.step(self.action_index)
            self.done = terminated or truncated
            
            # 액션 실행 후 위치 확인
            new_state = self.wrapper.get_state()
            current_pos_after = tuple(new_state['agent_pos'])
            if isinstance(current_pos_after, np.ndarray):
                current_pos_after = (int(current_pos_after[0]), int(current_pos_after[1]))
            
            # 위치 변화 확인
            position_changed = (current_pos_before != current_pos_after)
            
            # 액션 결과 판단
            action_success = position_changed or self.reward > 0
            failure_reason = ""
            if not action_success:
                # 실패 원인 추론 (이미지에서 확인 가능한 정보 기반)
                if not position_changed:
                    failure_reason = "blocked_by_obstacle"
                else:
                    failure_reason = "unknown"
            
            # Last action result 업데이트
            self.last_action_result = {
                "action": self.action_name,
                "success": action_success,
                "failure_reason": failure_reason,
                "position_changed": position_changed
            }
            
            print(f"보상: {self.reward}, 종료: {self.done}")
            print(f"액션 결과: {'성공' if action_success else '실패'} (위치 변화: {'예' if position_changed else '아니오'})")
            if not action_success:
                print(f"실패 원인: {failure_reason}")
                
        except Exception as e:
            print(f"액션 실행 실패: {e}")
            import traceback
            traceback.print_exc()
            self.action_index = 0
            self.action_name = "move up"
            try:
                _, self.reward, terminated, truncated, _ = self.wrapper.step(0)
                self.done = terminated or truncated
            except:
                pass
            
            # 예외 발생 시에도 last_action_result 업데이트
            self.last_action_result = {
                "action": self.action_name,
                "success": False,
                "failure_reason": "exception",
                "position_changed": False
            }
        
        # Previous action 업데이트 (실제 실행된 액션)
        self.prompt_organizer.previous_action = self.action_name
        
        # new_state는 이미 위에서 가져왔으므로 재사용
        if 'new_state' not in locals():
            new_state = self.wrapper.get_state()
        self.state = new_state
        self.visualizer.visualize_grid_cli(self.wrapper, new_state)
        updated_image = self.wrapper.get_image()
        self.image = updated_image
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
    import sys
    
    # 명령줄 인자로 JSON 맵 파일 경로 지정
    json_map_path = "scenario135_example_map.json"
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("사용법:")
            print("  python scenario2_test_absolutemove.py [json_map_path]")
            print("  예: python scenario2_test_absolutemove.py scenario135_example_map.json")
            return
        else:
            json_map_path = sys.argv[1]
    
    try:
        experiment = Scenario2Experiment(json_map_path=json_map_path)
        experiment.run()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

