"""
Scenario 2 experiment environment test script (absolute movement version - class-based)

Scenario 2: Go to the blue pillar, turn right, then stop next to the table.

> Maps are now loaded from JSON files. To change the map, simply modify the JSON file.
Environment configuration: (needs update)
- Walls: Black (outer walls)
- Blue pillar: Blue 2x2 Grid (impassable, colored wall)
- Table: Purple 1x3 Grid (impassable, colored wall)
- Start position: (1, 8)
- Goal position: (8, 1)

Layout (14x14): Defined in `example_map.json`, loaded by emoji_map_loader.py
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

Usage:
    python scenario2_test_absolutemove.py [json_map_path]
    Example: python scenario2_test_absolutemove.py example_map.json
"""
## Import common libraries
from typing import Union  # Union is used in visualize_grid_cli
import numpy as np
import cv2
import json
import csv
from datetime import datetime
from pathlib import Path
from PIL import Image

# Import MiniGrid and VLM related classes
from minigrid import register_minigrid_envs
# Actual paths: utils.map_manager.minigrid_customenv_emoji, utils.map_manager.emoji_map_loader
from utils import MiniGridEmojiWrapper, load_emoji_map_from_json
# Actual paths: utils.vlm.vlm_wrapper, utils.vlm.vlm_postprocessor
from utils import ChatGPT4oVLMWrapper, VLMResponsePostProcessor

# Register MiniGrid environments
register_minigrid_envs()

# VLM configuration
VLM_MODEL = "gpt-4o"
VLM_TEMPERATURE = 0.0
VLM_MAX_TOKENS = 1000

# Mission/Task configuration
DEFAULT_MISSION = "Go to the blue pillar, turn right, then stop next to the table."


class PromptOrganizer:
    """Prompt management class (absolute coordinate version)"""
    
    def __init__(self):
        self.grounding = ""
        self.previous_action = ""
        self.task_process = {"goal": "", "status": ""}  # status: pending | in_progress | completed | blocked
    
    def get_system_prompt(self, wrapper=None, last_action_result=None) -> str:
        """Generate complete System Prompt (absolute coordinate version)"""
        ## For prompt error handling
        # Grounding content (always displayed, empty string if empty)
        grounding_content = self.grounding if self.grounding else ""
        
        # Previous Action (always displayed, "None" if empty)
        previous_action = self.previous_action if self.previous_action else "None"
        
        # Task Process (always displayed, default value if empty)
        task_goal = self.task_process.get("goal", "") if self.task_process.get("goal") else "None"
        task_status = self.task_process.get("status", "") if self.task_process.get("status") else "None"
        task_process_str = f"Goal: {task_goal}, Status: {task_status}"
        
        # Last Action Result (failure information)
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
        
        
        ## Actual prompt starts here (absolute coordinate version)
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
- The robot CANNOT enter or move into colored blocks (blue, purple, green) or walls (black).
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
        """System Prompt for feedback generation (absolute coordinate version)"""
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
        """Accumulate and update grounding knowledge"""
        if new_grounding and new_grounding.strip():
            if self.grounding:
                self.grounding = f"{self.grounding}\n\n{new_grounding.strip()}"
            else:
                self.grounding = new_grounding.strip()
    
    def get_user_prompt(self, default_prompt: str = None) -> str:
        """Get user prompt input"""
        # Determine default prompt (use DEFAULT_MISSION if default_prompt is None)
        actual_default = default_prompt if default_prompt else DEFAULT_MISSION
        
        if default_prompt:
            print(f"Task Hint: {default_prompt}")
        print(f"Enter command (Enter: {actual_default}):")
        user_input = input("> ").strip()
        
        if not user_input:
            if default_prompt:
                return f"Task: {default_prompt}\n\nBased on the current image, choose the next action to complete this task. Use absolute directions (up/down/left/right)."
            return f"Based on the current image, choose the next action to complete the mission: {DEFAULT_MISSION}. Use absolute directions (up/down/left/right)."
        
        return user_input


class VLMProcessor:
    """VLM request and parsing processing class"""
    
    def __init__(self, model: str = VLM_MODEL, temperature: float = VLM_TEMPERATURE, max_tokens: int = VLM_MAX_TOKENS):
        self.vlm = ChatGPT4oVLMWrapper(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.postprocessor_action = VLMResponsePostProcessor(required_fields=["action", "reasoning", "grounding", "memory"])
        self.postprocessor_feedback = VLMResponsePostProcessor(required_fields=["knowledge"])
    
    def requester(self, image: np.ndarray, system_prompt: str, user_prompt: str) -> str:
        """Send request to VLM (base method)"""
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
        """Parse action generation response"""
        try:
            parsed = self.postprocessor_action.process(raw_response, strict=True)
            return parsed
        except ValueError as e:
            print(f"Response parsing failed: {e}")
            return {
                "action": ["0"],  # Default: move up
                "reasoning": "Parsing failed",
                "grounding": "",
                "memory": {
                    "spatial_description": "",
                    "task_process": {"goal": "", "status": ""},
                    "previous_action": ""
                }
            }
    
    def parser_feedback(self, raw_response: str) -> dict:
        """Parse feedback generation response"""
        try:
            parsed = self.postprocessor_feedback.process(raw_response, strict=True)
            return parsed
        except ValueError as e:
            print(f"Feedback response parsing failed: {e}")
            return {"knowledge": ""}


class Visualizer:
    """Visualization class"""
    
    def __init__(self, window_name: str = "Scenario 2: VLM Control (Absolute)"):
        self.window_name = window_name
    
    def visualize_grid_cli(self, wrapper: MiniGridEmojiWrapper, state: dict):
        """Visualize grid as text in CLI"""
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
                elif cell is not None and cell.type == 'emoji':
                    # Emoji object display
                    if hasattr(cell, 'emoji_name'):
                        emoji_map = {
                            'brick': '🧱',
                            'desktop': '🖥️',
                            'workstation': '📱',
                            'tree': '🌲',
                            'mushroom': '🍄',
                            'flower': '🌼',
                            'cat': '🐈',
                            'grass': '🌾',
                            'rock': '🗿',
                            'box': '📦',
                            'chair': '🪑',
                            'apple': '🍎'
                        }
                        emoji_char = emoji_map.get(cell.emoji_name, '❓')
                        row.append(emoji_char)
                    else:
                        row.append('❓')
                elif cell is not None:
                    if hasattr(cell, 'color'):
                        if cell.color == 'blue':
                            row.append('🟦')
                        elif cell.color == 'purple':
                            row.append('🟪')
                        elif cell.color == 'green':
                            row.append('🟩')
                        elif cell.color == 'yellow':
                            row.append('🟨')
                        else:
                            row.append('🟨')
                    else:
                        row.append('🟨')
                else:
                    # Check for floor tiles (empty cell with floor color)
                    floor_char = '⬜️'
                    if hasattr(env, 'floor_tiles') and (x, y) in env.floor_tiles:
                        floor_color = env.floor_tiles[(x, y)]
                        if floor_color == 'blue':
                            floor_char = '🟦'
                        elif floor_color == 'purple':
                            floor_char = '🟪'
                        elif floor_color == 'green':
                            floor_char = '🟩'
                        elif floor_color == 'yellow':
                            floor_char = '🟨'
                    row.append(floor_char)
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
        """Display image using OpenCV"""
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
        """Clean up resources"""
        cv2.destroyAllWindows()


class UserInteraction:
    """User interaction class"""
    
    def get_input(self, prompt: str = "> ") -> str:
        """Get user input"""
        return input(prompt).strip()




class Scenario2Experiment:
    """Scenario 2 experiment main class (Runner) - absolute coordinate version"""
    
    def __init__(self, log_dir: Path = None, json_map_path: str = None):
        """
        Args:
            log_dir: Log directory path
            json_map_path: JSON map file path. If None, uses default path relative to script location.
        """
        # Resolve json_map_path if provided
        if json_map_path is None:
            script_dir = Path(__file__).parent.resolve()
            json_map_path = str(script_dir / "config" / "example_map.json")
        else:
            json_path_obj = Path(json_map_path)
            if not json_path_obj.is_absolute():
                # Try relative to script directory first
                script_dir = Path(__file__).parent.resolve()
                script_relative = script_dir / json_path_obj
                if script_relative.exists():
                    json_map_path = str(script_relative.resolve())
                else:
                    # Try relative to current working directory
                    cwd_relative = Path.cwd() / json_path_obj
                    if cwd_relative.exists():
                        json_map_path = str(cwd_relative.resolve())
                    else:
                        # Use as-is (will raise error in load_emoji_map_from_json if not found)
                        json_map_path = str(json_path_obj)
            else:
                json_map_path = str(json_path_obj.resolve())
        
        self.wrapper = None
        self.json_map_path = json_map_path
        self.prompt_organizer = PromptOrganizer()
        self.vlm_processor = VLMProcessor()
        self.visualizer = Visualizer()
        self.user_interaction = UserInteraction()
        
        if log_dir is None:
            map_name = Path(json_map_path).stem
            log_dir = Path("../logs") / f"scenario2_absolute_{map_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.step = 0
        self.done = False
        self.state = None
        self.image = None
        self.user_prompt = ""
        self.vlm_response_raw = ""
        self.vlm_response_parsed = {}
        self.action_index = 0  # Default: move up
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
        """Evaluate feedback (internal method)"""
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
        """VLM call for action generation"""
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
        """VLM call for feedback generation"""
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
            print("\n[4-F-2] Grounding update complete")
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
                print("(none)")
            print("=" * 80)
        
        return knowledge
    
    def _init_csv_logging(self):
        """Initialize CSV logging"""
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
        """Log current step"""
        timestamp = datetime.now().isoformat()
        
        agent_pos = self.state['agent_pos']
        if isinstance(agent_pos, np.ndarray):
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        else:
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        
        image_path = f"step_{self.step:04d}.png"
        
        # Parse memory
        memory = self.vlm_response_parsed.get('memory', {})
        if isinstance(memory, str):
            try:
                memory = json.loads(memory)
            except Exception:
                memory = {"spatial_description": "", "task_process": {"goal": "", "status": ""}, "previous_action": ""}
        elif not isinstance(memory, dict):
            memory = {"spatial_description": "", "task_process": {"goal": "", "status": ""}, "previous_action": ""}
        
        # Parse task_process
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
        
        # Get last_action_result
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
        """Initialize experiment"""
        print("=" * 60)
        print("Scenario 2: VLM Control Experiment (Absolute Movement Version)")
        print("=" * 60)
        print(f"\nMission: {DEFAULT_MISSION}")
        print("\nAction space: Direct movement up/down/left/right (absolute coordinates)")
        print(f"\nLog directory: {self.log_dir}")
        
        print("\n[1] Creating environment...")
        print(f"  Map file: {self.json_map_path}")
        self.wrapper = load_emoji_map_from_json(self.json_map_path)
        self.wrapper.reset()
        
        self.state = self.wrapper.get_state()
        print(f"Agent start position: {self.state['agent_pos']}")
        print(f"Agent direction: {self.state['agent_dir']}")
        
        # Save initial position
        initial_pos = tuple(self.state['agent_pos'])
        if isinstance(initial_pos, np.ndarray):
            initial_pos = (int(initial_pos[0]), int(initial_pos[1]))
        self.previous_position = initial_pos
        
        # Initialize last_action_result
        self.last_action_result = {
            "action": "",
            "success": True,
            "failure_reason": "",
            "position_changed": True
        }
        
        # Print action space information
        action_space = self.wrapper.get_absolute_action_space()
        print(f"\nAbsolute direction action space:")
        print(f"  - Available actions: {action_space['actions']}")
        
        print("\n[2] VLM initialization complete")
        print("\n" + "=" * 60)
        print("Experiment started")
        print("=" * 60)
    
    def run_step(self):
        """Execute one step"""
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
        
        # Evaluate feedback
        has_feedback = self._evaluate_feedback(self.user_prompt)
        
        if has_feedback:
            # Process feedback: if it starts with "feedback : "
            if self.user_prompt.lower().startswith("feedback :"):
                feedback_text = self.user_prompt[10:].strip()  # Remove "feedback : "
            else:
                feedback_text = self.user_prompt
            
            # Call VLM for feedback generation
            system_prompt = self.prompt_organizer.get_system_prompt(self.wrapper)
            self.vlm_gen_feedback(system_prompt, feedback_text)
            
            # Skip normal action generation after feedback processing
            print("\n[4-1] Feedback processing complete. Proceeding to next step.")
            return True
        
        # Normal action generation
        system_prompt = self.prompt_organizer.get_system_prompt(self.wrapper, self.last_action_result)
        self.vlm_response_parsed = self.vlm_gen_action(
            image=self.image,
            system_prompt=system_prompt,
            user_prompt=self.user_prompt
        )
        
        if not self.vlm_response_parsed:
            return False
        
        # Extract only the first action from action chunk
        action_chunk = self.vlm_response_parsed.get('action', [])
        if isinstance(action_chunk, str):
            try:
                action_chunk = json.loads(action_chunk)
            except Exception:
                action_chunk = [action_chunk] if action_chunk else []
        if not isinstance(action_chunk, list):
            action_chunk = [str(action_chunk)]
        
        if len(action_chunk) == 0:
            action_str = '0'  # Default: move up
        else:
            action_str = str(action_chunk[0])
        
        # Parse memory
        memory = self.vlm_response_parsed.get('memory', {})
        if isinstance(memory, str):
            try:
                memory = json.loads(memory)
            except Exception:
                memory = {}
        if not isinstance(memory, dict):
            memory = {}
        
        # Parse task_process
        task_process = memory.get('task_process', {})
        if not isinstance(task_process, dict):
            task_process = {"goal": "", "status": "", "blocked_reason": ""}
        
        # Parse last_action_result (from VLM response)
        vlm_last_action_result = memory.get('last_action_result', {})
        if not isinstance(vlm_last_action_result, dict):
            vlm_last_action_result = {}
        
        # Update memory
        if isinstance(memory, dict):
            self.prompt_organizer.previous_action = memory.get('previous_action', action_str)
            self.prompt_organizer.task_process = {
                "goal": task_process.get('goal', ''),
                "status": task_process.get('status', ''),
                "blocked_reason": task_process.get('blocked_reason', '')
            }
            
            # Reflect if VLM set status to blocked
            if task_process.get('status') == 'blocked':
                blocked_reason = task_process.get('blocked_reason', '')
                if blocked_reason:
                    print(f"\n[Memory] Task marked as blocked: {blocked_reason}")
        
        # Update grounding (if from response)
        grounding_update = self.vlm_response_parsed.get('grounding', '')
        grounding_updated = False
        if grounding_update and grounding_update.strip():
            self.prompt_organizer.update_grounding(grounding_update)
            grounding_updated = True
        
        # CLI output: Action, Reasoning, Memory, Grounding
        print("\n" + "=" * 80)
        print("[VLM Response Information]")
        print("=" * 80)
        
        # Print Action Chunk
        print("\n[Action Chunk]")
        print("-" * 80)
        if len(action_chunk) > 0:
            for i, action in enumerate(action_chunk, 1):
                marker = "→ Execute" if i == 1 else "  Predict"
                print(f"  {marker} [{i}] {action}")
        else:
            print("  (no action)")
        
        # Print Reasoning
        reasoning = self.vlm_response_parsed.get('reasoning', '')
        print("\n[Reasoning]")
        print("-" * 80)
        if reasoning:
            print(f"  {reasoning}")
        else:
            print("  (none)")
        
        # Print Memory
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
            print("    (none)")
        
        print("  Task Process:")
        if task_goal or task_status:
            print(f"    Goal: {task_goal if task_goal else '(none)'}")
            print(f"    Status: {task_status if task_status else '(none)'}")
        else:
            print("    (none)")
        
        print("  Previous Action:")
        if prev_action:
            print(f"    {prev_action}")
        else:
            print("    (none)")
        
        # Print Grounding (only if updated)
        if grounding_updated:
            print("\n[Grounding Update]")
            print("-" * 80)
            print(f"  {grounding_update}")
        
        print("=" * 80)
        
        print("\n[5] Executing action...")
        
        # Save current position (before action execution)
        current_pos_before = tuple(self.state['agent_pos'])
        if isinstance(current_pos_before, np.ndarray):
            current_pos_before = (int(current_pos_before[0]), int(current_pos_before[1]))
        
        try:
            self.action_index = self.wrapper.parse_absolute_action(action_str)
            action_space = self.wrapper.get_absolute_action_space()
            self.action_name = action_space['action_mapping'].get(self.action_index, f"action_{self.action_index}")
            print(f"Action to execute: {self.action_name} (index: {self.action_index})")
            
            # step() handles absolute movement since use_absolute_movement=True
            _, self.reward, terminated, truncated, _ = self.wrapper.step(self.action_index)
            self.done = terminated or truncated
            
            # Check position after action execution
            new_state = self.wrapper.get_state()
            current_pos_after = tuple(new_state['agent_pos'])
            if isinstance(current_pos_after, np.ndarray):
                current_pos_after = (int(current_pos_after[0]), int(current_pos_after[1]))
            
            # Check position change
            position_changed = (current_pos_before != current_pos_after)
            
            # Determine action result
            action_success = position_changed or self.reward > 0
            failure_reason = ""
            if not action_success:
                # Infer failure reason (based on information available from image)
                if not position_changed:
                    failure_reason = "blocked_by_obstacle"
                else:
                    failure_reason = "unknown"
            
            # Update last action result
            self.last_action_result = {
                "action": self.action_name,
                "success": action_success,
                "failure_reason": failure_reason,
                "position_changed": position_changed
            }
            
            print(f"Reward: {self.reward}, Done: {self.done}")
            print(f"Action result: {'Success' if action_success else 'Failed'} (Position changed: {'Yes' if position_changed else 'No'})")
            if not action_success:
                print(f"Failure reason: {failure_reason}")
                
        except Exception as e:
            print(f"Action execution failed: {e}")
            import traceback
            traceback.print_exc()
            self.action_index = 0
            self.action_name = "move up"
            try:
                _, self.reward, terminated, truncated, _ = self.wrapper.step(0)
                self.done = terminated or truncated
            except:
                pass
            
            # Update last_action_result even on exception
            self.last_action_result = {
                "action": self.action_name,
                "success": False,
                "failure_reason": "exception",
                "position_changed": False
            }
        
        # Update previous action (actually executed action)
        self.prompt_organizer.previous_action = self.action_name
        
        # Reuse new_state already fetched above
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
        """Execute main loop"""
        self.initialize()
        
        while not self.done:
            if not self.run_step():
                break
            
            if self.done:
                print("\n" + "=" * 80)
                print("Goal reached! Terminating")
                print("=" * 80)
                break
            
            if self.step >= 100:
                print("\nMaximum step count (100) reached.")
                break
    
    def cleanup(self):
        """Clean up resources"""
        self.visualizer.cleanup()
        if self.wrapper:
            self.wrapper.close()
        if self.csv_file:
            self.csv_file.close()
        print(f"\nExperiment complete. Logs saved to {self.log_dir}")


def main():
    """Main function"""
    import sys
    
    # Get script directory for relative path resolution
    script_dir = Path(__file__).parent.resolve()
    
    # Specify JSON map file path via command line argument
    # Default path is relative to script location
    default_json_path = script_dir / "config" / "example_map.json"
    json_map_path = str(default_json_path)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage:")
            print("  python scenario2_test_absolutemove.py [json_map_path]")
            print("  Example: python scenario2_test_absolutemove.py ./config/example_map.json")
            print("  Example: python scenario2_test_absolutemove.py ../config/example_map.json")
            return
        else:
            # Resolve path relative to script directory if it's a relative path
            user_path = Path(sys.argv[1])
            if user_path.is_absolute():
                json_map_path = str(user_path)
            else:
                # Try relative to script directory first, then relative to current working directory
                script_relative = script_dir / user_path
                cwd_relative = Path.cwd() / user_path
                if script_relative.exists():
                    json_map_path = str(script_relative.resolve())
                elif cwd_relative.exists():
                    json_map_path = str(cwd_relative.resolve())
                else:
                    # Use as-is and let load_emoji_map_from_json handle the error
                    json_map_path = str(user_path)
    
    try:
        experiment = Scenario2Experiment(json_map_path=json_map_path)
        experiment.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

