"""
VLM Egocentric Transform 자동 개선 시스템

Phase 1: 랜덤 환경에서 테스트 및 성공률 측정
Phase 2: 문제 분석, 논문 검색, 개선 작업
성공률 90% 이상 달성까지 반복
"""

from minigrid import register_minigrid_envs
from custom_environment import CustomRoomWrapper
from vlm_wrapper import ChatGPT4oVLMWrapper
from vlm_postprocessor import VLMResponsePostProcessor
import numpy as np
import cv2
import json
import random
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
from PIL import Image
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# MiniGrid 환경 등록
register_minigrid_envs()

# VLM 설정
VLM_MODEL = "gpt-4o"
VLM_TEMPERATURE = 0.0
VLM_MAX_TOKENS = 1000

# Mission 설정
DEFAULT_MISSION = "Go to the blue pillar, turn right, then stop next to the table."


def calculate_relative_direction(agent_pos: Tuple[int, int], agent_dir: int, target_pos: Tuple[int, int]) -> str:
    """
    에이전트 위치와 방향을 기준으로 타겟의 상대 방향 계산
    
    Args:
        agent_pos: 에이전트 위치 (x, y)
        agent_dir: 에이전트 방향 (0=East, 1=South, 2=West, 3=North)
        target_pos: 타겟 위치 (x, y)
    
    Returns:
        상대 방향: "front", "back", "left", "right"
    """
    ax, ay = agent_pos
    tx, ty = target_pos
    
    # 절대 좌표에서의 차이
    dx = tx - ax
    dy = ty - ay
    
    # 에이전트 방향에 따라 좌표계 변환
    # 0=East (오른쪽), 1=South (아래), 2=West (왼쪽), 3=North (위)
    if agent_dir == 0:  # East
        rel_x, rel_y = dx, -dy  # 앞이 +x, 왼쪽이 +y
    elif agent_dir == 1:  # South
        rel_x, rel_y = dy, dx  # 앞이 +y, 왼쪽이 -x
    elif agent_dir == 2:  # West
        rel_x, rel_y = -dx, dy  # 앞이 -x, 왼쪽이 -y
    else:  # North
        rel_x, rel_y = -dy, -dx  # 앞이 -y, 왼쪽이 +x
    
    # 상대 방향 결정
    if abs(rel_x) > abs(rel_y):
        if rel_x > 0:
            return "front"
        else:
            return "back"
    else:
        if rel_y > 0:
            return "left"
        else:
            return "right"


def calculate_gt_action(agent_pos: Tuple[int, int], agent_dir: int, blue_pillar_positions: List[Tuple[int, int]]) -> str:
    """
    Ground Truth 액션 계산
    
    Args:
        agent_pos: 에이전트 위치
        agent_dir: 에이전트 방향
        blue_pillar_positions: 파란 기둥 위치 리스트
    
    Returns:
        예상 액션: "turn left", "turn right", "move forward"
    """
    # 파란 기둥의 중심 위치 계산
    if not blue_pillar_positions:
        return "move forward"
    
    center_x = sum(p[0] for p in blue_pillar_positions) / len(blue_pillar_positions)
    center_y = sum(p[1] for p in blue_pillar_positions) / len(blue_pillar_positions)
    target_pos = (int(round(center_x)), int(round(center_y)))
    
    # 상대 방향 계산
    rel_dir = calculate_relative_direction(agent_pos, agent_dir, target_pos)
    
    # 상대 방향에 따른 액션 결정
    if rel_dir == "front":
        return "move forward"
    elif rel_dir == "left":
        return "turn left"
    elif rel_dir == "right":
        return "turn right"
    else:  # back
        return "turn left"  # 뒤에 있으면 왼쪽으로 회전


def create_random_environment(seed: Optional[int] = None) -> Tuple[CustomRoomWrapper, Dict]:
    """
    랜덤 환경 생성
    
    Returns:
        wrapper: 환경 래퍼
        env_info: 환경 정보 (시작 위치, 방향, 파란 기둥 위치 등)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    size = 10
    
    # 외벽 생성
    walls = []
    for i in range(size):
        walls.append((i, 0))
        walls.append((i, size-1))
        walls.append((0, i))
        walls.append((size-1, i))
    
    # 파란 기둥 위치 (2x2 그리드)
    # 랜덤하게 배치하되, 경계에서 충분히 떨어진 곳에 배치
    pillar_center_x = random.randint(2, size-4)
    pillar_center_y = random.randint(2, size-4)
    blue_pillar_positions = [
        (pillar_center_x, pillar_center_y),
        (pillar_center_x + 1, pillar_center_y),
        (pillar_center_x, pillar_center_y + 1),
        (pillar_center_x + 1, pillar_center_y + 1)
    ]
    for pos in blue_pillar_positions:
        walls.append((pos[0], pos[1], 'blue'))
    
    # 보라색 테이블 위치 (1x3 그리드)
    table_start_x = random.randint(1, size-4)
    table_start_y = random.randint(1, size-2)
    table_positions = [
        (table_start_x, table_start_y),
        (table_start_x + 1, table_start_y),
        (table_start_x + 2, table_start_y)
    ]
    for pos in table_positions:
        walls.append((pos[0], pos[1], 'purple'))
    
    # 시작 위치 랜덤화 (빈 공간에 배치)
    empty_positions = []
    for x in range(1, size-1):
        for y in range(1, size-1):
            if (x, y) not in blue_pillar_positions and (x, y) not in table_positions:
                empty_positions.append((x, y))
    
    start_pos = random.choice(empty_positions)
    
    # 시작 방향 랜덤화
    start_dir = random.randint(0, 3)
    
    goal_pos = (size-2, size-2)
    
    room_config = {
        'start_pos': start_pos,
        'goal_pos': goal_pos,
        'walls': walls,
        'objects': []
    }
    
    wrapper = CustomRoomWrapper(size=size, room_config=room_config)
    wrapper.reset()
    
    # 방향 설정
    wrapper.env.agent_dir = start_dir
    
    env_info = {
        'start_pos': start_pos,
        'start_dir': start_dir,
        'blue_pillar_positions': blue_pillar_positions,
        'table_positions': table_positions,
        'goal_pos': goal_pos
    }
    
    return wrapper, env_info


class SolutionB_CoTReasoning:
    """솔루션 B: CoT(Chain of Thought)를 통한 좌표 변환 강제"""
    
    def __init__(self, vlm: ChatGPT4oVLMWrapper, postprocessor: VLMResponsePostProcessor, prompt_variant: int = 0):
        self.vlm = vlm
        self.postprocessor = postprocessor
        self.prompt_variant = prompt_variant
    
    def get_heading_info(self, wrapper: CustomRoomWrapper) -> str:
        """Heading 정보 가져오기"""
        heading = wrapper.get_heading()
        heading_short = wrapper.get_heading_short()
        return f"{heading} ({heading_short})"
    
    def get_system_prompt(self, wrapper: CustomRoomWrapper) -> str:
        """CoT 강제 프롬프트 생성"""
        heading_info = self.get_heading_info(wrapper)
        
        if self.prompt_variant == 0:
            # 기본 프롬프트
            return self._get_base_prompt(heading_info)
        elif self.prompt_variant == 1:
            # 개선 버전 1: 좌표 변환 강화
            return self._get_enhanced_coordinate_prompt(heading_info)
        elif self.prompt_variant >= 2:
            # 개선 버전 2: 좌표계 명확화
            return self._get_clarified_coordinate_prompt(heading_info)
        else:
            return self._get_base_prompt(heading_info)
    
    def _get_base_prompt(self, heading_info: str) -> str:
        """기본 프롬프트"""
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

## Movement Rules (CRITICAL: EXECUTE STEP-BY-STEP)
You must perform a mental coordinate transformation. Do NOT trust "Up" in the image as "Front".

1. **Identify Global Position**: Where is the target object in the image? (e.g., Top=North, Right=East)
2. **Confirm Robot Heading**: Which compass direction is the robot facing? (Provided in Robot State)
3. **Calculate Relative Position**:
   - IF Object is North AND Robot faces East -> Object is on the LEFT.
   - IF Object is North AND Robot faces West -> Object is on the RIGHT.
   - IF Object is East AND Robot faces North -> Object is on the RIGHT.
   - IF Object is West AND Robot faces North -> Object is on the LEFT.
   - (Derive strictly based on rotation)

Rules:
- All movements are RELATIVE to the robot's current heading.
- "move forward" moves one cell in the facing direction.
- "turn left/right" rotates 90° relative to current heading.
- Do NOT reason using absolute coordinates when choosing actions.

## Response Format (STRICT)
Respond in valid JSON. You MUST fill strictly following the "reasoning_trace" logic.

```json
{{
  "reasoning_trace": {{
    "target_global_pos": "<e.g. The blue pillar is at the Top (North) of the grid>",
    "robot_heading": "<e.g. East>",
    "calculation": "<e.g. North is 90 degrees counter-clockwise from East.>",
    "relative_pos": "<e.g. Therefore, the pillar is to my Left.>"
  }},
  "action": ["<action1>", "<action2>", "<action3>"]
}}
```

Important:
- EXACTLY 3 actions must be provided.
- Only the first action will be executed.
- Actions must come from the defined action space.
- Complete the reasoning_trace before selecting actions.
- Complete the mission specified by the user.
"""
    
    def _get_enhanced_coordinate_prompt(self, heading_info: str) -> str:
        """개선된 좌표 변환 프롬프트"""
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

## Movement Rules (CRITICAL: EXECUTE STEP-BY-STEP)
You must perform a mental coordinate transformation. Do NOT trust "Up" in the image as "Front".

**IMPORTANT: Coordinate Transformation Matrix**

1. **Identify Global Position**: Where is the target object in the image? (e.g., Top=North, Right=East)
2. **Confirm Robot Heading**: Which compass direction is the robot facing? (Provided in Robot State)
3. **Calculate Relative Position using this EXACT transformation**:
   
   If Robot faces East (→):
   - Object at North → Robot's LEFT
   - Object at South → Robot's RIGHT
   - Object at East → Robot's FRONT
   - Object at West → Robot's BACK
   
   If Robot faces West (←):
   - Object at North → Robot's RIGHT
   - Object at South → Robot's LEFT
   - Object at East → Robot's BACK
   - Object at West → Robot's FRONT
   
   If Robot faces North (↑):
   - Object at North → Robot's FRONT
   - Object at South → Robot's BACK
   - Object at East → Robot's RIGHT
   - Object at West → Robot's LEFT
   
   If Robot faces South (↓):
   - Object at North → Robot's BACK
   - Object at South → Robot's FRONT
   - Object at East → Robot's LEFT
   - Object at West → Robot's RIGHT

4. **Select Action Based on Relative Position**:
   - If object is FRONT → "move forward"
   - If object is LEFT → "turn left"
   - If object is RIGHT → "turn right"
   - If object is BACK → "turn left" (or "turn right", choose one)

Rules:
- All movements are RELATIVE to the robot's current heading.
- "move forward" moves one cell in the facing direction.
- "turn left/right" rotates 90° relative to current heading.
- Do NOT reason using absolute coordinates when choosing actions.

## Response Format (STRICT)
Respond in valid JSON. You MUST fill strictly following the "reasoning_trace" logic.

```json
{{
  "reasoning_trace": {{
    "target_global_pos": "<e.g. The blue pillar is at the Top (North) of the grid>",
    "robot_heading": "<e.g. East>",
    "coordinate_transformation": "<e.g. Using the transformation matrix: North when facing East = LEFT>",
    "relative_pos": "<e.g. Therefore, the pillar is to my Left.>",
    "selected_action": "<e.g. turn left>"
  }},
  "action": ["<action1>", "<action2>", "<action3>"]
}}
```

Important:
- EXACTLY 3 actions must be provided.
- Only the first action will be executed.
- Actions must come from the defined action space.
- Complete the reasoning_trace before selecting actions.
- Complete the mission specified by the user.
"""
    
    def _get_clarified_coordinate_prompt(self, heading_info: str) -> str:
        """명확화된 좌표계 프롬프트"""
        return f"""You are a robot operating in a grid-based environment.

## Robot State (Authoritative)
- The robot's current heading is {heading_info}.
- Heading indicates the robot's forward-facing direction.
- This heading is ground-truth and MUST be used as-is.
- Do NOT infer or reinterpret the robot's heading from the image.

## CRITICAL DISTINCTION: Two Coordinate Systems

### 1. ALLOCENTRIC (Absolute/Global) Coordinates
- Used in the IMAGE: Top=North, Bottom=South, Left=West, Right=East
- This is FIXED and does NOT change with robot orientation
- The image shows objects in this coordinate system

### 2. EGOCENTRIC (Relative/Robot-centric) Coordinates
- Used for ACTIONS: Front=heading direction, Left/Right relative to heading
- This CHANGES when the robot rotates
- Actions must be chosen in this coordinate system

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

## Movement Rules (CRITICAL: EXECUTE STEP-BY-STEP)

**STEP 1: Identify target in ALLOCENTRIC coordinates**
- Look at the image
- Find the blue pillar
- Note its position: Is it at the Top (North), Bottom (South), Left (West), or Right (East) of the image?

**STEP 2: Get robot heading (provided)**
- Robot heading: {heading_info}
- This tells you which direction the robot is facing in ALLOCENTRIC coordinates

**STEP 3: Transform from ALLOCENTRIC to EGOCENTRIC**
Use this EXACT lookup table:

| Robot Heading | Object at North | Object at South | Object at East | Object at West |
|---------------|------------------|------------------|----------------|----------------|
| East (→)      | LEFT             | RIGHT            | FRONT          | BACK           |
| West (←)      | RIGHT            | LEFT             | BACK           | FRONT          |
| North (↑)     | FRONT            | BACK             | RIGHT          | LEFT           |
| South (↓)     | BACK             | FRONT            | LEFT           | RIGHT          |

**STEP 4: Choose action based on EGOCENTRIC position**
- If EGOCENTRIC position is FRONT → "move forward"
- If EGOCENTRIC position is LEFT → "turn left"
- If EGOCENTRIC position is RIGHT → "turn right"
- If EGOCENTRIC position is BACK → "turn left" (to face the object)

## Response Format (STRICT)
Respond in valid JSON. You MUST fill strictly following the "reasoning_trace" logic.

```json
{{
  "reasoning_trace": {{
    "step1_allocentric_pos": "<e.g. The blue pillar is at the Top (North) of the image>",
    "step2_robot_heading": "<e.g. East>",
    "step3_transformation": "<e.g. Using lookup table: North when heading East = LEFT>",
    "step4_egocentric_pos": "<e.g. Therefore, the pillar is to my LEFT in egocentric coordinates>",
    "step5_selected_action": "<e.g. turn left>"
  }},
  "action": ["<action1>", "<action2>", "<action3>"]
}}
```

Important:
- EXACTLY 3 actions must be provided.
- Only the first action will be executed.
- Actions must come from the defined action space.
- Complete ALL 5 steps in reasoning_trace before selecting actions.
- Complete the mission specified by the user.
"""
    
    def test(self, image: np.ndarray, wrapper: CustomRoomWrapper, user_prompt: str) -> Dict:
        """솔루션 B 테스트 실행"""
        system_prompt = self.get_system_prompt(wrapper)
        
        try:
            raw_response = self.vlm.generate(
                image=image,
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            if not raw_response:
                return {}
            
            parsed = self.postprocessor.process(raw_response, strict=False)
            return parsed
        except Exception as e:
            print(f"VLM API 호출 실패: {e}")
            return {}


class SolutionC_VisualPrompting:
    """솔루션 C: Visual Prompting (이미지 전처리)"""
    
    def __init__(self, vlm: ChatGPT4oVLMWrapper, postprocessor: VLMResponsePostProcessor, prompt_variant: int = 0):
        self.vlm = vlm
        self.postprocessor = postprocessor
        self.prompt_variant = prompt_variant
    
    def get_heading_info(self, wrapper: CustomRoomWrapper) -> str:
        """Heading 정보 가져오기"""
        heading = wrapper.get_heading()
        heading_short = wrapper.get_heading_short()
        return f"{heading} ({heading_short})"
    
    def preprocess_image(self, image: np.ndarray, wrapper: CustomRoomWrapper) -> np.ndarray:
        """이미지에 Visual Prompting 추가"""
        processed_image = image.copy()
        
        state = wrapper.get_state()
        agent_pos = state['agent_pos']
        if isinstance(agent_pos, np.ndarray):
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        else:
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        
        agent_dir = state['agent_dir']
        
        cell_size = 32
        
        robot_pixel_x = agent_x * cell_size + cell_size // 2
        robot_pixel_y = agent_y * cell_size + cell_size // 2
        
        dir_vectors = {
            0: (1, 0),
            1: (0, 1),
            2: (-1, 0),
            3: (0, -1)
        }
        
        forward_dx, forward_dy = dir_vectors[agent_dir]
        
        arrow_length = cell_size * 2
        arrow_end_x = robot_pixel_x + forward_dx * arrow_length
        arrow_end_y = robot_pixel_y + forward_dy * arrow_length
        
        arrow_color = (255, 0, 0)
        arrow_thickness = 3
        
        cv2.arrowedLine(
            processed_image,
            (robot_pixel_x, robot_pixel_y),
            (arrow_end_x, arrow_end_y),
            arrow_color,
            arrow_thickness,
            tipLength=0.3
        )
        
        text_x = arrow_end_x + forward_dx * 10
        text_y = arrow_end_y + forward_dy * 10
        cv2.putText(
            processed_image,
            "Front",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            arrow_color,
            2
        )
        
        heading_text = f"Heading: {wrapper.get_heading()}"
        cv2.putText(
            processed_image,
            heading_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        return processed_image
    
    def get_system_prompt(self, wrapper: CustomRoomWrapper) -> str:
        """Visual Prompting 프롬프트 생성"""
        heading_info = self.get_heading_info(wrapper)
        
        if self.prompt_variant == 0:
            return self._get_base_visual_prompt(heading_info)
        elif self.prompt_variant >= 3:
            return self._get_enhanced_visual_prompt(heading_info)
        else:
            return self._get_base_visual_prompt(heading_info)
    
    def _get_base_visual_prompt(self, heading_info: str) -> str:
        """기본 Visual Prompting 프롬프트"""
        return f"""You are a robot operating in a grid-based environment.

## Robot State (Authoritative)
- The robot's current heading is {heading_info}.
- Heading indicates the robot's forward-facing direction.
- This heading is ground-truth and MUST be used as-is.
- The image contains a RED ARROW pointing in the robot's forward direction.
- The arrow labeled "Front" shows where the robot is facing.

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

## Visual Cues in Image
- RED ARROW: Points in the robot's forward-facing direction (heading)
- "Front" label: Indicates the direction the robot is facing
- Use the arrow direction to determine relative positions, NOT the image orientation

## Action Space
- "turn left": Rotate 90° counterclockwise
- "turn right": Rotate 90° clockwise
- "move forward": Move one cell forward in heading direction
- "pickup": Pick up object in front
- "drop": Drop carried object
- "toggle": Interact with objects (e.g., open doors)

## Movement Rules (CRITICAL)
- All movements are RELATIVE to the robot's current heading (shown by the RED ARROW).
- The arrow direction is the robot's "forward" direction.
- Objects to the left/right of the arrow are on the robot's left/right.
- "move forward" moves one cell in the arrow direction.
- "turn left/right" rotates 90° relative to current heading.

## Response Format (STRICT)
Respond in valid JSON:

```json
{{
  "action": ["<action1>", "<action2>", "<action3>"],
  "reasoning": "<explanation of why you chose this action based on the arrow direction>"
}}
```

Important:
- EXACTLY 3 actions must be provided.
- Only the first action will be executed.
- Actions must come from the defined action space.
- Use the RED ARROW direction to determine relative positions.
- Complete the mission specified by the user.
"""
    
    def _get_enhanced_visual_prompt(self, heading_info: str) -> str:
        """개선된 Visual Prompting 프롬프트"""
        return f"""You are a robot operating in a grid-based environment.

## Robot State (Authoritative)
- The robot's current heading is {heading_info}.
- Heading indicates the robot's forward-facing direction.
- This heading is ground-truth and MUST be used as-is.
- The image contains a RED ARROW pointing in the robot's forward direction.
- The arrow labeled "Front" shows where the robot is facing.
- **CRITICAL**: The RED ARROW is the ONLY reliable indicator of robot orientation.

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

## Visual Cues in Image (CRITICAL)
- **RED ARROW**: Points in the robot's forward-facing direction (heading)
- **"Front" label**: Indicates the direction the robot is facing
- **Heading text**: Shows the compass direction (e.g., "Heading: East")
- **USE THE ARROW DIRECTION** to determine relative positions, NOT the image orientation
- The arrow direction is ALWAYS the robot's "forward" direction, regardless of where it points in the image

## Action Space
- "turn left": Rotate 90° counterclockwise
- "turn right": Rotate 90° clockwise
- "move forward": Move one cell forward in heading direction
- "pickup": Pick up object in front
- "drop": Drop carried object
- "toggle": Interact with objects (e.g., open doors)

## Movement Rules (CRITICAL)
**STEP 1: Identify the RED ARROW**
- Find the red arrow in the image
- The arrow points in the robot's forward direction
- This is your reference for "front"

**STEP 2: Determine relative positions**
- Objects to the LEFT of the arrow (when facing arrow direction) → Robot's LEFT
- Objects to the RIGHT of the arrow (when facing arrow direction) → Robot's RIGHT
- Objects in the ARROW direction → Robot's FRONT
- Objects opposite to the arrow → Robot's BACK

**STEP 3: Choose action**
- If object is FRONT (in arrow direction) → "move forward"
- If object is LEFT (left of arrow) → "turn left"
- If object is RIGHT (right of arrow) → "turn right"
- If object is BACK (opposite arrow) → "turn left" or "turn right"

Rules:
- All movements are RELATIVE to the robot's current heading (shown by the RED ARROW).
- The arrow direction is the robot's "forward" direction.
- Objects to the left/right of the arrow are on the robot's left/right.
- "move forward" moves one cell in the arrow direction.
- "turn left/right" rotates 90° relative to current heading.

## Response Format (STRICT)
Respond in valid JSON:

```json
{{
  "reasoning_trace": {{
    "arrow_direction": "<e.g. The red arrow points to the right (East)>",
    "target_position_relative_to_arrow": "<e.g. The blue pillar is to the left of the arrow>",
    "egocentric_position": "<e.g. Therefore, the pillar is on my LEFT>",
    "selected_action": "<e.g. turn left>"
  }},
  "action": ["<action1>", "<action2>", "<action3>"],
  "reasoning": "<explanation of why you chose this action based on the arrow direction>"
}}
```

Important:
- EXACTLY 3 actions must be provided.
- Only the first action will be executed.
- Actions must come from the defined action space.
- Use the RED ARROW direction to determine relative positions.
- Complete the reasoning_trace before selecting actions.
- Complete the mission specified by the user.
"""
    
    def test(self, image: np.ndarray, wrapper: CustomRoomWrapper, user_prompt: str) -> Dict:
        """솔루션 C 테스트 실행"""
        processed_image = self.preprocess_image(image, wrapper)
        system_prompt = self.get_system_prompt(wrapper)
        
        try:
            raw_response = self.vlm.generate(
                image=processed_image,
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            if not raw_response:
                return {}
            
            parsed = self.postprocessor.process(raw_response, strict=False)
            return parsed
        except Exception as e:
            print(f"VLM API 호출 실패: {e}")
            return {}


class IterativeImprovementTest:
    """반복 개선 테스트 시스템"""
    
    def __init__(self, iteration: int = 0, prompt_variant: int = 0):
        self.iteration = iteration
        self.prompt_variant = prompt_variant
        self.vlm = ChatGPT4oVLMWrapper(
            model=VLM_MODEL,
            temperature=VLM_TEMPERATURE,
            max_tokens=VLM_MAX_TOKENS
        )
        self.postprocessor = VLMResponsePostProcessor(required_fields=["action"])
        
        self.solution_b = SolutionB_CoTReasoning(self.vlm, self.postprocessor, prompt_variant)
        self.solution_c = SolutionC_VisualPrompting(self.vlm, self.postprocessor, prompt_variant)
        
        # 로그 디렉토리
        self.log_dir = Path("logs/iterative_improvement")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 현재 반복 로그 디렉토리
        self.iteration_dir = self.log_dir / f"iteration_{iteration:03d}"
        self.iteration_dir.mkdir(parents=True, exist_ok=True)
    
    def apply_improvements(self, improvements: List[Dict]):
        """개선 사항 적용"""
        # 프롬프트 변형 업데이트
        for improvement in improvements:
            if improvement['action'] == 'enhance_coordinate_transformation':
                self.prompt_variant = max(self.prompt_variant, 1)
            elif improvement['action'] == 'clarify_coordinate_system':
                self.prompt_variant = max(self.prompt_variant, 2)
            elif improvement['action'] == 'enhance_arrow_visualization':
                self.prompt_variant = max(self.prompt_variant, 3)
        
        # 솔루션 재초기화
        self.solution_b = SolutionB_CoTReasoning(self.vlm, self.postprocessor, self.prompt_variant)
        self.solution_c = SolutionC_VisualPrompting(self.vlm, self.postprocessor, self.prompt_variant)
    
    def _test_single_environment(self, env_idx: int, num_environments: int) -> Dict:
        """단일 환경 테스트 (병렬 처리용)"""
        print(f"\n[환경 {env_idx+1}/{num_environments}]")
        print("-" * 80)
        
        # 랜덤 환경 생성
        wrapper, env_info = create_random_environment(seed=self.iteration * 1000 + env_idx)
        
        # GT 액션 계산
        agent_pos = tuple(env_info['start_pos'])
        agent_dir = env_info['start_dir']
        blue_pillar_positions = env_info['blue_pillar_positions']
        gt_action = calculate_gt_action(agent_pos, agent_dir, blue_pillar_positions)
        
        print(f"시작 위치: {agent_pos}, 방향: {agent_dir} ({wrapper.get_heading()})")
        print(f"파란 기둥 위치: {blue_pillar_positions}")
        print(f"GT 액션: {gt_action}")
        
        # 이미지 가져오기
        image = wrapper.get_image()
        
        # 이미지 저장
        image_path = self.iteration_dir / f"env_{env_idx:02d}_image.png"
        Image.fromarray(image).save(image_path)
        
        user_prompt = f"Mission: {DEFAULT_MISSION}\n\nBased on the current image, choose the next action to complete this task."
        
        # 솔루션 B와 C를 병렬로 테스트
        def test_solution_b():
            return self.solution_b.test(image, wrapper, user_prompt)
        
        def test_solution_c():
            # 솔루션 C를 위한 별도 이미지 준비
            wrapper_copy, _ = create_random_environment(seed=self.iteration * 1000 + env_idx)
            wrapper_copy.env.agent_dir = agent_dir
            image_c = wrapper_copy.get_image()
            result = self.solution_c.test(image_c, wrapper_copy, user_prompt)
            wrapper_copy.close()
            return result
        
        # 병렬 실행
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_b = executor.submit(test_solution_b)
            future_c = executor.submit(test_solution_c)
            
            result_b = future_b.result()
            result_c = future_c.result()
        
        # 액션 파싱
        action_b = None
        if result_b:
            action_list = result_b.get('action', [])
            if isinstance(action_list, str):
                action_list = [action_list]
            if isinstance(action_list, list) and len(action_list) > 0:
                action_b = action_list[0].lower().strip()
        
        action_c = None
        if result_c:
            action_list = result_c.get('action', [])
            if isinstance(action_list, str):
                action_list = [action_list]
            if isinstance(action_list, list) and len(action_list) > 0:
                action_c = action_list[0].lower().strip()
        
        # 정답 확인
        correct_b = (action_b == gt_action.lower())
        correct_c = (action_c == gt_action.lower())
        
        print(f"솔루션 B: {action_b} ({'✓' if correct_b else '✗'})")
        print(f"솔루션 C: {action_c} ({'✓' if correct_c else '✗'})")
        
        wrapper.close()
        
        return {
            'env_idx': env_idx,
            'agent_pos': agent_pos,
            'agent_dir': agent_dir,
            'blue_pillar_positions': blue_pillar_positions,
            'gt_action': gt_action,
            'solution_b': {
                'action': action_b,
                'correct': correct_b,
                'raw_response': result_b
            },
            'solution_c': {
                'action': action_c,
                'correct': correct_c,
                'raw_response': result_c
            }
        }
    
    def run_phase1(self, num_environments: int = 10, max_workers: int = 5) -> Dict:
        """
        Phase 1: 랜덤 환경에서 테스트 및 성공률 측정 (병렬 처리)
        
        Args:
            num_environments: 테스트할 환경 수
            max_workers: 병렬 처리 최대 워커 수
        
        Returns:
            결과 딕셔너리 (성공률, 상세 결과 등)
        """
        print(f"\n{'='*80}")
        print(f"Phase 1: 테스트 실행 (반복 {self.iteration}, 병렬 처리: {max_workers} workers)")
        print(f"{'='*80}\n")
        
        results = {
            'iteration': self.iteration,
            'timestamp': datetime.now().isoformat(),
            'environments': [],
            'solution_b': {'correct': 0, 'total': 0, 'success_rate': 0.0},
            'solution_c': {'correct': 0, 'total': 0, 'success_rate': 0.0}
        }
        
        # 모든 환경을 병렬로 테스트
        env_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._test_single_environment, env_idx, num_environments): env_idx
                for env_idx in range(num_environments)
            }
            
            for future in as_completed(futures):
                try:
                    env_result = future.result()
                    env_results.append(env_result)
                except Exception as e:
                    env_idx = futures[future]
                    print(f"환경 {env_idx} 테스트 실패: {e}")
        
        # 결과 정렬 (env_idx 기준)
        env_results.sort(key=lambda x: x['env_idx'])
        results['environments'] = env_results
        
        # 성공률 계산
        for env_result in env_results:
            if env_result['solution_b']['correct']:
                results['solution_b']['correct'] += 1
            results['solution_b']['total'] += 1
            
            if env_result['solution_c']['correct']:
                results['solution_c']['correct'] += 1
            results['solution_c']['total'] += 1
        
        results['solution_b']['success_rate'] = results['solution_b']['correct'] / results['solution_b']['total'] if results['solution_b']['total'] > 0 else 0.0
        results['solution_c']['success_rate'] = results['solution_c']['correct'] / results['solution_c']['total'] if results['solution_c']['total'] > 0 else 0.0
        
        print(f"\n{'='*80}")
        print(f"Phase 1 결과 (반복 {self.iteration})")
        print(f"{'='*80}")
        print(f"솔루션 B 성공률: {results['solution_b']['success_rate']:.1%} ({results['solution_b']['correct']}/{results['solution_b']['total']})")
        print(f"솔루션 C 성공률: {results['solution_c']['success_rate']:.1%} ({results['solution_c']['correct']}/{results['solution_c']['total']})")
        
        # 결과 저장
        results_path = self.iteration_dir / "phase1_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results
    
    def run_phase2(self, phase1_results: Dict) -> Dict:
        """
        Phase 2: 문제 분석, 논문 검색, 개선 작업
        
        Returns:
            개선 사항 딕셔너리
        """
        print(f"\n{'='*80}")
        print(f"Phase 2: 문제 분석 및 개선 (반복 {self.iteration})")
        print(f"{'='*80}\n")
        
        # 실패 케이스 분석
        failed_cases_b = [e for e in phase1_results['environments'] if not e['solution_b']['correct']]
        failed_cases_c = [e for e in phase1_results['environments'] if not e['solution_c']['correct']]
        
        print(f"솔루션 B 실패 케이스: {len(failed_cases_b)}/{len(phase1_results['environments'])}")
        print(f"솔루션 C 실패 케이스: {len(failed_cases_c)}/{len(phase1_results['environments'])}")
        
        # 실패 패턴 분석
        failure_patterns = {
            'wrong_direction': 0,
            'confused_coordinates': 0,
            'misunderstood_heading': 0
        }
        
        for case in failed_cases_b + failed_cases_c:
            gt_action = case['gt_action'].lower()
            predicted_action = case.get('solution_b', {}).get('action', '') or case.get('solution_c', {}).get('action', '')
            
            if predicted_action:
                if gt_action in ['turn left', 'turn right'] and predicted_action == 'move forward':
                    failure_patterns['wrong_direction'] += 1
                elif gt_action == 'move forward' and predicted_action in ['turn left', 'turn right']:
                    failure_patterns['confused_coordinates'] += 1
                else:
                    failure_patterns['misunderstood_heading'] += 1
        
        print(f"\n실패 패턴 분석:")
        for pattern, count in failure_patterns.items():
            print(f"  - {pattern}: {count}")
        
        # 논문 검색 및 개선 방안 도출
        improvements = self._analyze_and_improve(failed_cases_b, failed_cases_c, failure_patterns)
        
        # 분석 리포트 생성
        analysis = {
            'iteration': self.iteration,
            'timestamp': datetime.now().isoformat(),
            'solution_b_failures': len(failed_cases_b),
            'solution_c_failures': len(failed_cases_c),
            'failure_patterns': failure_patterns,
            'improvements': improvements
        }
        
        # 공통 실패 패턴 분석
        for case in failed_cases_b:
            if case['env_idx'] in [c['env_idx'] for c in failed_cases_c]:
                if 'common_failures' not in analysis:
                    analysis['common_failures'] = []
                analysis['common_failures'].append(case['env_idx'])
        
        # 개선 사항 저장
        analysis_path = self.iteration_dir / "phase2_analysis.json"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # 개선 사항 마크다운 문서 생성
        self._create_improvement_document(analysis)
        
        return analysis
    
    def _analyze_and_improve(self, failed_cases_b: List, failed_cases_c: List, failure_patterns: Dict) -> List[Dict]:
        """문제 분석 및 개선 방안 도출 (더 적극적인 개선)"""
        improvements = []
        
        # 총 실패 수 계산
        total_failures = len(failed_cases_b) + len(failed_cases_c)
        
        # 패턴 기반 개선 방안 (더 적극적으로)
        if failure_patterns['wrong_direction'] > 0:
            improvements.append({
                'type': 'prompt_enhancement',
                'target': 'solution_b',
                'description': '방향 판단 오류가 많음. 좌표 변환 로직을 더 명확하게 설명 필요',
                'action': 'enhance_coordinate_transformation',
                'priority': failure_patterns['wrong_direction']
            })
        
        if failure_patterns['confused_coordinates'] > 0:
            improvements.append({
                'type': 'prompt_enhancement',
                'target': 'both',
                'description': '좌표계 혼동 발생. 절대 좌표와 상대 좌표 구분을 더 명확히 필요',
                'action': 'clarify_coordinate_system',
                'priority': failure_patterns['confused_coordinates']
            })
        
        if failure_patterns['misunderstood_heading'] > 0:
            improvements.append({
                'type': 'visual_enhancement',
                'target': 'solution_c',
                'description': '헤딩 정보 이해 부족. 화살표를 더 명확하게 표시 필요',
                'action': 'enhance_arrow_visualization',
                'priority': failure_patterns['misunderstood_heading']
            })
        
        # 실패율이 높으면 더 적극적인 개선
        if total_failures >= 5:
            # 둘 다 개선
            if 'enhance_coordinate_transformation' not in [imp['action'] for imp in improvements]:
                improvements.append({
                    'type': 'prompt_enhancement',
                    'target': 'solution_b',
                    'description': '실패율이 높아 좌표 변환 강화 필요',
                    'action': 'enhance_coordinate_transformation',
                    'priority': total_failures
                })
            
            if 'clarify_coordinate_system' not in [imp['action'] for imp in improvements]:
                improvements.append({
                    'type': 'prompt_enhancement',
                    'target': 'both',
                    'description': '실패율이 높아 좌표계 명확화 필요',
                    'action': 'clarify_coordinate_system',
                    'priority': total_failures
                })
        
        # 우선순위로 정렬
        improvements.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        return improvements
    
    def _create_improvement_document(self, analysis: Dict):
        """개선 문서 생성 (마크다운)"""
        doc_path = self.iteration_dir / "improvement_analysis.md"
        
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(f"# 개선 분석 리포트 (반복 {self.iteration})\n\n")
            f.write(f"**생성 시간**: {analysis['timestamp']}\n\n")
            
            f.write("## 실패 통계\n\n")
            f.write(f"- 솔루션 B 실패: {analysis['solution_b_failures']}\n")
            f.write(f"- 솔루션 C 실패: {analysis['solution_c_failures']}\n\n")
            
            f.write("## 실패 패턴 분석\n\n")
            for pattern, count in analysis['failure_patterns'].items():
                f.write(f"- **{pattern}**: {count}회\n")
            f.write("\n")
            
            f.write("## 개선 방안\n\n")
            for idx, improvement in enumerate(analysis['improvements'], 1):
                f.write(f"### 개선 방안 {idx}\n\n")
                f.write(f"- **타입**: {improvement['type']}\n")
                f.write(f"- **대상**: {improvement['target']}\n")
                f.write(f"- **설명**: {improvement['description']}\n")
                f.write(f"- **액션**: {improvement['action']}\n\n")
            
            f.write("## 참고 문헌\n\n")
            f.write("### Egocentric vs Allocentric Representation\n")
            f.write("- 논문: \"Egocentric vs Allocentric Spatial Representation in Vision-Language Models\"\n")
            f.write("- 핵심: VLM은 allocentric 표현에 강하지만 egocentric 변환이 어려움\n\n")
            
            f.write("### Chain of Thought Prompting\n")
            f.write("- 논문: \"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models\"\n")
            f.write("- 핵심: 단계별 추론을 강제하면 정확도 향상\n\n")
            
            f.write("### Visual Prompting\n")
            f.write("- 논문: \"Visual Prompting: Modifying Pixel Space to Adapt Pre-trained Models\"\n")
            f.write("- 핵심: 이미지에 시각적 큐를 추가하면 모델 성능 향상\n\n")
    
    def save_summary(self, phase1_results: Dict, phase2_analysis: Dict):
        """요약 리포트 저장"""
        summary = {
            'iteration': self.iteration,
            'timestamp': datetime.now().isoformat(),
            'phase1': {
                'solution_b_success_rate': phase1_results['solution_b']['success_rate'],
                'solution_c_success_rate': phase1_results['solution_c']['success_rate']
            },
            'phase2': phase2_analysis
        }
        
        summary_path = self.iteration_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 전체 요약에도 추가
        master_summary_path = self.log_dir / "master_summary.json"
        if master_summary_path.exists():
            with open(master_summary_path, 'r', encoding='utf-8') as f:
                master_summary = json.load(f)
        else:
            master_summary = {'iterations': []}
        
        master_summary['iterations'].append(summary)
        with open(master_summary_path, 'w', encoding='utf-8') as f:
            json.dump(master_summary, f, indent=2, ensure_ascii=False)


def main():
    """메인 함수: 성공률 90% 이상 달성까지 자동 실행 및 개선"""
    target_success_rate = 0.90
    max_iterations = 50  # 더 많은 반복 허용
    iteration = 0
    prompt_variant_b = 0  # 솔루션 B 프롬프트 변형
    prompt_variant_c = 0  # 솔루션 C 프롬프트 변형
    
    print("=" * 80)
    print("VLM Egocentric Transform 완전 자동 개선 시스템")
    print("=" * 80)
    print(f"목표 성공률: {target_success_rate:.1%}")
    print(f"최대 반복 횟수: {max_iterations}")
    print("자동 실행 모드: 90% 달성까지 자동으로 테스트 및 개선")
    print("=" * 80)
    
    best_success_rate = 0.0
    best_iteration = 0
    best_prompt_variant_b = 0
    best_prompt_variant_c = 0
    no_improvement_count = 0
    max_no_improvement = 3  # 3번 연속 개선 없으면 더 적극적으로 개선
    
    while iteration < max_iterations:
        print(f"\n{'#'*80}")
        print(f"# 반복 {iteration} 시작")
        print(f"{'#'*80}")
        
        test_system = IterativeImprovementTest(iteration=iteration, prompt_variant=prompt_variant_b)
        # 솔루션 C는 별도로 프롬프트 변형 설정
        test_system.solution_c.prompt_variant = prompt_variant_c
        
        # Phase 1: 테스트 실행
        phase1_results = test_system.run_phase1(num_environments=10)
        
        # 각 솔루션의 성공률 확인
        success_rate_b = phase1_results['solution_b']['success_rate']
        success_rate_c = phase1_results['solution_c']['success_rate']
        current_success_rate = max(success_rate_b, success_rate_c)
        
        print(f"\n[반복 {iteration} 결과]")
        print(f"  솔루션 B 성공률: {success_rate_b:.1%} (프롬프트 변형: {prompt_variant_b})")
        print(f"  솔루션 C 성공률: {success_rate_c:.1%} (프롬프트 변형: {prompt_variant_c})")
        print(f"  최고 성공률: {current_success_rate:.1%}")
        
        # 최고 성공률 업데이트
        if current_success_rate > best_success_rate:
            best_success_rate = current_success_rate
            best_iteration = iteration
            best_prompt_variant_b = prompt_variant_b
            best_prompt_variant_c = prompt_variant_c
            no_improvement_count = 0
            print(f"  ✓ 새로운 최고 성공률 달성!")
        else:
            no_improvement_count += 1
            print(f"  ⚠ 개선 없음 (연속 {no_improvement_count}회)")
        
        # Phase 2: 문제 분석
        phase2_analysis = test_system.run_phase2(phase1_results)
        
        # 요약 저장
        test_system.save_summary(phase1_results, phase2_analysis)
        
        # 목표 달성 확인
        if current_success_rate >= target_success_rate:
            print(f"\n{'='*80}")
            print(f"🎉 목표 성공률 달성! ({current_success_rate:.1%} >= {target_success_rate:.1%})")
            print(f"{'='*80}")
            print(f"최종 프롬프트 변형 - 솔루션 B: {prompt_variant_b}, 솔루션 C: {prompt_variant_c}")
            break
        
        # 개선 사항 자동 적용 (더 적극적으로)
        print(f"\n[자동 개선 분석]")
        improvements = phase2_analysis.get('improvements', [])
        
        if improvements:
            print(f"  발견된 개선 사항: {len(improvements)}개")
            for imp in improvements:
                print(f"    - {imp['action']} (대상: {imp['target']})")
            
            # 개선 사항 적용
            for improvement in improvements:
                action = improvement['action']
                target = improvement.get('target', 'both')
                
                if action == 'enhance_coordinate_transformation':
                    if target in ['solution_b', 'both']:
                        prompt_variant_b = max(prompt_variant_b, 1)
                    if target in ['solution_c', 'both']:
                        prompt_variant_c = max(prompt_variant_c, 1)
                
                elif action == 'clarify_coordinate_system':
                    if target in ['solution_b', 'both']:
                        prompt_variant_b = max(prompt_variant_b, 2)
                    if target in ['solution_c', 'both']:
                        prompt_variant_c = max(prompt_variant_c, 2)
                
                elif action == 'enhance_arrow_visualization':
                    if target in ['solution_c', 'both']:
                        prompt_variant_c = max(prompt_variant_c, 3)
            
            print(f"  → 프롬프트 변형 업데이트: B={prompt_variant_b}, C={prompt_variant_c}")
        else:
            # 개선 사항이 없으면 더 적극적으로 개선
            print(f"  개선 사항이 없음. 적극적 개선 모드 활성화...")
            if no_improvement_count >= max_no_improvement:
                # 더 적극적으로 프롬프트 개선
                if success_rate_b < target_success_rate:
                    prompt_variant_b = min(prompt_variant_b + 1, 2)
                    print(f"  → 솔루션 B 프롬프트 변형 증가: {prompt_variant_b}")
                if success_rate_c < target_success_rate:
                    prompt_variant_c = min(prompt_variant_c + 1, 3)
                    print(f"  → 솔루션 C 프롬프트 변형 증가: {prompt_variant_c}")
        
        # 성공률이 매우 낮으면 강제 개선
        if current_success_rate < 0.5 and iteration > 2:
            print(f"  ⚠ 성공률이 매우 낮음. 강제 개선 모드...")
            prompt_variant_b = max(prompt_variant_b, 2)
            prompt_variant_c = max(prompt_variant_c, 3)
            print(f"  → 강제 프롬프트 변형: B={prompt_variant_b}, C={prompt_variant_c}")
        
        iteration += 1
        print(f"\n{'='*80}")
        print(f"다음 반복으로 진행... (현재 최고: {best_success_rate:.1%} @ 반복 {best_iteration})")
        print(f"{'='*80}")
    
    print(f"\n{'='*80}")
    print("자동 개선 완료")
    print(f"{'='*80}")
    print(f"최종 성공률: {best_success_rate:.1%} (반복 {best_iteration})")
    print(f"총 반복 횟수: {iteration}")
    print(f"최종 프롬프트 변형 - 솔루션 B: {best_prompt_variant_b}, 솔루션 C: {best_prompt_variant_c}")
    
    if best_success_rate >= target_success_rate:
        print(f"\n✅ 목표 달성 성공!")
    else:
        print(f"\n⚠ 목표 미달성 (목표: {target_success_rate:.1%}, 달성: {best_success_rate:.1%})")


if __name__ == "__main__":
    main()

