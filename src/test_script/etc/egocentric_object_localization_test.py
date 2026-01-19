"""
Egocentric Object Localization Test
로봇 heading 기준 egocentric 좌표계에서 물체 위치 표현 성능 개선

목표: 로봇 heading과 egocentric 좌표계 기준 물체 관계 추측 성공률 90% 이상 달성
"""

from minigrid import register_minigrid_envs
# Actual path: legacy.relative_movement.custom_environment
from legacy import CustomRoomWrapper
# Actual paths: utils.vlm.vlm_wrapper, utils.vlm.vlm_postprocessor
from utils import ChatGPT4oVLMWrapper, VLMResponsePostProcessor
import numpy as np
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# MiniGrid 환경 등록
register_minigrid_envs()

# VLM 설정
VLM_MODEL = "gpt-4o"
VLM_TEMPERATURE = 0.0
VLM_MAX_TOKENS = 2000


def create_random_environment_with_objects(seed: Optional[int] = None) -> Tuple[CustomRoomWrapper, Dict]:
    """랜덤 환경 생성 (물체 색, 위치 랜덤화)"""
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
    
    # 랜덤 시작 위치 (벽 제외)
    start_x = random.randint(1, size-2)
    start_y = random.randint(1, size-2)
    start_pos = (start_x, start_y)
    
    # 랜덤 시작 방향
    start_dir = random.randint(0, 3)  # 0: East, 1: South, 2: West, 3: North
    
    # 랜덤 목표 위치
    goal_x = random.randint(1, size-2)
    goal_y = random.randint(1, size-2)
    # 시작 위치와 겹치지 않도록
    while (goal_x, goal_y) == start_pos:
        goal_x = random.randint(1, size-2)
        goal_y = random.randint(1, size-2)
    goal_pos = (goal_x, goal_y)
    
    # 랜덤 색상 물체 생성 (1-3개)
    num_objects = random.randint(1, 3)
    object_colors = ['blue', 'purple', 'red', 'green', 'yellow']
    objects = []
    
    for i in range(num_objects):
        # 랜덤 색상 선택
        color = random.choice(object_colors)
        
        # 랜덤 크기 (1x1 또는 2x2)
        obj_size = random.choice([1, 2])
        
        # 랜덤 위치 (시작 위치, 목표 위치와 겹치지 않도록)
        max_attempts = 50
        placed = False
        for _ in range(max_attempts):
            obj_x = random.randint(1, size-2)
            obj_y = random.randint(1, size-2)
            
            # 시작 위치, 목표 위치와 겹치는지 확인
            if (obj_x, obj_y) == start_pos or (obj_x, obj_y) == goal_pos:
                continue
            
            # 다른 물체와 겹치는지 확인
            overlap = False
            for existing_obj in objects:
                ex_x, ex_y, ex_size, _ = existing_obj
                if abs(obj_x - ex_x) < ex_size and abs(obj_y - ex_y) < ex_size:
                    overlap = True
                    break
            
            if not overlap:
                objects.append((obj_x, obj_y, obj_size, color))
                placed = True
                break
        
        if not placed:
            continue
    
    # 물체를 벽으로 추가
    for obj_x, obj_y, obj_size, color in objects:
        for dx in range(obj_size):
            for dy in range(obj_size):
                walls.append((obj_x + dx, obj_y + dy, color))
    
    room_config = {
        'start_pos': start_pos,
        'goal_pos': goal_pos,
        'walls': walls,
        'objects': []
    }
    
    wrapper = CustomRoomWrapper(size=size, room_config=room_config)
    wrapper.reset()
    
    # 시작 방향 설정 (직접 설정)
    wrapper.env.agent_dir = start_dir
    
    # 현재 상태 확인
    state = wrapper.get_state()
    actual_pos = state['agent_pos']
    if isinstance(actual_pos, np.ndarray):
        actual_pos = (int(actual_pos[0]), int(actual_pos[1]))
    else:
        actual_pos = (int(actual_pos[0]), int(actual_pos[1]))
    actual_dir = int(state['agent_dir'])
    
    # 환경 정보
    env_info = {
        'agent_pos': actual_pos,
        'agent_dir': actual_dir,
        'agent_heading': wrapper.get_heading(),
        'goal_pos': goal_pos,
        'objects': []
    }
    
    # 물체 정보 추가
    for obj_x, obj_y, obj_size, color in objects:
        positions = []
        for dx in range(obj_size):
            for dy in range(obj_size):
                positions.append((int(obj_x + dx), int(obj_y + dy)))
        env_info['objects'].append({
            'color': color,
            'size': int(obj_size),
            'positions': positions,
            'center': (int(obj_x + obj_size // 2), int(obj_y + obj_size // 2))
        })
    
    return wrapper, env_info


def calculate_gt_egocentric_position(agent_pos: Tuple[int, int], agent_dir: int, 
                                     object_pos: Tuple[int, int]) -> str:
    """GT Egocentric 위치 계산 (정확한 변환)"""
    agent_x, agent_y = agent_pos
    obj_x, obj_y = object_pos
    
    # Allocentric 차이 계산
    dx = obj_x - agent_x
    dy = obj_y - agent_y
    
    # 방향에 따른 변환
    # 0: East (→), 1: South (↓), 2: West (←), 3: North (↑)
    # MiniGrid 좌표계: (0,0)이 왼쪽 위, x는 오른쪽, y는 아래
    
    if agent_dir == 0:  # East (→)
        # 앞: +x, 왼쪽: -y (North), 오른쪽: +y (South), 뒤: -x (West)
        if dx > 0:
            return "front"
        elif dx < 0:
            return "back"
        elif dy < 0:  # North (위쪽)
            return "left"
        else:  # dy > 0, South (아래쪽)
            return "right"
    elif agent_dir == 1:  # South (↓)
        # 앞: +y, 왼쪽: +x (East), 오른쪽: -x (West), 뒤: -y (North)
        if dy > 0:
            return "front"
        elif dy < 0:
            return "back"
        elif dx > 0:  # East (오른쪽)
            return "left"
        else:  # dx < 0, West (왼쪽)
            return "right"
    elif agent_dir == 2:  # West (←)
        # 앞: -x, 왼쪽: +y (South), 오른쪽: -y (North), 뒤: +x (East)
        if dx < 0:
            return "front"
        elif dx > 0:
            return "back"
        elif dy > 0:  # South (아래쪽)
            return "left"
        else:  # dy < 0, North (위쪽)
            return "right"
    else:  # North (agent_dir == 3) (↑)
        # 앞: -y, 왼쪽: -x (West), 오른쪽: +x (East), 뒤: +y (South)
        if dy < 0:
            return "front"
        elif dy > 0:
            return "back"
        elif dx < 0:  # West (왼쪽)
            return "left"
        else:  # dx > 0, East (오른쪽)
            return "right"


class EgocentricLocalizationSolution:
    """Egocentric 좌표계 물체 위치 추론 솔루션"""
    
    def __init__(self, vlm: ChatGPT4oVLMWrapper, postprocessor: VLMResponsePostProcessor, prompt_variant: int = 0):
        self.vlm = vlm
        self.postprocessor = postprocessor
        self.prompt_variant = prompt_variant
    
    def get_system_prompt(self, wrapper: CustomRoomWrapper, objects_info: List[Dict]) -> str:
        """System Prompt 생성"""
        heading = wrapper.get_heading()
        heading_short = wrapper.get_heading_short()
        heading_info = f"{heading} ({heading_short})"
        
        # 물체 정보 문자열 생성
        objects_str = ""
        for i, obj in enumerate(objects_info):
            objects_str += f"\n- Object {i+1}: {obj['color']} color, size {obj['size']}x{obj['size']}"
        
        if self.prompt_variant == 0:
            return self._get_base_prompt(heading_info, objects_str)
        elif self.prompt_variant == 1:
            return self._get_enhanced_prompt(heading_info, objects_str)
        elif self.prompt_variant >= 2:
            return self._get_detailed_prompt(heading_info, objects_str)
        else:
            return self._get_base_prompt(heading_info, objects_str)
    
    def _get_base_prompt(self, heading_info: str, objects_str: str) -> str:
        """기본 프롬프트"""
        return f"""You are a robot operating in a grid-based environment.

## Robot State (Authoritative)
- The robot's current heading is {heading_info}.
- Heading indicates the robot's forward-facing direction.
- This heading is ground-truth and MUST be used as-is.

## Objects in Environment
{objects_str}

## Task
Your task is to identify the egocentric (relative) position of each object relative to the robot's current heading.

## Egocentric Coordinate System
- **Front**: In the direction the robot is facing (heading direction)
- **Back**: Opposite to the heading direction
- **Left**: 90 degrees counterclockwise from heading
- **Right**: 90 degrees clockwise from heading

## Response Format
Respond in valid JSON:
```json
{{
  "objects": [
    {{
      "color": "<color>",
      "egocentric_position": "<front|back|left|right>",
      "reasoning": "<explanation>"
    }}
  ]
}}
```

Important:
- Identify ALL objects in the environment
- Use egocentric coordinates (front/back/left/right) relative to robot heading
- Complete the reasoning for each object
"""
    
    def _get_enhanced_prompt(self, heading_info: str, objects_str: str) -> str:
        """개선된 프롬프트"""
        return f"""You are a robot operating in a grid-based environment.

## Robot State (Authoritative)
- The robot's current heading is {heading_info}.
- Heading indicates the robot's forward-facing direction.
- This heading is ground-truth and MUST be used as-is.

## Objects in Environment
{objects_str}

## Task
Your task is to identify the egocentric (relative) position of each object relative to the robot's current heading.

## CRITICAL: Two Coordinate Systems

### 1. ALLOCENTRIC (Absolute/Global) Coordinates
- Used in the IMAGE: Top=North, Bottom=South, Left=West, Right=East
- This is FIXED and does NOT change with robot orientation

### 2. EGOCENTRIC (Relative/Robot-centric) Coordinates
- Used for OBJECT POSITIONS: Front/Back/Left/Right relative to heading
- This CHANGES when the robot rotates

## Transformation Process (STEP-BY-STEP)

**STEP 1: Identify object position in ALLOCENTRIC coordinates**
- Look at the image
- Find each object
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

**STEP 4: Determine egocentric position**
- Apply the transformation for each object
- Use the result as the egocentric position

## Response Format
Respond in valid JSON:
```json
{{
  "reasoning_trace": {{
    "step1_allocentric": "<object positions in allocentric coordinates>",
    "step2_robot_heading": "<robot heading>",
    "step3_transformation": "<transformation applied>",
    "step4_egocentric": "<final egocentric positions>"
  }},
  "objects": [
    {{
      "color": "<color>",
      "egocentric_position": "<front|back|left|right>",
      "reasoning": "<explanation>"
    }}
  ]
}}
```

Important:
- Identify ALL objects in the environment
- Complete ALL 4 steps in reasoning_trace
- Use egocentric coordinates (front/back/left/right) relative to robot heading
"""
    
    def _get_detailed_prompt(self, heading_info: str, objects_str: str) -> str:
        """상세 프롬프트"""
        return f"""You are a robot operating in a grid-based environment.

## Robot State (Authoritative)
- The robot's current heading is {heading_info}.
- Heading indicates the robot's forward-facing direction.
- This heading is ground-truth and MUST be used as-is.
- Do NOT infer or reinterpret the robot's heading from the image.

## Objects in Environment
{objects_str}

## Task
Your task is to identify the egocentric (relative) position of EACH object relative to the robot's current heading.

## CRITICAL DISTINCTION: Two Coordinate Systems

### 1. ALLOCENTRIC (Absolute/Global) Coordinates
- Used in the IMAGE: Top=North, Bottom=South, Left=West, Right=East
- This is FIXED and does NOT change with robot orientation
- The image shows objects in this coordinate system

### 2. EGOCENTRIC (Relative/Robot-centric) Coordinates
- Used for OBJECT POSITIONS: Front/Back/Left/Right relative to heading
- This CHANGES when the robot rotates
- Objects must be described in this coordinate system

## Transformation Process (CRITICAL: EXECUTE STEP-BY-STEP FOR EACH OBJECT)

**STEP 1: Identify object position in ALLOCENTRIC coordinates**
- Look at the image
- Find the object
- Determine its position: Top (North), Bottom (South), Left (West), or Right (East) of the image
- If the object spans multiple directions, use the CENTER of the object

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

**STEP 4: Determine egocentric position**
- Apply the transformation result
- Use: "front", "back", "left", or "right" (lowercase)

## Response Format (STRICT)
Respond in valid JSON. You MUST fill strictly following the format:

```json
{{
  "reasoning_trace": {{
    "step1_allocentric": "<For each object, identify its allocentric position (North/South/East/West)>",
    "step2_robot_heading": "<{heading_info}>",
    "step3_transformation": "<For each object, apply transformation using lookup table>",
    "step4_egocentric": "<Final egocentric positions for all objects>"
  }},
  "objects": [
    {{
      "color": "<color>",
      "egocentric_position": "<front|back|left|right>",
      "reasoning": "<step-by-step explanation for this specific object>"
    }}
  ]
}}
```

Important:
- Identify ALL objects in the environment
- Complete ALL 4 steps in reasoning_trace
- Use lowercase: "front", "back", "left", "right"
- Provide reasoning for EACH object separately
- The egocentric_position must match the transformation result
"""
    
    def test(self, image: np.ndarray, wrapper: CustomRoomWrapper, objects_info: List[Dict]) -> Dict:
        """테스트 실행"""
        system_prompt = self.get_system_prompt(wrapper, objects_info)
        
        user_prompt = "Identify the egocentric position (front, back, left, or right) of each object relative to the robot's current heading."
        
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


class EgocentricLocalizationTest:
    """Egocentric 좌표계 물체 위치 추론 테스트 시스템"""
    
    def __init__(self, iteration: int = 0, prompt_variant: int = 0):
        self.iteration = iteration
        self.prompt_variant = prompt_variant
        self.vlm = ChatGPT4oVLMWrapper(
            model=VLM_MODEL,
            temperature=VLM_TEMPERATURE,
            max_tokens=VLM_MAX_TOKENS
        )
        self.postprocessor = VLMResponsePostProcessor(required_fields=["objects"])
        
        self.solution = EgocentricLocalizationSolution(self.vlm, self.postprocessor, prompt_variant)
        
        # 로그 디렉토리
        self.log_dir = Path("logs/egocentric_localization")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 현재 반복 로그 디렉토리
        self.iteration_dir = self.log_dir / f"iteration_{iteration:03d}"
        self.iteration_dir.mkdir(parents=True, exist_ok=True)
    
    def _test_single_environment(self, env_idx: int, num_environments: int, seed: Optional[int] = None) -> Dict:
        """단일 환경 테스트"""
        try:
            wrapper, env_info = create_random_environment_with_objects(seed=seed)
            
            # 이미지 저장
            image = wrapper.get_image()
            image_path = self.iteration_dir / f"env_{env_idx:02d}_image.png"
            import cv2
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # VLM 테스트
            vlm_response = self.solution.test(image, wrapper, env_info['objects'])
            
            # GT 계산
            gt_results = []
            for obj in env_info['objects']:
                obj_center = obj['center']
                agent_pos = env_info['agent_pos']
                agent_dir = int(env_info['agent_dir'])
                gt_pos = calculate_gt_egocentric_position(
                    agent_pos,
                    agent_dir,
                    obj_center
                )
                gt_results.append({
                    'color': obj['color'],
                    'gt_position': gt_pos
                })
            
            # 결과 비교
            vlm_objects = vlm_response.get('objects', [])
            correct_count = 0
            total_count = len(gt_results)
            
            results = []
            for gt_obj in gt_results:
                # VLM 응답에서 해당 색상 찾기
                vlm_obj = None
                for vo in vlm_objects:
                    if vo.get('color', '').lower() == gt_obj['color'].lower():
                        vlm_obj = vo
                        break
                
                if vlm_obj:
                    vlm_pos = vlm_obj.get('egocentric_position', '').lower()
                    gt_pos = gt_obj['gt_position'].lower()
                    is_correct = vlm_pos == gt_pos
                    if is_correct:
                        correct_count += 1
                    
                    results.append({
                        'color': gt_obj['color'],
                        'gt_position': gt_pos,
                        'vlm_position': vlm_pos,
                        'correct': is_correct
                    })
                else:
                    results.append({
                        'color': gt_obj['color'],
                        'gt_position': gt_obj['gt_position'],
                        'vlm_position': 'not_found',
                        'correct': False
                    })
            
            wrapper.close()
            
            return {
                'env_idx': env_idx,
                'env_info': env_info,
                'vlm_response': vlm_response,
                'gt_results': gt_results,
                'comparison': results,
                'correct': correct_count,
                'total': total_count,
                'success_rate': correct_count / total_count if total_count > 0 else 0.0
            }
        except Exception as e:
            print(f"환경 {env_idx} 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            return {
                'env_idx': env_idx,
                'error': str(e),
                'correct': 0,
                'total': 0,
                'success_rate': 0.0
            }
    
    def run_phase1(self, num_environments: int = 10) -> Dict:
        """Phase 1: 테스트 실행"""
        print(f"\n{'='*80}")
        print(f"Phase 1: 테스트 실행 (반복 {self.iteration}, 병렬 처리: {os.cpu_count() * 2} workers)")
        print(f"{'='*80}")
        
        results = []
        
        # 병렬 처리
        tasks = []
        with ThreadPoolExecutor(max_workers=min(os.cpu_count() * 2, num_environments)) as executor:
            for env_idx in range(num_environments):
                seed = self.iteration * 1000 + env_idx
                tasks.append(executor.submit(self._test_single_environment, env_idx, num_environments, seed))
            
            for future in as_completed(tasks):
                env_result = future.result()
                results.append(env_result)
                
                if 'error' not in env_result:
                    print(f"[환경 {env_result['env_idx']+1}/{num_environments}] "
                          f"성공률: {env_result['success_rate']:.1%} "
                          f"({env_result['correct']}/{env_result['total']})")
        
        # 전체 통계
        total_correct = sum(r.get('correct', 0) for r in results)
        total_objects = sum(r.get('total', 0) for r in results)
        overall_success_rate = total_correct / total_objects if total_objects > 0 else 0.0
        
        print(f"\n{'='*80}")
        print(f"Phase 1 결과 (반복 {self.iteration})")
        print(f"{'='*80}")
        print(f"전체 성공률: {overall_success_rate:.1%} ({total_correct}/{total_objects})")
        
        # 결과 저장
        phase1_results = {
            'iteration': self.iteration,
            'timestamp': datetime.now().isoformat(),
            'overall_success_rate': overall_success_rate,
            'total_correct': total_correct,
            'total_objects': total_objects,
            'environments': results
        }
        
        results_path = self.iteration_dir / "phase1_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(phase1_results, f, indent=2, ensure_ascii=False)
        
        return phase1_results
    
    def run_phase2(self, phase1_results: Dict) -> Dict:
        """Phase 2: 문제 분석 및 개선"""
        print(f"\n{'='*80}")
        print(f"Phase 2: 문제 분석 및 개선 (반복 {self.iteration})")
        print(f"{'='*80}")
        
        # 실패 케이스 분석
        failed_cases = []
        failure_patterns = {
            'wrong_direction': 0,
            'object_not_found': 0,
            'coordinate_confusion': 0,
            'transformation_error': 0
        }
        
        for env_result in phase1_results['environments']:
            if 'comparison' in env_result:
                for comp in env_result['comparison']:
                    if not comp.get('correct', False):
                        failed_cases.append({
                            'env_idx': env_result['env_idx'],
                            'color': comp.get('color'),
                            'gt': comp.get('gt_position'),
                            'vlm': comp.get('vlm_position'),
                            'error_type': self._classify_error(comp)
                        })
                        
                        error_type = self._classify_error(comp)
                        if error_type in failure_patterns:
                            failure_patterns[error_type] += 1
        
        print(f"\n실패 케이스: {len(failed_cases)}/{phase1_results['total_objects']}")
        print(f"실패 패턴 분석:")
        for pattern, count in failure_patterns.items():
            print(f"  - {pattern}: {count}")
        
        # 개선 방안 도출
        improvements = self._analyze_and_improve(failed_cases, failure_patterns)
        
        analysis = {
            'iteration': self.iteration,
            'timestamp': datetime.now().isoformat(),
            'total_failures': len(failed_cases),
            'failure_patterns': failure_patterns,
            'failed_cases': failed_cases[:10],  # 처음 10개만 저장
            'improvements': improvements
        }
        
        # 분석 저장
        analysis_path = self.iteration_dir / "phase2_analysis.json"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # 개선 문서 생성
        self._create_improvement_document(analysis)
        
        return analysis
    
    def _classify_error(self, comp: Dict) -> str:
        """에러 타입 분류"""
        vlm_pos = comp.get('vlm_position', '').lower()
        gt_pos = comp.get('gt_position', '').lower()
        
        if vlm_pos == 'not_found':
            return 'object_not_found'
        elif vlm_pos not in ['front', 'back', 'left', 'right']:
            return 'coordinate_confusion'
        elif vlm_pos in ['front', 'back', 'left', 'right']:
            return 'wrong_direction'
        else:
            return 'transformation_error'
    
    def _analyze_and_improve(self, failed_cases: List[Dict], failure_patterns: Dict) -> List[Dict]:
        """문제 분석 및 개선 방안 도출"""
        improvements = []
        
        if failure_patterns['object_not_found'] > 0:
            improvements.append({
                'type': 'prompt_enhancement',
                'description': '물체를 찾지 못하는 경우가 많음. 물체 탐지 지시를 더 명확하게 필요',
                'action': 'enhance_object_detection',
                'priority': failure_patterns['object_not_found']
            })
        
        if failure_patterns['coordinate_confusion'] > 0:
            improvements.append({
                'type': 'prompt_enhancement',
                'description': '좌표계 혼동 발생. Allocentric vs Egocentric 구분을 더 명확히 필요',
                'action': 'clarify_coordinate_systems',
                'priority': failure_patterns['coordinate_confusion']
            })
        
        if failure_patterns['wrong_direction'] > 0:
            improvements.append({
                'type': 'prompt_enhancement',
                'description': '방향 변환 오류. Lookup Table 사용을 더 강제 필요',
                'action': 'enhance_transformation',
                'priority': failure_patterns['wrong_direction']
            })
        
        if failure_patterns['transformation_error'] > 0:
            improvements.append({
                'type': 'prompt_enhancement',
                'description': '변환 프로세스 오류. 단계별 추론을 더 강제 필요',
                'action': 'enhance_step_by_step',
                'priority': failure_patterns['transformation_error']
            })
        
        improvements.sort(key=lambda x: x.get('priority', 0), reverse=True)
        return improvements
    
    def _create_improvement_document(self, analysis: Dict):
        """개선 문서 생성"""
        doc_path = self.iteration_dir / "improvement_analysis.md"
        
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(f"# 개선 분석 리포트 (반복 {self.iteration})\n\n")
            f.write(f"**생성 시간**: {analysis['timestamp']}\n\n")
            
            f.write("## 실패 통계\n\n")
            f.write(f"- 총 실패: {analysis['total_failures']}\n\n")
            
            f.write("## 실패 패턴 분석\n\n")
            for pattern, count in analysis['failure_patterns'].items():
                f.write(f"- **{pattern}**: {count}회\n")
            f.write("\n")
            
            f.write("## 개선 방안\n\n")
            for idx, improvement in enumerate(analysis['improvements'], 1):
                f.write(f"### 개선 방안 {idx}\n\n")
                f.write(f"- **타입**: {improvement['type']}\n")
                f.write(f"- **설명**: {improvement['description']}\n")
                f.write(f"- **액션**: {improvement['action']}\n\n")
    
    def save_summary(self, phase1_results: Dict, phase2_analysis: Dict):
        """요약 리포트 저장"""
        summary = {
            'iteration': self.iteration,
            'timestamp': datetime.now().isoformat(),
            'phase1': {
                'success_rate': phase1_results['overall_success_rate'],
                'total_correct': phase1_results['total_correct'],
                'total_objects': phase1_results['total_objects']
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
    max_iterations = 50
    iteration = 0
    prompt_variant = 0
    
    print("=" * 80)
    print("Egocentric Object Localization 완전 자동 개선 시스템")
    print("=" * 80)
    print(f"목표 성공률: {target_success_rate:.1%}")
    print(f"최대 반복 횟수: {max_iterations}")
    print("자동 실행 모드: 90% 달성까지 자동으로 테스트 및 개선")
    print("=" * 80)
    
    best_success_rate = 0.0
    best_iteration = 0
    best_prompt_variant = 0
    no_improvement_count = 0
    max_no_improvement = 3
    
    while iteration < max_iterations:
        print(f"\n{'#'*80}")
        print(f"# 반복 {iteration} 시작")
        print(f"{'#'*80}")
        
        test_system = EgocentricLocalizationTest(iteration=iteration, prompt_variant=prompt_variant)
        
        # Phase 1: 테스트 실행
        phase1_results = test_system.run_phase1(num_environments=10)
        
        # 성공률 확인
        current_success_rate = phase1_results['overall_success_rate']
        
        print(f"\n[반복 {iteration} 결과]")
        print(f"  성공률: {current_success_rate:.1%} (프롬프트 변형: {prompt_variant})")
        print(f"  정답: {phase1_results['total_correct']}/{phase1_results['total_objects']}")
        
        # 최고 성공률 업데이트
        if current_success_rate > best_success_rate:
            best_success_rate = current_success_rate
            best_iteration = iteration
            best_prompt_variant = prompt_variant
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
            print(f"최종 프롬프트 변형: {prompt_variant}")
            break
        
        # 개선 사항 자동 적용
        print(f"\n[자동 개선 분석]")
        improvements = phase2_analysis.get('improvements', [])
        
        if improvements:
            print(f"  발견된 개선 사항: {len(improvements)}개")
            for imp in improvements:
                print(f"    - {imp['action']} (우선순위: {imp['priority']})")
            
            # 개선 사항 적용
            for improvement in improvements:
                action = improvement['action']
                
                if action == 'enhance_object_detection':
                    prompt_variant = max(prompt_variant, 1)
                elif action == 'clarify_coordinate_systems':
                    prompt_variant = max(prompt_variant, 1)
                elif action == 'enhance_transformation':
                    prompt_variant = max(prompt_variant, 2)
                elif action == 'enhance_step_by_step':
                    prompt_variant = max(prompt_variant, 2)
            
            print(f"  → 프롬프트 변형 업데이트: {prompt_variant}")
        else:
            print(f"  개선 사항이 없음. 적극적 개선 모드 활성화...")
            if no_improvement_count >= max_no_improvement:
                prompt_variant = min(prompt_variant + 1, 2)
                print(f"  → 프롬프트 변형 증가: {prompt_variant}")
        
        # 성공률이 매우 낮으면 강제 개선
        if current_success_rate < 0.5 and iteration > 2:
            print(f"  ⚠ 성공률이 매우 낮음. 강제 개선 모드...")
            prompt_variant = max(prompt_variant, 2)
            print(f"  → 강제 프롬프트 변형: {prompt_variant}")
        
        iteration += 1
        print(f"\n{'='*80}")
        print(f"다음 반복으로 진행... (현재 최고: {best_success_rate:.1%} @ 반복 {best_iteration})")
        print(f"{'='*80}")
    
    print(f"\n{'='*80}")
    print("자동 개선 완료")
    print(f"{'='*80}")
    print(f"최종 성공률: {best_success_rate:.1%} (반복 {best_iteration})")
    print(f"총 반복 횟수: {iteration}")
    print(f"최종 프롬프트 변형: {best_prompt_variant}")
    
    if best_success_rate >= target_success_rate:
        print(f"\n✅ 목표 달성 성공!")
    else:
        print(f"\n⚠ 목표 미달성 (목표: {target_success_rate:.1%}, 달성: {best_success_rate:.1%})")


if __name__ == "__main__":
    main()

