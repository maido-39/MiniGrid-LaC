"""
VLM을 사용하여 MiniGrid 환경을 제어하는 클래스

이 모듈은 VLM(Vision Language Model)을 사용하여 MiniGrid 환경을 조작하기 위한
클래스를 제공합니다. 환경 생성 및 관리는 minigrid_customenv_emoji.py에서 처리합니다.

주요 기능:
- VLM을 사용한 액션 생성
- 프롬프트 관리 (인스턴스화 시 편하게 조작 가능)
- 환경 상태 시각화
- VLM 응답 파싱 및 액션 실행
"""

from minigrid_customenv_emoji import MiniGridEmojiWrapper
from vlm_wrapper import ChatGPT4oVLMWrapper
from vlm_postprocessor import VLMResponsePostProcessor
import numpy as np
import cv2
from typing import Union, Tuple, Dict, Optional


class MiniGridVLMController:
    """
    VLM을 사용하여 MiniGrid 환경을 제어하는 클래스
    
    사용 예시:
        # 환경 생성
        env = MiniGridEmojiWrapper(size=10, room_config={...})
        env.reset()
        
        # 컨트롤러 생성
        controller = MiniGridVLMController(
            env=env,
            model="gpt-4o",
            system_prompt="You are a robot...",
            user_prompt_template="Complete the mission: {mission}"
        )
        
        # VLM으로 액션 생성 및 실행
        response = controller.generate_action()
        controller.execute_action(response['action'])
    """
    
    def __init__(
        self,
        env: MiniGridEmojiWrapper,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        required_fields: Optional[list] = None
    ):
        """
        컨트롤러 초기화
        
        Args:
            env: MiniGridEmojiWrapper 환경 인스턴스
            model: VLM 모델명 (기본값: "gpt-4o")
            temperature: 생성 온도 (기본값: 0.0)
            max_tokens: 최대 토큰 수 (기본값: 1000)
            system_prompt: 시스템 프롬프트 (None이면 기본값 사용)
            user_prompt_template: 사용자 프롬프트 템플릿 (None이면 기본값 사용)
            required_fields: VLM 응답 필수 필드 리스트 (기본값: ["action", "environment_info"])
        """
        self.env = env
        
        self.vlm = ChatGPT4oVLMWrapper(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.postprocessor = VLMResponsePostProcessor(
            required_fields=required_fields or ["action", "environment_info"]
        )
        
        # 프롬프트 설정
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.user_prompt_template = user_prompt_template or self._get_default_user_prompt_template()
    
    def _get_default_system_prompt(self) -> str:
        """기본 시스템 프롬프트 반환"""
        return """You are a robot operating on a grid map.

## Environment
Grid world with walls (black), blue pillar (impassable), purple table (impassable), robot (red arrow shows heading), and goal (green marker if present).

## Coordinate System
The top of the image is North (up), and the bottom is South (down).
The left is West (left), and the right is East (right).

## Robot Orientation
In the image, the red triangle represents the robot.
The robot's heading direction is shown by the triangle's apex (sharp tip).
However, you can move in ANY direction regardless of the robot's current heading.

## Action Space (Absolute Directions)
You can move directly in absolute directions:
- "up": Move North
- "down": Move South
- "left": Move West
- "right": Move East
- "pickup": Pick up object
- "drop": Drop object
- "toggle": Interact with objects

## Movement Rules
**CRITICAL**: All movements are in ABSOLUTE directions (North/South/East/West).
- "up" = move North (upward on the image)
- "down" = move South (downward on the image)
- "left" = move West (leftward on the image)
- "right" = move East (rightward on the image)
- The robot will automatically rotate to face the correct direction before moving
- You don't need to think about the robot's current heading - just specify the direction you want to go

## Response Format
Respond in JSON format:
```json
{
    "action": "<action_name_or_number>",
    "environment_info": "<description of current state with spatial relationships in absolute coordinates (North/South/East/West)>",
    "reasoning": "<explanation of why you chose this action>"
}
```

**Important**: 
- Valid JSON format required
- Actions must be from the list above
- Complete mission from user prompt
- Use absolute directions (up/down/left/right), not relative to robot heading
- Think in terms of the image: up=North, down=South, left=West, right=East
"""
    
    def _get_default_user_prompt_template(self) -> str:
        """기본 사용자 프롬프트 템플릿 반환"""
        return "Based on the current image, choose the next action to complete the mission: {mission}. Use absolute directions (up/down/left/right)."
    
    def set_system_prompt(self, prompt: str):
        """시스템 프롬프트 설정"""
        self.system_prompt = prompt
    
    def set_user_prompt_template(self, template: str):
        """사용자 프롬프트 템플릿 설정"""
        self.user_prompt_template = template
    
    def get_user_prompt(self, mission: Optional[str] = None, **kwargs) -> str:
        """
        사용자 프롬프트 생성
        
        Args:
            mission: 미션 텍스트 (None이면 환경의 미션 사용)
            **kwargs: 템플릿에 추가할 키워드 인자
        
        Returns:
            생성된 사용자 프롬프트
        """
        if mission is None:
            state = self.env.get_state()
            mission = state.get('mission', 'explore')
        
        return self.user_prompt_template.format(mission=mission, **kwargs)
    
    def generate_action(
        self,
        user_prompt: Optional[str] = None,
        mission: Optional[str] = None
    ) -> Dict:
        """
        VLM을 사용하여 액션 생성
        
        Args:
            user_prompt: 사용자 프롬프트 (None이면 템플릿에서 생성)
            mission: 미션 텍스트 (user_prompt가 None일 때만 사용)
        
        Returns:
            파싱된 VLM 응답 딕셔너리
        """
        image = self.env.get_image()
        
        if user_prompt is None:
            user_prompt = self.get_user_prompt(mission=mission)
        
        try:
            vlm_response_raw = self.vlm.generate(
                image=image,
                system_prompt=self.system_prompt,
                user_prompt=user_prompt
            )
        except Exception as e:
            raise RuntimeError(f"VLM API 호출 실패: {e}")
        
        try:
            vlm_response = self.postprocessor.process(vlm_response_raw, strict=True)
            return vlm_response
        except ValueError as e:
            raise ValueError(f"VLM 응답 파싱 실패: {e}\n원본 응답: {vlm_response_raw[:200]}...")
    
    def execute_action(self, action: Union[int, str]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        액션 실행 (절대 좌표 이동 사용)
        
        Args:
            action: 액션 (정수 인덱스 또는 액션 이름 문자열)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        return self.env.step_absolute(action)
    
    def step(
        self,
        user_prompt: Optional[str] = None,
        mission: Optional[str] = None
    ) -> Tuple[Dict, float, bool, bool, Dict, Dict]:
        """
        VLM으로 액션 생성 후 실행 (한 번에 처리)
        
        Args:
            user_prompt: 사용자 프롬프트 (None이면 템플릿에서 생성)
            mission: 미션 텍스트 (user_prompt가 None일 때만 사용)
        
        Returns:
            observation, reward, terminated, truncated, info, vlm_response
        """
        vlm_response = self.generate_action(user_prompt=user_prompt, mission=mission)
        action = vlm_response.get('action', 'up')
        
        obs, reward, terminated, truncated, info = self.execute_action(action)
        
        return obs, reward, terminated, truncated, info, vlm_response
    
    def visualize_state(self, window_name: str = "MiniGrid VLM Control", cell_size: int = 32):
        """
        현재 환경 상태를 OpenCV로 시각화
        
        Args:
            window_name: 창 이름
            cell_size: 셀 크기 (이미지 확대용)
        """
        image = self.env.get_image()
        
        if image is not None:
            try:
                img_bgr = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
                
                height, width = img_bgr.shape[:2]
                max_size = 800
                if height < max_size and width < max_size:
                    scale = min(max_size // height, max_size // width, 4)
                    if scale > 1:
                        new_width = width * scale
                        new_height = height * scale
                        img_bgr = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                
                cv2.imshow(window_name, img_bgr)
                cv2.waitKey(1)
            except Exception as e:
                print(f"이미지 표시 오류: {e}")
    
    def visualize_grid_cli(self):
        """CLI에서 그리드를 텍스트로 시각화"""
        state = self.env.get_state()
        env = self.env.env
        size = self.env.size
        
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
                        color_map = {
                            'blue': '🟦',
                            'purple': '🟪',
                            'red': '🟥',
                            'green': '🟩',
                            'yellow': '🟨'
                        }
                        row.append(color_map.get(cell.color, '⬛'))
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
    
    def run_interactive(
        self,
        mission: Optional[str] = None,
        max_steps: int = 100,
        window_name: str = "MiniGrid VLM Control"
    ):
        """
        대화형 모드로 실행 (사용자 입력 받아서 실행)
        
        Args:
            mission: 미션 텍스트
            max_steps: 최대 스텝 수
            window_name: 창 이름
        """
        step = 0
        done = False
        
        print("=" * 60)
        print("MiniGrid VLM 상호작용 시작")
        print("=" * 60)
        
        while not done and step < max_steps:
            step += 1
            print("\n" + "=" * 80)
            print(f"STEP {step}")
            print("=" * 80)
            
            state = self.env.get_state()
            print(f"위치: {state['agent_pos']}, 방향: {state['agent_dir']}")
            
            self.visualize_grid_cli()
            self.visualize_state(window_name)
            
            print("명령을 입력하세요 (Enter: 기본 프롬프트):")
            user_prompt = input("> ").strip()
            if not user_prompt:
                user_prompt = None
            
            try:
                obs, reward, terminated, truncated, info, vlm_response = self.step(
                    user_prompt=user_prompt,
                    mission=mission
                )
                done = terminated or truncated
                
                action_str = vlm_response.get('action', 'N/A')
                print(f"파싱된 액션: {action_str}")
                print(f"Environment Info: {vlm_response.get('environment_info', 'N/A')}")
                print(f"Reasoning: {vlm_response.get('reasoning', 'N/A')}")
                print(f"보상: {reward}, 종료: {done}")
                
            except Exception as e:
                print(f"오류 발생: {e}")
                import traceback
                traceback.print_exc()
                break
            
            if done:
                print("\n" + "=" * 80)
                print("Goal 도착! 종료")
                print("=" * 80)
                break
        
        if step >= max_steps:
            print(f"\n최대 스텝 수({max_steps})에 도달했습니다.")
        
        cv2.destroyAllWindows()
        print("\n실험 완료.")


def create_scenario2_environment():
    """시나리오 2 환경 생성 예제"""
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
    
    return MiniGridEmojiWrapper(size=size, room_config=room_config)


def main():
    """메인 함수 (예제)"""
    print("=" * 60)
    print("MiniGrid VLM 상호작용 (절대 좌표 이동 버전)")
    print("=" * 60)
    
    env = create_scenario2_environment()
    env.reset()
    
    controller = MiniGridVLMController(env=env)
    
    mission = "파란 기둥으로 가서 오른쪽으로 돌고, 테이블 옆에 멈추시오"
    controller.run_interactive(mission=mission, max_steps=100)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

