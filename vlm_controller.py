"""
VLM을 사용하여 환경을 제어하는 범용 컨트롤러 클래스

이 모듈은 VLM(Vision Language Model)을 사용하여 환경을 조작하기 위한
범용 컨트롤러 클래스를 제공합니다. 환경에 독립적으로 작동하며,
Protocol 기반 인터페이스를 통해 다양한 환경과 호환됩니다.

주요 기능:
- VLM을 사용한 액션 생성
- 프롬프트 관리 (인스턴스화 시 편하게 조작 가능)
- 환경 상태 시각화 (환경 특정 시각화는 별도 헬퍼 사용)
- VLM 응답 파싱 및 액션 실행
"""

from vlm_wrapper import ChatGPT4oVLMWrapper
from vlm_postprocessor import VLMResponsePostProcessor
import numpy as np
from typing import Union, Tuple, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class EnvironmentProtocol(Protocol):
    """
    VLM 컨트롤러가 요구하는 환경 인터페이스
    
    이 Protocol을 구현하는 모든 환경은 VLMController와 호환됩니다.
    """
    
    def get_image(self, fov_range: Optional[int] = None, fov_width: Optional[int] = None) -> np.ndarray:
        """현재 환경의 이미지를 반환 (VLM 입력용)"""
        ...
    
    def get_state(self) -> Dict:
        """현재 환경 상태 정보 반환"""
        ...
    
    def step(self, action: Union[int, str]) -> Tuple[Dict, float, bool, bool, Dict]:
        """액션 실행"""
        ...
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """환경 초기화"""
        ...
    
    def parse_absolute_action(self, action_str: str) -> int:
        """절대 방향 액션 문자열을 인덱스로 변환 (선택적)"""
        ...
    
    def get_absolute_action_space(self) -> Dict:
        """절대 방향 액션 공간 정보 반환 (선택적)"""
        ...


class VLMController:
    """
    VLM을 사용하여 환경을 제어하는 범용 컨트롤러 클래스
    
    이 클래스는 EnvironmentProtocol을 구현하는 모든 환경과 호환됩니다.
    환경 특정 코드는 포함하지 않으며, 순수하게 VLM과 환경 인터페이스만 사용합니다.
    
    사용 예시:
        # 환경 생성 (프로젝트별로 별도 파일에서)
        from my_project.environment import create_my_environment
        env = create_my_environment()
        env.reset()
        
        # 컨트롤러 생성
        controller = VLMController(
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
        env: EnvironmentProtocol,
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
            env: EnvironmentProtocol을 구현하는 환경 인스턴스
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
        """기본 시스템 프롬프트 반환 (절대 방향 이동 기준)"""
        return """You are a robot operating on a grid map.

## Environment
Grid world with walls, objects, robot, and goal markers.

## Coordinate System
The top of the image is North (up), and the bottom is South (down).
The left is West (left), and the right is East (right).

## Robot Orientation
The robot's heading direction is shown in the image.
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
        액션 실행
        
        Args:
            action: 액션 (정수 인덱스 또는 액션 이름 문자열)
        
        Returns:
            observation, reward, terminated, truncated, info
        
        Note:
            환경이 use_absolute_movement=True로 설정되어 있으면 step()이 자동으로 절대 움직임을 처리합니다.
            환경이 parse_absolute_action 메서드를 제공하면 문자열 액션을 파싱합니다.
        """
        # 환경이 parse_absolute_action을 제공하는 경우 사용
        if isinstance(action, str) and hasattr(self.env, 'parse_absolute_action'):
            try:
                action = self.env.parse_absolute_action(action)
            except (AttributeError, ValueError):
                pass  # 파싱 실패 시 원본 action 사용
        
        return self.env.step(action)
    
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

