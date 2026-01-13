"""
MiniGrid 전용 VLM 컨트롤러 (하위 호환성 유지)

이 모듈은 기존 코드와의 호환성을 위해 유지되며,
내부적으로 vlm_controller.VLMController를 사용합니다.

새로운 프로젝트에서는 vlm_controller.VLMController를 직접 사용하는 것을 권장합니다.
"""

# Actual paths: lib.vlm.vlm_controller, lib.map_manager.minigrid_customenv_emoji
from lib import VLMController, MiniGridEmojiWrapper
import cv2
from typing import Optional


class MiniGridVLMController(VLMController):
    """
    MiniGrid 전용 VLM 컨트롤러 (하위 호환성)
    
    VLMController를 상속받아 MiniGrid 특정 기능을 추가합니다.
    새로운 프로젝트에서는 VLMController를 직접 사용하는 것을 권장합니다.
    
    사용 예시:
        # 환경 생성
        from minigrid_customenv_emoji import MiniGridEmojiWrapper
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
        super().__init__(
            env=env,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            required_fields=required_fields
        )
        self.minigrid_env = env  # MiniGrid 특정 기능용
    
    def visualize_state(self, window_name: str = "MiniGrid VLM Control", cell_size: int = 32):  # noqa: ARG002
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
                print(f"Image display error: {e}")
    
    def visualize_grid_cli(self):
        """CLI에서 그리드를 텍스트로 시각화 (MiniGrid 전용)"""
        # Actual path: legacy.vlm_rels.minigrid_vlm_helpers
        from legacy import visualize_minigrid_grid_cli
        state = self.env.get_state()
        visualize_minigrid_grid_cli(self.minigrid_env, state)
    
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
        print("MiniGrid VLM Interaction Started")
        print("=" * 60)
        
        while not done and step < max_steps:
            step += 1
            print("\n" + "=" * 80)
            print(f"STEP {step}")
            print("=" * 80)
            
            state = self.env.get_state()
            print(f"Position: {state['agent_pos']}, Direction: {state['agent_dir']}")
            
            self.visualize_grid_cli()
            self.visualize_state(window_name)
            
            print("Enter command (Enter: default prompt):")
            user_prompt = input("> ").strip()
            if not user_prompt:
                user_prompt = None
            
            try:
                _, reward, terminated, truncated, _, vlm_response = self.step(
                    user_prompt=user_prompt,
                    mission=mission
                )
                done = terminated or truncated
                
                action_str = vlm_response.get('action', 'N/A')
                print(f"Parsed action: {action_str}")
                print(f"Environment Info: {vlm_response.get('environment_info', 'N/A')}")
                print(f"Reasoning: {vlm_response.get('reasoning', 'N/A')}")
                print(f"Reward: {reward}, Done: {done}")
                
            except Exception as e:
                print(f"Error occurred: {e}")
                import traceback
                traceback.print_exc()
                break
            
            if done:
                print("\n" + "=" * 80)
                print("Goal reached! Terminating")
                print("=" * 80)
                break
        
        if step >= max_steps:
            print(f"\nMaximum step count ({max_steps}) reached.")
        
        cv2.destroyAllWindows()
        print("\nExperiment completed.")


# 환경 생성 함수는 프로젝트별로 별도 파일에서 관리
# 예제는 examples/ 디렉토리나 프로젝트별 파일에 위치
