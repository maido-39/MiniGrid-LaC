"""
MiniGrid 시나리오 2 환경 생성 예제

프로젝트별 환경 생성 함수를 별도 파일로 관리하는 예제입니다.
"""

# Actual path: lib.map_manager.minigrid_customenv_emoji
from lib import MiniGridEmojiWrapper
from typing import Dict


def create_scenario2_environment() -> MiniGridEmojiWrapper:
    """
    시나리오 2 환경 생성
    
    Returns:
        MiniGridEmojiWrapper: 절대 움직임이 활성화된 환경 인스턴스
    """
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
    
    # 절대 움직임 모드 활성화 (기본값이지만 명시적으로 설정)
    return MiniGridEmojiWrapper(size=size, room_config=room_config, use_absolute_movement=True)


def main():
    """메인 함수 (예제)"""
    from vlm_controller import VLMController
    
    print("=" * 60)
    print("MiniGrid VLM 상호작용 (절대 좌표 이동 버전)")
    print("=" * 60)
    
    # 환경 생성
    env = create_scenario2_environment()
    env.reset()
    
    # 범용 VLM 컨트롤러 사용
    controller = VLMController(env=env)
    
    mission = "파란 기둥으로 가서 오른쪽으로 돌고, 테이블 옆에 멈추시오"
    
    # 대화형 실행 (MiniGrid 전용 기능이 필요한 경우 MiniGridVLMController 사용)
    # Actual path: legacy.vlm_rels.minigrid_vlm_controller
    from legacy import MiniGridVLMController
    minigrid_controller = MiniGridVLMController(env=env)
    minigrid_controller.run_interactive(mission=mission, max_steps=100)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

