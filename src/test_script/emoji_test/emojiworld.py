"""
이모지 숲 환경 예제

이모지가 있는 숲 환경을 생성하고 에이전트가 돌아다니는 예제입니다.
"""

from minigrid import register_minigrid_envs
# Actual path: legacy.relative_movement.custom_environment
from legacy import CustomRoomWrapper
import numpy as np
import cv2
import random

# MiniGrid 환경 등록
register_minigrid_envs()


def create_emoji_obj_dict(emoji_name: str, pos: tuple, color: str = 'yellow', can_pickup: bool = False) -> dict:
    """
    EmojiObject를 위한 딕셔너리 생성 헬퍼 함수
    
    Args:
        emoji_name: 이모지 이름 (예: 'tree', 'rock', 'flower')
        pos: 위치 (x, y) 튜플
        color: 색상 (기본값: 'yellow')
        can_pickup: 집기 가능 여부 (기본값: False)
            - True: 에이전트가 앞에서 바라보면 집을 수 있음
            - False: 집을 수 없음 (장애물)
    
    Returns:
        room_config의 objects 리스트에 추가할 딕셔너리
    """
    return {
        'type': 'emoji',
        'pos': pos,
        'emoji_name': emoji_name,
        'color': color,
        'can_pickup': can_pickup
    }


def create_forest_environment(
    size: int = 15,
    tree_density: float = 0.15,
    rock_density: float = 0.05,
    flower_density: float = 0.10,
    grass_density: float = 0.20
) -> CustomRoomWrapper:
    """
    이모지가 있는 숲 환경 생성
    
    Args:
        size: 환경 크기 (기본값: 15)
        tree_density: 나무 밀도 (0.0 ~ 1.0, 기본값: 0.15)
        rock_density: 돌 밀도 (기본값: 0.05)
        flower_density: 꽃 밀도 (기본값: 0.10)
        grass_density: 풀 밀도 (기본값: 0.20)
    
    Returns:
        CustomRoomWrapper: 숲 환경 인스턴스
    """
    # 외벽은 자동 생성되므로 walls는 빈 리스트
    walls = []
    
    # 사용 가능한 위치 리스트 (외벽 제외)
    available_positions = []
    for x in range(1, size - 1):
        for y in range(1, size - 1):
            available_positions.append((x, y))
    
    # 객체 리스트
    objects = []
    used_positions = set()
    
    # 나무 배치 (통과 불가능, 장애물)
    num_trees = int(len(available_positions) * tree_density)
    tree_positions = random.sample(available_positions, min(num_trees, len(available_positions)))
    for pos in tree_positions:
        if pos not in used_positions:
            objects.append(create_emoji_obj_dict('tree', pos, 'green', can_pickup=False))
            used_positions.add(pos)
    
    # 돌 배치 (통과 불가능, 장애물) 🗿
    remaining_positions = [p for p in available_positions if p not in used_positions]
    num_rocks = int(len(remaining_positions) * rock_density)
    if num_rocks > 0 and len(remaining_positions) > 0:
        rock_positions = random.sample(remaining_positions, min(num_rocks, len(remaining_positions)))
        for pos in rock_positions:
            if pos not in used_positions:
                objects.append(create_emoji_obj_dict('rock', pos, 'grey', can_pickup=False))
                used_positions.add(pos)
    
    # 꽃 배치 (통과 가능, 장식)
    remaining_positions = [p for p in available_positions if p not in used_positions]
    num_flowers = int(len(remaining_positions) * flower_density)
    if num_flowers > 0 and len(remaining_positions) > 0:
        flower_positions = random.sample(remaining_positions, min(num_flowers, len(remaining_positions)))
        for pos in flower_positions:
            if pos not in used_positions:
                # 다양한 색상의 꽃
                flower_colors = ['yellow', 'red', 'purple']
                color = random.choice(flower_colors)
                objects.append(create_emoji_obj_dict('flower', pos, color, can_pickup=True))
                used_positions.add(pos)
    
    # 풀 배치 (통과 가능, 장식) 🌾
    remaining_positions = [p for p in available_positions if p not in used_positions]
    num_grass = int(len(remaining_positions) * grass_density)
    if num_grass > 0 and len(remaining_positions) > 0:
        grass_positions = random.sample(remaining_positions, min(num_grass, len(remaining_positions)))
        for pos in grass_positions:
            if pos not in used_positions:
                objects.append(create_emoji_obj_dict('grass', pos, 'green', can_pickup=True))
                used_positions.add(pos)
    
    # 시작 위치 (빈 공간 중 하나 선택)
    empty_positions = [p for p in available_positions if p not in used_positions]
    if len(empty_positions) == 0:
        # 모든 위치가 사용된 경우 시작 위치를 강제로 설정
        start_pos = (1, 1)
    else:
        start_pos = random.choice(empty_positions)
    
    # 목표 위치 (시작 위치와 다른 빈 공간)
    remaining_empty = [p for p in empty_positions if p != start_pos]
    if len(remaining_empty) == 0:
        goal_pos = (size - 2, size - 2)
    else:
        goal_pos = random.choice(remaining_empty)
    
    room_config = {
        'start_pos': start_pos,
        'goal_pos': goal_pos,
        'walls': walls,
        'objects': objects
    }
    
    return CustomRoomWrapper(size=size, room_config=room_config)


def print_grid_info(wrapper: CustomRoomWrapper):
    """그리드 정보 출력"""
    grid_map = wrapper.parse_grid()
    
    # 이모지 통계
    emoji_counts = {}
    for _, obj_name in grid_map.items():
        if obj_name is not None:
            if obj_name not in emoji_counts:
                emoji_counts[obj_name] = 0
            emoji_counts[obj_name] += 1
    
    print("\n=== 숲 환경 정보 ===")
    print(f"환경 크기: {wrapper.size}x{wrapper.size}")
    print("\n이모지 통계:")
    for emoji_name, count in sorted(emoji_counts.items()):
        print(f"  {emoji_name}: {count}개")
    
    # 상태 정보 가져오기 (이미지 제외)
    if hasattr(wrapper.env, 'agent_pos'):
        agent_pos = wrapper.env.agent_pos
        if isinstance(agent_pos, np.ndarray):
            agent_pos = tuple(agent_pos.tolist())
        agent_dir = wrapper.env.agent_dir if hasattr(wrapper.env, 'agent_dir') else None
    else:
        agent_pos = None
        agent_dir = None
    
    print(f"\n에이전트 위치: {agent_pos}")
    print(f"에이전트 방향: {agent_dir}")
    print(f"목표 위치: {wrapper.env.room_config.get('goal_pos', 'N/A')}")


def explore_forest_keyboard(wrapper: CustomRoomWrapper):
    """
    키보드로 숲을 탐험하는 함수
    
    Args:
        wrapper: CustomRoomWrapper 인스턴스
    """
    print("\n=== 키보드 탐험 모드 ===")
    print("조작법:")
    print("  w: 앞으로 이동")
    print("  a: 왼쪽으로 회전")
    print("  d: 오른쪽으로 회전")
    print("  s: 뒤로 이동")
    print("  r: 환경 리셋")
    print("  p: 현재 위치의 이모지 확인")
    print("  q: 종료")
    print("\n탐험을 시작합니다...")
    
    window_name = "Forest Exploration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while True:
        # 현재 이미지 가져오기
        image = wrapper.get_image()
        
        # 상태 정보
        state = wrapper.get_state()
        agent_pos = state['agent_pos']
        agent_dir = state['agent_dir']
        
        # 이미지에 정보 표시
        info_image = image.copy()
        cv2.putText(info_image, f"Pos: {agent_pos}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_image, f"Dir: {agent_dir}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 현재 위치의 이모지 확인
        if isinstance(agent_pos, (tuple, list, np.ndarray)):
            if isinstance(agent_pos, np.ndarray):
                x, y = int(agent_pos[0]), int(agent_pos[1])
            else:
                x, y = int(agent_pos[0]), int(agent_pos[1])
            
            emoji = wrapper.get_emoji_at(x, y)
            if emoji:
                cv2.putText(info_image, f"Emoji: {emoji}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 이미지 표시
        cv2.imshow(window_name, cv2.cvtColor(info_image, cv2.COLOR_RGB2BGR))
        
        # 키 입력 대기
        key = cv2.waitKey(100) & 0xFF
        
        if key == ord('q'):
            print("탐험을 종료합니다.")
            break
        elif key == ord('w'):
            wrapper.step("move forward")
            print(f"앞으로 이동 -> 위치: {wrapper.get_state()['agent_pos']}")
        elif key == ord('a'):
            wrapper.step("turn left")
            print(f"왼쪽으로 회전 -> 방향: {wrapper.get_state()['agent_dir']}")
        elif key == ord('d'):
            wrapper.step("turn right")
            print(f"오른쪽으로 회전 -> 방향: {wrapper.get_state()['agent_dir']}")
        elif key == ord('s'):
            wrapper.step("move backward")
            print(f"뒤로 이동 -> 위치: {wrapper.get_state()['agent_pos']}")
        elif key == ord('r'):
            wrapper.reset()
            print("환경을 리셋했습니다.")
        elif key == ord('p'):
            if isinstance(agent_pos, (tuple, list, np.ndarray)):
                if isinstance(agent_pos, np.ndarray):
                    x, y = int(agent_pos[0]), int(agent_pos[1])
                else:
                    x, y = int(agent_pos[0]), int(agent_pos[1])
                emoji = wrapper.get_emoji_at(x, y)
                if emoji:
                    print(f"현재 위치 ({x}, {y})의 이모지: {emoji}")
                else:
                    print(f"현재 위치 ({x}, {y})에는 이모지가 없습니다.")
    
    cv2.destroyAllWindows()


def explore_forest_auto(wrapper: CustomRoomWrapper, num_steps: int = 50):
    """
    자동으로 숲을 탐험하는 함수
    
    Args:
        wrapper: CustomRoomWrapper 인스턴스
        num_steps: 탐험할 스텝 수 (기본값: 50)
    """
    print("\n=== 자동 탐험 모드 ===")
    print(f"{num_steps} 스텝 동안 자동으로 탐험합니다...")
    
    window_name = "Forest Auto Exploration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    visited_positions = set()
    emoji_encounters = {}
    
    for step in range(num_steps):
        # 현재 위치 기록
        state = wrapper.get_state()
        agent_pos = state['agent_pos']
        
        if isinstance(agent_pos, np.ndarray):
            pos_tuple = (int(agent_pos[0]), int(agent_pos[1]))
        else:
            pos_tuple = (int(agent_pos[0]), int(agent_pos[1]))
        
        visited_positions.add(pos_tuple)
        
        # 현재 위치의 이모지 확인
        emoji = wrapper.get_emoji_at(pos_tuple[0], pos_tuple[1])
        if emoji:
            if emoji not in emoji_encounters:
                emoji_encounters[emoji] = 0
            emoji_encounters[emoji] += 1
            print(f"Step {step+1}: {emoji} 발견! (위치: {pos_tuple})")
        
        # 랜덤 액션 선택 (앞으로 이동, 회전)
        actions = ["move forward", "turn left", "turn right"]
        action = random.choice(actions)
        
        # 액션 실행
        _, _, terminated, truncated, _ = wrapper.step(action)
        
        # 이미지 표시
        image = wrapper.get_image()
        info_image = image.copy()
        cv2.putText(info_image, f"Step: {step+1}/{num_steps}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_image, f"Pos: {pos_tuple}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if emoji:
            cv2.putText(info_image, f"Emoji: {emoji}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(window_name, cv2.cvtColor(info_image, cv2.COLOR_RGB2BGR))
        
        # 종료 조건
        if terminated or truncated:
            print("목표에 도달하거나 종료 조건이 만족되었습니다.")
            break
        
        # 키 입력 확인 (q로 종료)
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            print("사용자가 탐험을 중단했습니다.")
            break
    
    # 탐험 결과 출력
    print("\n=== 탐험 결과 ===")
    print(f"방문한 위치 수: {len(visited_positions)}")
    print("발견한 이모지:")
    for emoji_name, count in sorted(emoji_encounters.items()):
        print(f"  {emoji_name}: {count}회")
    
    cv2.waitKey(2000)  # 2초 대기
    cv2.destroyAllWindows()


def main():
    """메인 함수"""
    print("=" * 60)
    print("이모지 숲 환경 예제")
    print("=" * 60)
    
    # 숲 환경 생성
    print("\n숲 환경을 생성합니다...")
    forest = create_forest_environment(
        size=15,
        tree_density=0.15,
        rock_density=0.05,
        flower_density=0.10,
        grass_density=0.20
    )
    
    # 환경 초기화
    forest.reset()
    
    # 그리드 정보 출력
    print_grid_info(forest)
    
    # 모드 선택
    print("\n=== 탐험 모드 선택 ===")
    print("1. 키보드 탐험 (수동 조작)")
    print("2. 자동 탐험 (랜덤 움직임)")
    print("3. 둘 다 실행")
    
    choice = input("\n선택 (1/2/3, 기본값: 1): ").strip()
    
    if choice == '2':
        # 자동 탐험
        explore_forest_auto(forest, num_steps=100)
    elif choice == '3':
        # 둘 다 실행
        explore_forest_auto(forest, num_steps=50)
        print("\n이제 키보드 탐험 모드로 전환합니다...")
        explore_forest_keyboard(forest)
    else:
        # 키보드 탐험 (기본값)
        explore_forest_keyboard(forest)
    
    forest.close()
    print("\n프로그램을 종료합니다.")


if __name__ == "__main__":
    main()

