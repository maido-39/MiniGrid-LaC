"""
시나리오 2 실험 환경 테스트 스크립트

시나리오 2: 파란 기둥으로 가서 오른쪽으로 돌고, 테이블 옆에 멈추시오

환경 구성:
- 벽: 검은색 (외벽)
- 파란 기둥: 파란색 2x2 Grid (통과불가)
- 테이블: 보라색 1x3 Grid (통과불가)
- 시작점: 빨강 1x1
- 종료점: 초록 1x1

레이아웃 (10x10):
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬜️⬜️⬜️⬜️🟪🟪🟪🟩⬛
⬛⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬛
⬛⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬛
⬛⬜️⬜️🟦🟦⬜️⬜️⬜️⬜️⬛ 
⬛⬜️⬜️🟦🟦⬜️⬜️⬜️⬜️⬛
⬛⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬛
⬛⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬛
⬛🟥⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
"""

from minigrid import register_minigrid_envs
from custom_environment import CustomRoomWrapper
import cv2

# MiniGrid 환경 등록
register_minigrid_envs()


def create_scenario2_environment():
    """
    시나리오 2 실험 환경 생성
    
    Returns:
        CustomRoomWrapper: 시나리오 2 환경 Wrapper 인스턴스
    """
    size = 10
    
    # 외벽 생성 (검은색 벽)
    walls = []
    for i in range(size):
        walls.append((i, 0))      # 상단 벽
        walls.append((i, size-1))  # 하단 벽
        walls.append((0, i))      # 좌측 벽
        walls.append((size-1, i))  # 우측 벽
    
    # 파란 기둥: 2x2 Grid (통과불가)
    # 위치: (3, 4), (4, 4), (3, 5), (4, 5)
    # MiniGrid 좌표계: (x, y) = (열, 행)
    blue_pillar_positions = [
        (3, 4),  # 왼쪽 위
        (4, 4),  # 오른쪽 위
        (3, 5),  # 왼쪽 아래
        (4, 5),  # 오른쪽 아래
    ]
    
    # 테이블: 보라색 1x3 Grid (통과불가)
    # 위치: (5, 1), (6, 1), (7, 1)
    table_positions = [
        (5, 1),  # 왼쪽
        (6, 1),  # 중앙
        (7, 1),  # 오른쪽
    ]
    
    # 객체 리스트 생성
    objects = []
    
    # 파란 기둥 배치 (파란색 Box로 구현)
    # 참고: Box는 통과 가능하지만, 시각적으로는 색상이 있는 객체로 표시됩니다.
    # 통과 불가능하게 만들려면 나중에 CustomRoomEnv를 확장해야 합니다.
    for pos in blue_pillar_positions:
        objects.append({
            'type': 'box',
            'pos': pos,
            'color': 'blue'
        })
    
    # 테이블 배치 (보라색 Box로 구현)
    for pos in table_positions:
        objects.append({
            'type': 'box',
            'pos': pos,
            'color': 'purple'
        })
    
    # 시작점: 빨강 1x1 (에이전트 시작 위치)
    # 위치: (1, 8) - 레이아웃에서 🟥 위치
    start_pos = (1, 8)
    
    # 종료점: 초록 1x1 (Goal)
    # 위치: (8, 1) - 레이아웃에서 🟩 위치
    goal_pos = (8, 1)
    
    # room_config 구성
    room_config = {
        'start_pos': start_pos,
        'goal_pos': goal_pos,
        'walls': walls,
        'objects': objects
    }
    
    # Wrapper 생성 및 반환
    return CustomRoomWrapper(size=size, room_config=room_config)


def visualize_scenario2():
    """
    시나리오 2 환경을 시각화 (OpenCV 사용)
    """
    print("=" * 60)
    print("시나리오 2 실험 환경 시각화")
    print("=" * 60)
    print("\n환경 구성:")
    print("  - 파란 기둥: 2x2 Grid (통과불가)")
    print("  - 테이블: 보라색 1x3 Grid (통과불가)")
    print("  - 시작점: 빨강 (1, 8)")
    print("  - 종료점: 초록 (8, 1)")
    print("\nMission: 파란 기둥으로 가서 오른쪽으로 돌고, 테이블 옆에 멈추시오")
    print("\n환경을 표시합니다...")
    print("아무 키나 누르면 종료됩니다.\n")
    
    # 환경 생성
    wrapper = create_scenario2_environment()
    
    # 환경 초기화
    wrapper.reset()
    
    # 환경 상태 정보 출력
    state = wrapper.get_state()
    print(f"에이전트 시작 위치: {state['agent_pos']}")
    print(f"에이전트 방향: {state['agent_dir']} (0=오른쪽, 1=아래, 2=왼쪽, 3=위)")
    print(f"미션: {state['mission']}")
    
    # 메인 루프: 이미지를 계속 표시
    while True:
        # 현재 환경 이미지 가져오기
        img = wrapper.get_image()
        
        if img is not None:
            try:
                # RGB를 BGR로 변환 (OpenCV는 BGR 형식을 사용)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # 이미지 크기 조정 (더 크게 표시)
                # keyboard_control.py와 동일한 방식 사용하되, 더 큰 scale 사용
                height, width = img_bgr.shape[:2]
                # keyboard_control.py 방식: 최대 크기 제한
                # 하지만 더 크게 보이도록 max_size를 늘림
                max_size = 1200  # 최대 1200x1200 픽셀 (더 크게 표시)
                scale = 1
                if height < max_size and width < max_size:
                    # 적절한 scale 계산 (최대 1200x1200 이하)
                    scale = min(max_size // height, max_size // width, 6)  # 최대 6배까지 확대
                
                if scale > 1:
                    new_width = width * scale
                    new_height = height * scale
                    img_bgr = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                
                # 이미지 창에 표시 (한글 제거 - OpenCV 호환성)
                cv2.imshow('Scenario 2: Blue Pillar -> Turn Right -> Table', img_bgr)
            except Exception as e:
                print(f"이미지 처리 오류: {e}")
        
        # cv2.waitKey()는 imshow() 직후에 호출되어야 키 입력을 받을 수 있음 (keyboard_control.py 주석 참고)
        # keyboard_control.py와 동일하게 30ms 대기 (1ms는 너무 짧아서 창이 업데이트되기 전에 다음 루프로 넘어갈 수 있음)
        key = cv2.waitKey(30) & 0xFF  # 30ms 대기
        if key == 27 or key == ord('q'):  # ESC 또는 'q' 키
            break
    
    # 리소스 정리
    cv2.destroyAllWindows()
    wrapper.close()
    print("\n시각화 종료.")


def main():
    """
    메인 함수
    """
    try:
        visualize_scenario2()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

