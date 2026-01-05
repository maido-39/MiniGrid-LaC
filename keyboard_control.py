"""
키보드로 MiniGrid 환경을 제어하는 예제 스크립트

사용법:
    python keyboard_control.py

조작법:
    - 'w': 앞으로 이동 (move forward)
    - 'a': 왼쪽으로 회전 (turn left)
    - 'd': 오른쪽으로 회전 (turn right)
    - 's': 뒤로 이동 (move backward) - 일부 환경에서만 지원
    - 'r': 환경 리셋
    - 'q': 종료
"""

from minigrid import register_minigrid_envs
import gymnasium as gym
import cv2

# MiniGrid 환경 등록 (필수: 환경을 사용하기 전에 등록해야 함)
register_minigrid_envs()


def get_key():
    """
    키보드 입력을 받아서 액션으로 변환하는 함수
    
    OpenCV의 waitKey()를 사용하여 키보드 입력을 받고,
    입력된 키에 따라 액션 인덱스 또는 특수 명령을 반환합니다.
    
    Returns:
        int 또는 str: 액션 인덱스 (0-3) 또는 특수 명령 ('quit', 'reset')
        None: 유효하지 않은 키 입력 또는 키 입력 없음
    """
    # 키보드 입력 확인 (1ms 대기, -1이면 입력 없음)
    key = cv2.waitKey(1) & 0xFF
    
    # 키 입력이 없으면 None 반환
    if key == 0 or key == 255:  # 0 또는 255는 키 입력 없음
        return None
    
    # 'q' 키: 프로그램 종료
    if key == ord('q'):
        return 'quit'
    # 'r' 키: 환경 리셋
    elif key == ord('r'):
        return 'reset'
    # 'w' 키: 앞으로 이동 (액션 2)
    elif key == ord('w'):
        return 2  # move forward
    # 's' 키: 뒤로 이동 (액션 3)
    elif key == ord('s'):
        return 3  # move backward
    # 'a' 키: 왼쪽으로 회전 (액션 0)
    elif key == ord('a'):
        return 0  # turn left
    # 'd' 키: 오른쪽으로 회전 (액션 1)
    elif key == ord('d'):
        return 1  # turn right
    # 그 외의 키: 무시
    else:
        return None


def main():
    """
    메인 함수: 키보드로 환경을 제어하는 메인 루프
    
    이 함수는 다음 순서로 동작합니다:
    1. 표준 MiniGrid 환경 생성
    2. 환경 초기화 (reset)
    3. 메인 루프:
       - 현재 환경 이미지 가져오기
       - OpenCV로 이미지 표시
       - 키보드 입력 받기
       - 액션 실행
       - 상태 출력
    """
    # 표준 MiniGrid 환경 생성
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode='rgb_array')
    
    print("=" * 50)
    print("MiniGrid 키보드 제어 예제")
    print("=" * 50)
    print("\n조작법:")
    print("  - w/a/s/d: 에이전트 이동 (w: 앞으로, a: 왼쪽 회전, d: 오른쪽 회전, s: 뒤로)")
    print("  - 'r': 환경 리셋")
    print("  - 'q': 종료")
    print("\n환경을 시작합니다...\n")
    
    # 환경 초기화
    obs, info = env.reset()
    done = False
    step_count = 0
    
    # 메인 루프: 키보드 입력을 받아 환경을 제어
    while True:
        # 환경 렌더링
        img = env.render()
        
        # OpenCV로 이미지 표시
        # MiniGrid는 (H, W, C) 형식이므로 BGR로 변환
        if img is not None:
            # RGB를 BGR로 변환 (OpenCV는 BGR 형식을 사용)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # 이미지 크기 조정 (적절한 크기로 확대, 최대 크기 제한)
            # 원본 이미지가 256x256이므로 적절한 크기로 확대
            height, width = img_bgr.shape[:2]
            # 최대 크기를 800x800으로 제한하여 적절한 크기로 조정
            max_size = 800
            if height > max_size or width > max_size:
                # 이미지가 이미 크면 그대로 사용
                scale = 1
            else:
                # 적절한 scale 계산 (최대 800x800 이하)
                scale = min(max_size // height, max_size // width, 4)  # 최대 4배까지만 확대
            
            if scale > 1:
                new_width = width * scale
                new_height = height * scale
                img_bgr = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            
            # 이미지 창에 표시
            cv2.imshow('MiniGrid Environment', img_bgr)
            
            # cv2.waitKey()는 imshow() 직후에 호출되어야 키 입력을 받을 수 있음
            key = cv2.waitKey(30) & 0xFF  # 30ms 대기
            
            # 키 입력 처리
            if key != 0 and key != 255:  # 키 입력이 있으면
                action = None
                if key == ord('q'):
                    action = 'quit'
                elif key == ord('r'):
                    action = 'reset'
                elif key == ord('w'):
                    action = 2  # move forward
                elif key == ord('s'):
                    action = 3  # move backward
                elif key == ord('a'):
                    action = 0  # turn left
                elif key == ord('d'):
                    action = 1  # turn right
            else:
                action = None
        else:
            action = None
        
        # 키 입력이 없으면 계속 렌더링만 수행
        if action is None:
            continue
        
        # 'q' 키 입력: 프로그램 종료
        if action == 'quit':
            print("\n프로그램을 종료합니다.")
            break
        # 'r' 키 입력: 환경 리셋
        elif action == 'reset':
            print("\n환경을 리셋합니다...")
            obs, info = env.reset()
            done = False
            step_count = 0
            continue
        
        # 액션 실행
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        
        # 상태 출력 (디버깅 및 모니터링용)
        print(f"Step {step_count}: Action={action}, Reward={reward}, Done={done}")
        
        # 에피소드 종료 시 메시지 출력
        if done:
            print(f"\n에피소드 종료! 총 {step_count} 스텝, 최종 보상: {reward}")
            print("'r'를 눌러 리셋하거나 'q'를 눌러 종료하세요.")
    
    # 리소스 정리
    cv2.destroyAllWindows()  # OpenCV 창 닫기
    env.close()  # 환경 종료 및 리소스 정리


if __name__ == "__main__":
    main()
