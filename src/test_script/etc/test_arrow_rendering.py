"""
에이전트 arrow.png 렌더링 테스트 스크립트
"""
from minigrid import register_minigrid_envs
# Actual path: legacy.relative_movement.custom_environment
from legacy import CustomRoomWrapper
import cv2
import numpy as np

# MiniGrid 환경 등록
register_minigrid_envs()

def test_arrow_rendering():
    """arrow.png 렌더링 테스트"""
    print("=" * 60)
    print("에이전트 arrow.png 렌더링 테스트")
    print("=" * 60)
    
    # 환경 생성
    print("\n[1] 환경 생성 중...")
    wrapper = CustomRoomWrapper(
        size=10,
        room_config={
            'start_pos': (1, 1),
            'walls': []
        }
    )
    wrapper.reset()
    
    print(f"에이전트 위치: {wrapper.get_state()['agent_pos']}")
    print(f"에이전트 방향: {wrapper.get_state()['agent_dir']}")
    
    # 이미지 렌더링
    print("\n[2] 이미지 렌더링 중...")
    image = wrapper.get_image()
    
    if image is None:
        print("오류: 이미지가 None입니다.")
        return
    
    print(f"이미지 크기: {image.shape}")
    
    # OpenCV로 표시
    print("\n[3] 이미지 표시 중...")
    print("키보드 입력을 기다립니다...")
    print("  - 'q': 종료")
    print("  - 'w': 앞으로 이동")
    print("  - 'a': 왼쪽 회전")
    print("  - 'd': 오른쪽 회전")
    print("  - 's': 뒤로 이동")
    
    while True:
        # 이미지 표시
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width = img_bgr.shape[:2]
        max_size = 800
        if height < max_size and width < max_size:
            scale = min(max_size // height, max_size // width, 4)
            if scale > 1:
                new_width = width * scale
                new_height = height * scale
                img_bgr = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow('Arrow Rendering Test', img_bgr)
        
        # 키 입력 처리
        key = cv2.waitKey(100) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('w'):
            wrapper.step(2)  # move forward
            print("앞으로 이동")
        elif key == ord('a'):
            wrapper.step(0)  # turn left
            print("왼쪽 회전")
        elif key == ord('d'):
            wrapper.step(1)  # turn right
            print("오른쪽 회전")
        elif key == ord('s'):
            wrapper.step(3)  # move backward
            print("뒤로 이동")
        
        # 상태 업데이트
        state = wrapper.get_state()
        print(f"위치: {state['agent_pos']}, 방향: {state['agent_dir']}")
        
        # 이미지 다시 렌더링
        image = wrapper.get_image()
    
    cv2.destroyAllWindows()
    wrapper.close()
    print("\n테스트 완료!")

if __name__ == "__main__":
    try:
        test_arrow_rendering()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

