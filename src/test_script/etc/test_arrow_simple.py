"""
간단한 arrow.png 렌더링 테스트 (이미지 저장)
"""
from minigrid import register_minigrid_envs
# Actual path: legacy.relative_movement.custom_environment
from legacy import CustomRoomWrapper
from PIL import Image
import numpy as np

# MiniGrid 환경 등록
register_minigrid_envs()

def test_arrow_simple():
    """arrow.png 렌더링 간단 테스트"""
    print("=" * 60)
    print("에이전트 arrow.png 렌더링 테스트 (이미지 저장)")
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
    
    state = wrapper.get_state()
    print(f"에이전트 위치: {state['agent_pos']}")
    print(f"에이전트 방향: {state['agent_dir']}")
    
    # 이미지 렌더링
    print("\n[2] 이미지 렌더링 중...")
    image = wrapper.get_image()
    
    if image is None:
        print("오류: 이미지가 None입니다.")
        return
    
    print(f"이미지 크기: {image.shape}")
    print(f"이미지 dtype: {image.dtype}")
    print(f"이미지 min/max: {image.min()}/{image.max()}")
    
    # 이미지 저장
    print("\n[3] 이미지 저장 중...")
    img_pil = Image.fromarray(image.astype(np.uint8))
    img_pil.save('test_arrow_output.png')
    print("이미지가 'test_arrow_output.png'로 저장되었습니다.")
    
    # 방향 변경 테스트
    print("\n[4] 방향 변경 테스트...")
    for direction_name, action in [("왼쪽", 0), ("오른쪽", 1), ("앞으로", 2)]:
        wrapper.step(action)
        state = wrapper.get_state()
        print(f"  {direction_name} -> 위치: {state['agent_pos']}, 방향: {state['agent_dir']}")
        image = wrapper.get_image()
        img_pil = Image.fromarray(image.astype(np.uint8))
        img_pil.save(f'test_arrow_{direction_name}.png')
        print(f"  이미지 저장: test_arrow_{direction_name}.png")
    
    wrapper.close()
    print("\n테스트 완료!")

if __name__ == "__main__":
    try:
        test_arrow_simple()
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

