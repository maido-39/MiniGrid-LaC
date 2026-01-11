"""
이미지 기반 맵 환경 생성 모듈

제공된 맵을 기반으로 환경을 생성합니다.

레이아웃 (14x14):
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬜️⬜️⬜️🟩🟩🟩⬜️🟦🟦🟦⬜️⬜️⬛
⬛⬜️⬜️⬜️🟩🟩🟩⬜️🟦🟦🟦⬜️⬜️⬛
⬛⬜️⬜️⬜️🟩🟩🟩⬜🟦🟦🟦⬜️⬜️⬛
⬛⬜️⬜️⬜️⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟩🟩🟩⬜️⬜️🟩🟩🟩⬜️⬜️⬜️⬜️⬛
⬛🟩🟩🟩⬜️⬜️🟩🟩🟩⬜️⬜️⬜️⬜️⬛
⬛🟩🟩🟩⬜️⬜️🟩🟩🟩⬜️🟩🟩🟩⬛
⬛⬜️⬜️⬜️⬜️⬛⬜️⬜️⬜️⬜️🟩🟩🟩⬛
⬛⬜️⬜️⬜️⬜️⬛⬜️⬜️⬜️⬜️🟩🟩🟩⬛
⬛⬛⬛⬛⬛⬛⬜️⬜️⬜️⬛⬛⬛⬛⬛
⬛⬜️⬜️⬜️⬜️🟩🟩🟩⬜️⬜️⬜️⬜️⬜️⬛
⬛⬜️🟥⬜️⬜️🟩🟩🟩⬜️⬜️⬜️⬜️⬜️⬛
⬛⬜️⬜️⬜️⬜️🟩🟩🟩⬜️⬜️⬜️⬜️⬜️⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛

범례:
⬛ = 외벽 및 내부 회색 블록
🟦 = 파란색 블록 (벽)
🟥 = 빨간색 블록 (벽)
🟩 = 초록색 블록 (🧱 이모지, 통과 불가)
⬜️ = 빈 공간
"""

from minigrid_vlm_interact_absolute_emoji import AbsoluteDirectionEmojiWrapper
from typing import Dict


class ImageMapEnvironment:
    """이미지 기반 맵 환경 생성 클래스"""
    
    def __init__(self, size: int = 14):
        """
        Args:
            size: 그리드 크기 (기본값: 14)
        """
        self.size = size
    
    def create_room_config(self) -> Dict:
        """
        제공된 맵을 기반으로 room_config 생성
        """
        walls = []
        
        # 외벽 생성 (회색)
        for i in range(self.size):
            walls.append((i, 0))
            walls.append((i, self.size-1))
            walls.append((0, i))
            walls.append((self.size-1, i))
        
        # 내부 회색 블록
        # Row 4: (4,4)부터 (10,4)까지
        internal_grey_blocks = [
            (4, 4), (5, 4), (6, 4), (7, 4), (8, 4), (9, 4), (10, 4),
            (5, 8),  # Row 8
            (5, 9),  # Row 9
            (0, 10), (1, 10), (2, 10), (3, 10), (4, 10),  # Row 10 왼쪽
            (9, 10), (10, 10), (11, 10)  # Row 10 오른쪽
        ]
        for pos in internal_grey_blocks:
            if 0 < pos[0] < self.size-1 and 0 < pos[1] < self.size-1:
                walls.append((pos[0], pos[1], 'grey'))
        
        # 파란색 블록
        blue_blocks = [
            (8, 1), (9, 1), (10, 1),  # Row 1
            (8, 2), (9, 2), (10, 2),  # Row 2
            (7, 3), (8, 3), (9, 3)    # Row 3
        ]
        for pos in blue_blocks:
            if 0 < pos[0] < self.size-1 and 0 < pos[1] < self.size-1:
                walls.append((pos[0], pos[1], 'blue'))
        
        # 빨간색 블록
        red_blocks = [(2, 12)]  # Row 12
        for pos in red_blocks:
            if 0 < pos[0] < self.size-1 and 0 < pos[1] < self.size-1:
                walls.append((pos[0], pos[1], 'red'))
        
        # 초록색 블록 (이모지로 표현)
        green_blocks = [
            # Row 1-3: 상단 왼쪽
            (4, 1), (5, 1), (6, 1),
            (4, 2), (5, 2), (6, 2),
            (4, 3), (5, 3), (6, 3),
            # Row 5-7: 중간 왼쪽 및 중앙
            (1, 5), (2, 5), (3, 5), (6, 5), (7, 5), (8, 5),
            (1, 6), (2, 6), (3, 6), (6, 6), (7, 6), (8, 6),
            (1, 7), (2, 7), (3, 7), (6, 7), (7, 7), (8, 7),
            # Row 7: 오른쪽
            (11, 7), (12, 7), (13, 7),
            # Row 8-9: 오른쪽
            (11, 8), (12, 8), (13, 8),
            (11, 9), (12, 9), (13, 9),
            # Row 11-13: 하단
            (5, 11), (6, 11), (7, 11),
            (5, 12), (6, 12), (7, 12),
            (5, 13), (6, 13), (7, 13)
        ]
        
        objects = []
        for pos in green_blocks:
            if 0 < pos[0] < self.size-1 and 0 < pos[1] < self.size-1:
                # 초록색 블록을 brick 이모지로 표현
                objects.append({
                    'type': 'emoji',
                    'pos': pos,
                    'emoji_name': 'brick',
                    'color': 'green',
                    'can_pickup': False,
                    'can_overlap': False,  # 통과 불가
                    'use_emoji_color': True
                })
        
        # 시작점과 종료점 설정
        # 적절한 빈 공간 선택
        start_pos = (1, 1)  # 상단 왼쪽 빈 공간
        goal_pos = (12, 1)  # 상단 오른쪽 빈 공간
        
        room_config = {
            'start_pos': start_pos,
            'goal_pos': goal_pos,
            'walls': walls,
            'objects': objects,
            # 로봇 이모지 설정
            'use_robot_emoji': True,
            'robot_emoji_color': 'red',
            'use_robot_emoji_color': True
        }
        
        return room_config
    
    def create_wrapper(self) -> AbsoluteDirectionEmojiWrapper:
        """환경 Wrapper 생성"""
        room_config = self.create_room_config()
        return AbsoluteDirectionEmojiWrapper(size=self.size, room_config=room_config)


def create_image_map_environment(size: int = 14) -> AbsoluteDirectionEmojiWrapper:
    """
    이미지 기반 맵 환경 생성 함수
    
    Args:
        size: 그리드 크기 (기본값: 14)
    
    Returns:
        AbsoluteDirectionEmojiWrapper: 이미지 기반 맵이 포함된 환경
    """
    map_env = ImageMapEnvironment(size=size)
    return map_env.create_wrapper()
