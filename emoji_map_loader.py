"""
이모지 맵 JSON 로더 및 변환 모듈

JSON 파일에서 이모지 맵을 읽어서 minigrid 환경으로 변환합니다.

JSON 구조:
{
  "map": {
    "emoji_render": "⬛⬛⬛⬛⬛...\n⬛⬜️⬜️⬜️...\n..." 
    또는
    "emoji_render": [
      "⬛⬛⬛⬛⬛...",
      "⬛⬜️⬜️⬜️...",
      ...
    ]
    또는
    "emoji_render": [
      ["⬛", "⬛", "⬛", ...],
      ["⬛", "⬜️", "⬜️", ...],
      ...
    ],
    "emoji_objects": {
      "⬛": {
        "type": "wall",
        "color": "grey",
        "can_pickup": false,
        "can_overlap": false
      },
      "🟩": {
        "type": "emoji",
        "emoji_name": "brick",
        "color": "green",
        "can_pickup": false,
        "can_overlap": false,
        "use_emoji_color": true
      },
      ...
    },
    "robot_config": {
      "use_robot_emoji": true,
      "robot_emoji_color": "red",
      "use_robot_emoji_color": true
    },
    "start_pos": [1, 1],
    "goal_pos": [12, 1]
  }
}
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from minigrid_customenv_emoji import MiniGridEmojiWrapper


class EmojiMapLoader:
    """이모지 맵 JSON 로더 및 변환 클래스"""
    
    def __init__(self, json_path: str):
        """
        Args:
            json_path: JSON 파일 경로
        """
        self.json_path = Path(json_path)
        self.map_data = None
        self.emoji_render = None
        self.emoji_objects = None
        self.robot_config = None
        self.start_pos = None
        self.goal_pos = None
        self.size = None
        self.num_rows = None
        self.num_cols = None
        
        self._load_json()
        self._parse_map_data()
    
    def _load_json(self):
        """JSON 파일 로드"""
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {self.json_path}")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'map' not in data:
            raise ValueError("JSON 파일에 'map' 키가 없습니다.")
        
        self.map_data = data['map']
    
    def _parse_emoji_text(self, text: str) -> List[List[str]]:
        """
        텍스트 형태의 이모지 맵을 2D 배열로 파싱
        
        Args:
            text: 줄바꿈으로 구분된 이모지 맵 텍스트
        
        Returns:
            2D 배열 (행 리스트의 리스트)
        """
        lines = text.strip().split('\n')
        # 빈 줄 제거
        lines = [line.strip() for line in lines if line.strip()]
        
        if len(lines) == 0:
            raise ValueError("이모지 맵 텍스트가 비어있습니다.")
        
        result = []
        for line in lines:
            emojis = []
            i = 0
            while i < len(line):
                char = line[i]
                # Variation Selector (U+FE0F)나 Zero Width Joiner (U+200D)는 이전 문자와 함께 묶음
                if ord(char) in [0xFE0F, 0x200D]:
                    # 이전 이모지에 추가 (이미 추가된 경우)
                    if emojis:
                        emojis[-1] += char
                    i += 1
                    continue
                
                # 다음 문자가 Variation Selector나 Zero Width Joiner인지 확인
                if i + 1 < len(line):
                    next_char = line[i + 1]
                    if ord(next_char) in [0xFE0F, 0x200D]:
                        # Variation Selector나 Zero Width Joiner가 있으면 함께 묶음
                        if i + 2 < len(line) and ord(line[i + 2]) in [0xFE0F, 0x200D]:
                            # 두 개의 조합 문자가 있는 경우 (드물지만 가능)
                            emojis.append(line[i:i+3])
                            i += 3
                        else:
                            emojis.append(line[i:i+2])
                            i += 2
                    else:
                        emojis.append(char)
                        i += 1
                else:
                    emojis.append(char)
                    i += 1
            
            # 빈 문자열이나 공백만 있는 항목 제거
            emojis = [e for e in emojis if e.strip()]
            if emojis:
                result.append(emojis)
        
        return result
    
    def _parse_map_data(self):
        """맵 데이터 파싱"""
        # 이모지 렌더 맵
        if 'emoji_render' not in self.map_data:
            raise ValueError("JSON 파일에 'emoji_render' 키가 없습니다.")
        
        emoji_render_raw = self.map_data['emoji_render']
        
        # 텍스트 형태인지 확인 (문자열 또는 문자열 배열)
        if isinstance(emoji_render_raw, str):
            # 단일 문자열: 줄바꿈으로 구분
            self.emoji_render = self._parse_emoji_text(emoji_render_raw)
        elif isinstance(emoji_render_raw, list) and len(emoji_render_raw) > 0:
            # 첫 번째 요소가 문자열이면 텍스트 배열 형태
            if isinstance(emoji_render_raw[0], str):
                # 문자열 배열: 각 줄이 문자열
                text = '\n'.join(emoji_render_raw)
                self.emoji_render = self._parse_emoji_text(text)
            else:
                # 2D 배열 형태 (기존 방식)
                self.emoji_render = emoji_render_raw
        else:
            raise ValueError("'emoji_render'는 문자열, 문자열 배열, 또는 2D 배열이어야 합니다.")
        
        # 맵 크기 확인
        if not isinstance(self.emoji_render, list) or len(self.emoji_render) == 0:
            raise ValueError("'emoji_render' 파싱 결과가 비어있습니다.")
        
        # 행 수와 열 수를 이모지 리스트에서 가져옴
        self.num_rows = len(self.emoji_render)
        row_lengths = [len(row) for row in self.emoji_render]
        
        # 모든 행의 길이가 같아야 함
        if not all(length == row_lengths[0] for length in row_lengths):
            raise ValueError(
                f"맵의 모든 행은 같은 길이여야 합니다. "
                f"행 수: {self.num_rows}, 각 행의 길이: {row_lengths}"
            )
        
        self.num_cols = row_lengths[0]
        
        # MiniGrid는 정사각형 그리드를 사용하므로, 행 수와 열 수 중 더 큰 값을 사용
        self.size = max(self.num_rows, self.num_cols)
        
        # 행 수와 열 수가 다르면 경고
        if self.num_rows != self.num_cols:
            print(f"경고: 맵이 정사각형이 아닙니다. 행 수: {self.num_rows}, 열 수: {self.num_cols}, "
                  f"그리드 크기: {self.size}x{self.size}로 설정됩니다.")
        
        # 이모지 객체 정의
        if 'emoji_objects' not in self.map_data:
            raise ValueError("JSON 파일에 'emoji_objects' 키가 없습니다.")
        
        self.emoji_objects = self.map_data['emoji_objects']
        
        # 로봇 설정
        self.robot_config = self.map_data.get('robot_config', {
            'use_robot_emoji': True,
            'robot_emoji_color': 'red',
            'use_robot_emoji_color': True
        })
        
        # 시작점과 종료점
        self.start_pos = tuple(self.map_data.get('start_pos', [1, 1]))
        self.goal_pos = tuple(self.map_data.get('goal_pos', [self.size - 2, self.size - 2]))
    
    def _parse_emoji_map(self) -> Tuple[List, List]:
        """
        이모지 맵을 파싱하여 walls와 objects 리스트 생성
        
        Returns:
            (walls, objects): 벽 리스트와 객체 리스트
        """
        walls = []
        objects = []
        
        for y, row in enumerate(self.emoji_render):
            for x, emoji in enumerate(row):
                # 이모지 정의 확인
                if emoji not in self.emoji_objects:
                    # 정의되지 않은 이모지는 무시 (또는 경고)
                    continue
                
                emoji_def = self.emoji_objects[emoji]
                obj_type = emoji_def.get('type', 'wall')
                
                # 로봇 위치 마커 확인 (🟥는 로봇 위치를 나타내는 마커)
                # start_pos가 설정되어 있지 않거나, 🟥가 start_pos 위치에 있으면 마커로 처리
                if emoji == '🟥' and obj_type == 'wall':
                    # 🟥는 로봇 위치 마커로 처리 (벽으로 추가하지 않음)
                    # start_pos가 명시적으로 설정되지 않았거나, 🟥 위치와 일치하면 start_pos 업데이트
                    if self.start_pos == (1, 1) or self.start_pos == (x, y):
                        self.start_pos = (x, y)
                    continue
                
                # 벽 추가 (외벽 포함, CustomRoomEnv가 자동으로 외벽을 생성하지만
                # emoji_render에 명시된 외벽도 처리하여 색상 등을 반영)
                if obj_type == 'wall':
                    color = emoji_def.get('color', 'grey')
                    # 모든 벽 추가 (외벽 포함)
                    walls.append((x, y, color))
                
                elif obj_type == 'emoji':
                    # 이모지 객체 생성
                    obj_config = {
                        'type': 'emoji',
                        'pos': (x, y),
                        'emoji_name': emoji_def.get('emoji_name', 'emoji'),
                        'color': emoji_def.get('color', 'yellow'),
                        'can_pickup': emoji_def.get('can_pickup', False),
                        'can_overlap': emoji_def.get('can_overlap', False),
                        'use_emoji_color': emoji_def.get('use_emoji_color', True)
                    }
                    # 외벽이 아닌 경우만 추가
                    if 0 < x < self.size - 1 and 0 < y < self.size - 1:
                        objects.append(obj_config)
                
                elif obj_type == 'empty' or obj_type == 'space':
                    # 빈 공간은 아무것도 하지 않음
                    pass
        
        return walls, objects
    
    def create_room_config(self) -> Dict:
        """
        room_config 생성
        
        Returns:
            room_config 딕셔너리
        """
        walls, objects = self._parse_emoji_map()
        
        # CustomRoomEnv는 자동으로 외벽을 생성하므로 외벽은 추가하지 않음
        # 하지만 emoji_render에 외벽이 명시적으로 표시되어 있으면 추가
        # (외벽 위치의 벽은 이미 _parse_emoji_map에서 처리됨)
        
        room_config = {
            'start_pos': self.start_pos,
            'goal_pos': self.goal_pos,
            'walls': walls,
            'objects': objects,
            **self.robot_config  # 로봇 설정 병합
        }
        
        return room_config
    
    def create_wrapper(self) -> MiniGridEmojiWrapper:
        """
        MiniGridEmojiWrapper 생성 (절대 움직임 모드 활성화)
        
        Returns:
            MiniGridEmojiWrapper 인스턴스 (use_absolute_movement=True)
        """
        room_config = self.create_room_config()
        return MiniGridEmojiWrapper(size=self.size, room_config=room_config, use_absolute_movement=True)


def load_emoji_map_from_json(json_path: str) -> MiniGridEmojiWrapper:
    """
    JSON 파일에서 이모지 맵을 로드하여 환경 생성
    
    Args:
        json_path: JSON 파일 경로
    
    Returns:
        MiniGridEmojiWrapper: 생성된 환경 (절대 움직임 모드 활성화)
    """
    loader = EmojiMapLoader(json_path)
    return loader.create_wrapper()


if __name__ == "__main__":
    # 사용 예제
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python emoji_map_loader.py <json_file_path>")
        print("예제: python emoji_map_loader.py example_map.json")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    print(f"JSON 파일에서 맵 로드 중: {json_path}")
    wrapper = load_emoji_map_from_json(json_path)
    
    print("환경 초기화 중...")
    wrapper.reset()
    
    state = wrapper.get_state()
    print(f"에이전트 위치: {state['agent_pos']}")
    print(f"에이전트 방향: {state['agent_dir']}")
    
    print("\n맵이 성공적으로 로드되었습니다!")

