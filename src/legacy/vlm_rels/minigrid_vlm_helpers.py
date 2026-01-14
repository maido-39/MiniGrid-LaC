"""
MiniGrid 환경 전용 헬퍼 함수들

환경 특정 시각화 및 유틸리티 함수들을 제공합니다.
VLM 컨트롤러는 이 헬퍼를 사용하여 환경 특정 기능을 수행합니다.
"""

import numpy as np
from typing import Dict


def visualize_minigrid_grid_cli(env, state: dict):
    """
    CLI에서 MiniGrid 그리드를 텍스트로 시각화
    
    Args:
        env: MiniGridEmojiWrapper 인스턴스
        state: 환경 상태 딕셔너리
    """
    minigrid_env = env.env
    size = env.size
    
    agent_pos = state['agent_pos']
    if isinstance(agent_pos, np.ndarray):
        agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
    else:
        agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
    
    agent_dir = state['agent_dir']
    direction_symbols = {0: '>', 1: 'v', 2: '<', 3: '^'}
    agent_symbol = direction_symbols.get(agent_dir, 'A')
    
    grid_chars = []
    for y in range(size):
        row = []
        for x in range(size):
            cell = minigrid_env.grid.get(x, y)
            
            if x == agent_x and y == agent_y:
                row.append(agent_symbol)
            elif cell is not None and cell.type == 'wall':
                if hasattr(cell, 'color'):
                    color_map = {
                        'blue': '🟦',
                        'purple': '🟪',
                        'red': '🟥',
                        'green': '🟩',
                        'yellow': '🟨'
                    }
                    row.append(color_map.get(cell.color, '⬛'))
                else:
                    row.append('⬛')
            elif cell is not None and cell.type == 'goal':
                row.append('🟩')
            elif cell is not None:
                if hasattr(cell, 'color'):
                    if cell.color == 'blue':
                        row.append('🟦')
                    elif cell.color == 'purple':
                        row.append('🟪')
                    else:
                        row.append('🟨')
                else:
                    row.append('🟨')
            else:
                row.append('⬜️')
        grid_chars.append(row)
    
    print("\n" + "=" * 60)
    print("Current Grid State:")
    print("=" * 60)
    for y in range(size):
        print(''.join(grid_chars[y]))
    print("=" * 60)
    print(f"Agent Position: ({agent_x}, {agent_y}), Direction: {agent_dir} ({agent_symbol})")
    print("=" * 60 + "\n")

