import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Wall, Floor
from minigrid.minigrid_env import MiniGridEnv

class CustomEmojiMapEnv(MiniGridEnv):
    def __init__(self, render_mode='rgb_array', **kwargs):
        mission_space = MissionSpace(mission_func=lambda: "Reach the goal")
        
        # 이모지 맵 크기에 맞춰 15x15로 설정
        super().__init__(
            mission_space=mission_space,
            grid_size=15,
            max_steps=100,
            render_mode=render_mode,
            **kwargs
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # --- 1. 내부 벽(Wall) 배치 ---
        
        # Row 4: 좌측(x=1~3)은 뚫려있고, x=4부터 끝까지 벽
        # ⬛⬜️⬜️⬜️⬛⬛⬛...
        self.grid.horz_wall(4, 4, length=11)

        # Row 8, 9: x=6 위치에 세로 벽 (중간 연결부)
        # 8번, 9번 줄에 벽이 하나씩 내려옴
        self.grid.set(6, 8, Wall())
        self.grid.set(6, 9, Wall())

        # Row 10: 중간(x=7~9)만 뚫려있고 나머지 벽
        # ⬛⬛⬛⬛⬛⬛⬛⬜️⬜️⬜️⬛⬛⬛⬛⬛
        self.grid.horz_wall(0, 10, length=7)  # x=0~6
        self.grid.horz_wall(10, 10, length=5) # x=10~14


        # --- 2. 색상 바닥(Floor) 배치 ---

        # 🟪 보라색 (Top Middle): x=5, y=1
        self._fill_color(5, 1, 3, 3, 'purple')

        # 🟥 빨간색 (Top Right): x=10, y=1 (플레이어가 아닌 장식 바닥)
        self._fill_color(10, 1, 3, 3, 'red')

        # 🟦 파란색 (Middle Left): x=1, y=5
        self._fill_color(1, 5, 3, 3, 'blue')

        # 🟩 초록색 (Middle Center): x=7, y=5
        self._fill_color(7, 5, 3, 3, 'green')

        # 🟨 노란색 (Middle Right): x=11, y=7 (약간 아래로 처짐)
        self._fill_color(11, 7, 3, 3, 'yellow')

        # 🟧 주황색 (Bottom Center): x=7, y=11
        # *Minigrid에 Orange가 없어 Yellow로 대체합니다.
        self._fill_color(7, 11, 3, 3, 'yellow') 


        # --- 3. 플레이어 배치 ---
        
        # 왼쪽 아래 빨간색 지점 (x=2, y=12)
        # ⬛⬜️🟥⬜️...
        self.agent_pos = (2, 12)
        self.agent_dir = 0 # 오른쪽을 보게 설정

    def _fill_color(self, x, y, w, h, color):
        for i in range(x, x + w):
            for j in range(y, y + h):
                self.grid.set(i, j, Floor(color))

# 실행 및 시각화
if __name__ == "__main__":
    env = CustomEmojiMapEnv()
    env.reset()

    # 렌더링 (타일 크기를 키워서 잘 보이게 설정)
    img = env.get_frame(highlight=True, tile_size=32)

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Emoji Map Recreated")
    plt.show()