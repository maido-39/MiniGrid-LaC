######################################################
#                                                    #
#                     VIZUALIZER                     #
#                                                    #
######################################################


""""""




######################################################
#                                                    #
#                      LIBRARIES                     #
#                                                    #
######################################################


import cv2
import numpy as np

from utils.map_manager.minigrid_customenv_emoji import MiniGridEmojiWrapper




######################################################
#                                                    #
#                       CLASS                        #
#                                                    #
######################################################


class Visualizer:
    """Visualization Class"""
    
    def __init__(self, window_name: str = "Scenario 2: VLM Control (Absolute)"):
        self.window_name = window_name
    
    def visualize_grid_cli(self, wrapper: MiniGridEmojiWrapper, state: dict):
        """Visualize the grid as text in the CLI"""
        
        env = wrapper.env
        size = wrapper.size
        
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
                cell = env.grid.get(x, y)
                
                if x == agent_x and y == agent_y:
                    row.append(agent_symbol)
                elif cell is not None and cell.type == 'wall':
                    if hasattr(cell, 'color'):
                        if cell.color == 'blue':
                            row.append('🟦')
                        elif cell.color == 'purple':
                            row.append('🟪')
                        elif cell.color == 'red':
                            row.append('🟥')
                        elif cell.color == 'green':
                            row.append('🟩')
                        elif cell.color == 'yellow':
                            row.append('🟨')
                        else:
                            row.append('⬛')
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

    def display_image(self, img: np.ndarray):
        """Displaying images using OpenCV"""
        
        if img is not None:
            try:
                img_bgr = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
                
                height, width = img_bgr.shape[:2]
                max_size = 800
                if height < max_size and width < max_size:
                    scale = min(max_size // height, max_size // width, 4)
                    if scale > 1:
                        new_width = width * scale
                        new_height = height * scale
                        img_bgr = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                
                cv2.imshow(self.window_name, img_bgr)
                cv2.waitKey(1)
            except Exception as e:
                print(f"Image display error: {e}")
    
    def cleanup(self):
        """Resource Cleanup"""
        
        cv2.destroyAllWindows()



