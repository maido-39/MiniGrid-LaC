"""
MiniGrid VLM Interaction Script (Absolute Coordinate Movement Version)

Control and visualize the MiniGrid environment using VLM.
It provides an absolute coordinate-based action space where the robot can move directly up, down, left, or right.

Usage:
    python minigrid_vlm_interact_absolute.py
"""

from minigrid import register_minigrid_envs
# Actual path: legacy.relative_movement.custom_environment
from legacy import CustomRoomWrapper
# Actual paths: utils.vlm.vlm_wrapper, utils.vlm.vlm_postprocessor
from utils import ChatGPT4oVLMWrapper, VLMResponsePostProcessor
import numpy as np
import cv2
from typing import Union, Tuple, Dict, Optional

# MiniGrid Environment Registration
register_minigrid_envs()

# VLM Settings
VLM_MODEL = "gpt-4o"
VLM_TEMPERATURE = 0.0
VLM_MAX_TOKENS = 1000


class AbsoluteDirectionWrapper(CustomRoomWrapper):
    """
    Wrapper supporting absolute directional movement (up/down/left/right)
    
    Extends the existing CustomRoomWrapper to provide an action space enabling direct movement up, down, left, and right. Movement is based on an absolute coordinate system, independent of the robot's current orientation.
    """
    
    # Absolute Direction Action Name and Index Mapping
    ABSOLUTE_ACTION_NAMES = {
        0: "move up",      # North (위)
        1: "move down",    # South (아래)
        2: "move left",    # West (왼쪽)
        3: "move right",   # East (오른쪽)
        4: "pickup",
        5: "drop",
        6: "toggle"
    }
    
    # Absolute Direction Action Alias
    ABSOLUTE_ACTION_ALIASES = {
        # Above (North)
        "move up": 0, "up": 0, "north": 0, "n": 0, "move north": 0,
        "go up": 0, "go north": 0,
        # Below (South)
        "move down": 1, "down": 1, "south": 1, "s": 1, "move south": 1,
        "go down": 1, "go south": 1,
        # Left (West)
        "move left": 2, "left": 2, "west": 2, "w": 2, "move west": 2,
        "go left": 2, "go west": 2,
        # Right (East)
        "move right": 3, "right": 3, "east": 3, "e": 3, "move east": 3,
        "go right": 3, "go east": 3,
        # Other actions
        "pickup": 4, "pick up": 4, "pick_up": 4, "grab": 4,
        "drop": 5, "put down": 5, "put_down": 5, "release": 5,
        "toggle": 6, "interact": 6, "use": 6, "activate": 6
    }
    
    # MiniGrid Direction Mapping (0=East, 1=South, 2=West, 3=North)
    DIRECTION_TO_AGENT_DIR = {
        "north": 3,  # Above
        "south": 1,  # Below
        "west": 2,   # Left
        "east": 0    # Right
    }
    
    def __init__(self, *args, **kwargs):
        """Absolute Direction Wrapper Initialization"""
        super().__init__(*args, **kwargs)
    
    def _get_target_direction(self, absolute_action: int) -> int:
        """
        Convert absolute actions to MiniGrid direction
        
        Args:
            absolute_action: Absolute Action Index (0=up, 1=down, 2=left, 3=right)
        
        Returns:
            target_dir: MiniGrid Direction (0=East, 1=South, 2=West, 3=North)
        """
        direction_map = {
            0: 3,  # up -> North
            1: 1,  # down -> South
            2: 2,  # left -> West
            3: 0   # right -> East
        }
        return direction_map.get(absolute_action, 0)
    
    def _calculate_rotation(self, current_dir: int, target_dir: int) -> list:
        """
        Calculating the action sequence to rotate from the current direction to the target direction
        
        Args:
            current_dir: Current direction (0=East, 1=South, 2=West, 3=North)
            target_dir: Target Direction (0=East, 1=South, 2=West, 3=North)
        
        Returns:
            rotation_actions: Rotation Action List (0=turn left, 1=turn right)
        """
        if current_dir == target_dir:
            return []  # Already heading in the right direction
        
        # Calculating Directional Difference
        diff = (target_dir - current_dir) % 4
        
        if diff == 1:
            # Clockwise 90 degrees (rotate once to the right)
            return [1]  # turn right
        elif diff == 2:
            # 180-degree rotation (two turns to the right or two turns to the left)
            return [1, 1]  # turn right twice (The shorter route)
        elif diff == 3:
            # 90 degrees counterclockwise (one turn to the left)
            return [0]  # turn left
        
        return []
    
    def step_absolute(self, action: Union[int, str]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute absolute directional action
        
        Args:
            action: Absolute Direction Action (integer index or action name string)
                - 0 or "move up": Move up (North)
                - 1 or "move down": Move Down (South)
                - 2 or "move left": Move left (West)
                - 3 or "move right": Move right (East)
                - 4 or "pickup": Object Pickup
                - 5 or "drop": Object Placement
                - 6 or "toggle": Interaction
        
        Returns:
            observation: New Observations (Dictionary)
            reward: Compensation (float)
            terminated: Whether the goal was achieved (bool)
            truncated: Whether time has been exceeded (bool)
            info: Additional Information (Dictionary)
        """
        # If the action is a string, convert it to an integer.
        if isinstance(action, str):
            action = self.parse_absolute_action(action)
        
        # When it is not a movement action (pickup, drop, toggle), execute directly.
        if action >= 4:
            # Convert to existing MiniGrid actions (4=pickup, 5=drop, 6=toggle)
            return super().step(action)
        
        # For movement actions: Confirm the current direction and perform the necessary rotation.
        current_dir = self.env.agent_dir
        target_dir = self._get_target_direction(action)
        
        # Calculation of rotational action
        rotation_actions = self._calculate_rotation(current_dir, target_dir)
        
        # Rotation Execution
        for rot_action in rotation_actions:
            obs, reward, terminated, truncated, info = super().step(rot_action)
            if terminated or truncated:
                return obs, reward, terminated, truncated, info
        
        # After completing rotation toward the target direction, proceed forward.
        obs, reward, terminated, truncated, info = super().step(2)  # move forward
        
        return obs, reward, terminated, truncated, info
    
    def parse_absolute_action(self, action_str: str) -> int:
        """
        Convert absolute direction action strings to indices
        
        Args:
            action_str: Action text (e.g., “move up”, ‘left’, “north”, etc.)
        
        Returns:
            action: Action Index (0-6)
        
        Raises:
            ValueError: In the case of an unknown action
        """
        # Remove spaces
        action_str = action_str.strip()
        
        # If it is a numeric string, convert it directly.
        try:
            action_int = int(action_str)
            if 0 <= action_int <= 6:
                return action_int
        except ValueError:
            pass
        
        # Convert to lowercase
        action_str_lower = action_str.lower()
        
        # Search by Action Alias
        if action_str_lower in self.ABSOLUTE_ACTION_ALIASES:
            return self.ABSOLUTE_ACTION_ALIASES[action_str_lower]
        
        # If not found, an error occurs.
        raise ValueError(
            f"Unknown absolute action: '{action_str}'. "
            f"Available actions: {list(self.ABSOLUTE_ACTION_ALIASES.keys())} or numbers 0-6"
        )
    
    def get_absolute_action_space(self) -> Dict:
        """
        Return absolute directional action spatial information
        
        Returns:
            action_space_info: Action Spatial Information Dictionary
        """
        return {
            'n': 7,
            'actions': list(self.ABSOLUTE_ACTION_NAMES.values()),
            'action_mapping': self.ABSOLUTE_ACTION_NAMES,
            'action_aliases': self.ABSOLUTE_ACTION_ALIASES
        }


def get_system_prompt() -> str:
    """System Prompt Creation (Absolute Coordinate Version)"""
    return """You are a robot operating on a grid map.

## Environment
Grid world with walls (black), blue pillar (impassable), purple table (impassable), robot (red arrow shows heading), and goal (green marker if present).

## Coordinate System
The top of the image is North (up), and the bottom is South (down).
The left is West (left), and the right is East (right).

## Robot Orientation
In the image, the red triangle represents the robot.
The robot's heading direction is shown by the triangle's apex (sharp tip).
However, you can move in ANY direction regardless of the robot's current heading.

## Action Space (Absolute Directions)
You can move directly in absolute directions:
- "move up" or "up" or "north" or "n": Move one cell North (upward)
- "move down" or "down" or "south" or "s": Move one cell South (downward)
- "move left" or "left" or "west" or "w": Move one cell West (leftward)
- "move right" or "right" or "east" or "e": Move one cell East (rightward)
- "pickup": Pick up object at current location
- "drop": Drop carried object
- "toggle": Interact with objects (e.g., open doors)

## Movement Rules
**CRITICAL**: All movements are in ABSOLUTE directions (North/South/East/West).
- "up" = move North (upward on the image)
- "down" = move South (downward on the image)
- "left" = move West (leftward on the image)
- "right" = move East (rightward on the image)
- The robot will automatically rotate to face the correct direction before moving
- You don't need to think about the robot's current heading - just specify the direction you want to go

## Response Format
Respond in JSON format:
```json
{
    "action": "<action_name_or_number>",
    "environment_info": "<description of current state with spatial relationships in absolute coordinates (North/South/East/West)>",
    "reasoning": "<explanation of why you chose this action>"
}
```

**Important**: 
- Valid JSON format required
- Actions must be from the list above
- Complete mission from user prompt
- Use absolute directions (up/down/left/right), not relative to robot heading
- Think in terms of the image: up=North, down=South, left=West, right=East
"""





def normalize_line(line: str) -> str:
    # Remove emoji variation selector (U+FE0F)
    return line.replace("\ufe0f", "")

def load_txt_map(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    def normalize_line(line):
        return line.replace("\ufe0f", "")

    grid = [[c for c in normalize_line(line)] for line in lines]

    height = len(grid)
    width = len(grid[0])

    for i, row in enumerate(grid):
        if len(row) != width:
            raise ValueError(
                f"Inconsistent width at row {i}: {len(row)} vs {width}"
            )

    return grid, width, height

def build_room_from_txt_map(txt_path):
    grid, width, height = load_txt_map(txt_path)

    walls = []
    rooms = []
    start_pos = None
    goal_pos = None

    for y in range(height):
        for x in range(width):
            symbol = grid[y][x]

            if symbol == "⬛":
                walls.append((x, y))
            elif symbol == "🟦":
                walls.append((x, y, "blue"))
            elif symbol == "🟪":
                walls.append((x, y, "purple"))
            elif symbol == "🟥":
                walls.append((x, y, "red"))
            elif symbol == "🟩":
                walls.append((x, y, "green"))
            elif symbol == "🟨":
                walls.append((x, y, "yellow"))
            elif symbol == "🎯":
                goal_pos = (x, y)
            elif symbol == "🤖":
                start_pos = (x, y)
            elif symbol == "⬜️" or symbol == "⬜":
                pass
            else:
                raise ValueError(f"Unknown map symbol: {symbol}")

    if start_pos is None:
        raise ValueError("No agent start position (🤖) in map")
    if goal_pos is None:
        raise ValueError("No goal position (🎯) in map")

    room_config = {
        "start_pos": start_pos,
        "goal_pos": goal_pos,
        "walls": walls,
        "objects": []
    }

    return AbsoluteDirectionWrapper(size=width, room_config=room_config)
    #return CustomRoomWrapper(size=width, room_config=room_config)







#def create_scenario2_environment():
#    """Scenario 2 Environment Creation"""
#    size = 10
#    
#    # Outer Wall Creation
#    walls = []
#    for i in range(size):
#        walls.append((i, 0))
#        walls.append((i, size-1))
#        walls.append((0, i))
#        walls.append((size-1, i))
#    
#    # 파란 기둥: 2x2 Grid (색상이 있는 벽으로 변경)
#    blue_pillar_positions = [(3, 4), (4, 4), (3, 5), (4, 5)]
#    for pos in blue_pillar_positions:
#        walls.append((pos[0], pos[1], 'blue'))
#    
#    # 테이블: 보라색 1x3 Grid (색상이 있는 벽으로 변경)
#    table_positions = [(5, 1), (6, 1), (7, 1)]
#    for pos in table_positions:
#        walls.append((pos[0], pos[1], 'purple'))
#    
#    # 시작점과 종료점
#    start_pos = (1, 8)
#    goal_pos = (8, 1)
#    
#    room_config = {
#        'start_pos': start_pos,
#        'goal_pos': goal_pos,
#        'walls': walls,
#        'objects': []  # box 객체 제거
#    }
#    
#    return AbsoluteDirectionWrapper(size=size, room_config=room_config)


def visualize_grid_cli(wrapper: AbsoluteDirectionWrapper, state: dict):
    """Visualize the grid as text in the CLI"""
    env = wrapper.env
    size = wrapper.size
    
    # Agent Position and Orientation
    agent_pos = state['agent_pos']
    if isinstance(agent_pos, np.ndarray):
        agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
    else:
        agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
    
    agent_dir = state['agent_dir']
    direction_symbols = {0: '>', 1: 'v', 2: '<', 3: '^'}
    agent_symbol = direction_symbols.get(agent_dir, 'A')
    
    # Grid Generation
    grid_chars = []
    for y in range(size):
        row = []
        for x in range(size):
            cell = env.grid.get(x, y)
            
            if x == agent_x and y == agent_y:
                row.append(agent_symbol)
            elif cell is not None and cell.type == 'wall':
                # Colored wall markings
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
                        row.append('⬛')  # Base color (grey)
                else:
                    row.append('⬛')  # No color
            elif cell is not None and cell.type == 'goal':
                row.append('🎯')
            elif cell is not None:
                if hasattr(cell, 'color'):
                    if cell.color == 'blue':
                        row.append('🟦')
                    elif cell.color == 'purple':
                        row.append('🟪')
                    elif cell.color == 'red':
                        row.append('🟥')
                    elif cell.color == 'green':
                        row.append('🟩')
                    else:
                        row.append('🟨')
                else:
                    row.append('🟨')
            else:
                row.append('⬜️')
        grid_chars.append(row)
    
    # Grid output
    print("\n" + "=" * 60)
    print("Current Grid State:")
    print("=" * 60)
    for y in range(size):
        print(''.join(grid_chars[y]))
    print("=" * 60)
    print(f"Agent Position: ({agent_x}, {agent_y}), Direction: {agent_dir} ({agent_symbol})")
    print("=" * 60 + "\n")


def display_image(img, window_name="MiniGrid VLM Control (Absolute)", cell_size=32):
    """Displaying Images Using OpenCV"""
    if img is not None:
        try:
            img_bgr = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
            
            # Image resizing
            height, width = img_bgr.shape[:2]
            max_size = 800
            if height < max_size and width < max_size:
                scale = min(max_size // height, max_size // width, 4)
                if scale > 1:
                    new_width = width * scale
                    new_height = height * scale
                    img_bgr = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            
            cv2.imshow(window_name, img_bgr)
            cv2.waitKey(1)
        except Exception as e:
            print(f"Image display error: {e}")


def main():
    """Main function"""
    print("=" * 60)
    print("MiniGrid VLM Interaction (Absolute Coordinate Movement Version)")
    print("=" * 60)
    print("\nEnvironment Configuration:")
    print("  - Blue Column: 2x2 Grid")
    print("  - Table: Purple 1x3 Grid")
    print("  - Starting point: (1, 8)")
    print("  - End point: (8, 1)")
    print("\nMission: Go to the blue pillar, turn right, and stop next to the table.")
    print("\nAction Space: Direct movement possible up/down/left/right (absolute coordinates)")

    # Environment Creation
    print("\n[1] Creating environment...")
    MAP_PATH = "map_copy.txt"   # your emoji map
    wrapper = build_room_from_txt_map(MAP_PATH)
    wrapper.reset()
    
    state = wrapper.get_state()
    print(f"Agent Start Position: {state['agent_pos']}")
    print(f"Agent Direction: {state['agent_dir']}")
    
    # Action Spatial Information Output
    action_space = wrapper.env.action_space
    print("  - Available actions:")
    for idx, name in wrapper.ABSOLUTE_ACTION_NAMES.items():
        print(f"    {idx}: {name}")
    print(f"\nAbsolute Direction Action Space:")
    print("  - Available Absolute Actions:")
    for idx, name in wrapper.ABSOLUTE_ACTION_NAMES.items():
        print(f"    {idx}: {name}")


    
    # VLM Initialization
    print("\n[2] Initializing VLM...")
    try:
        vlm = ChatGPT4oVLMWrapper(
            model=VLM_MODEL,
            temperature=VLM_TEMPERATURE,
            max_tokens=VLM_MAX_TOKENS
        )
        print(f"VLM initialization complete: {VLM_MODEL}")
    except Exception as e:
        print(f"VLM initialization failed: {e}")
        return
    
    # PostProcessor Initialization
    postprocessor = VLMResponsePostProcessor(required_fields=["action", "environment_info"])
    
    # System Prompt
    SYSTEM_PROMPT = get_system_prompt()
    
    # Main Loop
    step = 0
    done = False
    WINDOW_NAME = "MiniGrid VLM Control (Absolute)"
    
    print("\n" + "=" * 60)
    print("Experiment Start")
    print("=" * 60)
    
    while not done:
        step += 1
        print("\n" + "=" * 80)
        print(f"STEP {step}")
        print("=" * 80)
        
        # Current Status
        image = wrapper.get_image()
        state = wrapper.get_state()
        print(f"Location: {state['agent_pos']}, Direction: {state['agent_dir']}")
        
        # CLI Visualization
        visualize_grid_cli(wrapper, state)
        
        # GUI visualization
        display_image(image, WINDOW_NAME)
        
        # User prompt input
        print("Enter a command (Enter: default prompt):")
        user_prompt = input("> ").strip()
        if not user_prompt:
            user_prompt = "Based on the current image, choose the next action to complete the mission: Go to the blue pillar, turn right, then stop next to the table. Use absolute directions (up/down/left/right)."
        
        # VLM call
        print("\n[3] Sending request to VLM...")
        try:
            vlm_response_raw = vlm.generate(
                image=image,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
            print(f"VLM response received")
        except Exception as e:
            print(f"VLM API call failure: {e}")
            break
        
        # Response Parsing
        print("[4] Parsing the response...")
        try:
            vlm_response = postprocessor.process(vlm_response_raw, strict=True)
            action_str = vlm_response.get('action', 'up')
            print(f"Parsed action: {action_str}")
            print(f"Environment Info: {vlm_response.get('environment_info', 'N/A')}")
            print(f"Reasoning: {vlm_response.get('reasoning', 'N/A')}")
        except ValueError as e:
            print(f"Response parsing failed: {e}")
            print(f"Original response: {vlm_response_raw[:200]}...")
            action_str = 'up'  # Default value: move up
        
        # Action Execution
        print(f"\n[5] Action in progress...")
        try:
            action_index = wrapper.parse_absolute_action(action_str)
            action_name = wrapper.ABSOLUTE_ACTION_NAMES[action_index]
            print(f"Action to execute: {action_name} (Index: {action_index})")
            
            _, reward, terminated, truncated, _ = wrapper.step_absolute(action_index)
            done = terminated or truncated
            
            print(f"Reward: {reward}, Done: {done}")
        except Exception as e:
            print(f"Action execution failed: {e}")
            import traceback
            traceback.print_exc()
            # Use basic actions
            try:
                _, reward, terminated, truncated, _ = wrapper.step_absolute(0)  # move up
                done = terminated or truncated
            except:
                break
        
        # Updated status indicator
        new_state = wrapper.get_state()
        visualize_grid_cli(wrapper, new_state)
        updated_image = wrapper.get_image()
        display_image(updated_image, WINDOW_NAME)
        
        # Confirm Termination
        if done:
            print("\n" + "=" * 80)
            print("Goal scored! Game over.")
            print("=" * 80)
            break
        
        # Maximum step limit
        if step >= 100:
            print("\nThe maximum number of steps (100) has been reached.")
            break
    
    # Resource Cleanup
    cv2.destroyAllWindows()
    wrapper.close()
    print("\nExperiment completed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTerminated by the user.")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()