"""
MiniGrid VLM Interaction Script (Absolute Movement Version - Emoji Environment)

Controls and visualizes MiniGrid environments using VLM.
Provides absolute coordinate-based action space where robots can move directly up/down/left/right.

Environment configuration:
- 🧱(brick) emoji: 2x2 Grid, blue, can step on
- 🖥️📱(desktop/workstation) emoji: 1x2 Grid, purple, can step on

Usage:
    python minigrid_vlm_interact_absolute_emoji.py
"""

from minigrid import register_minigrid_envs
# Actual path: lib.map_manager.minigrid_customenv_emoji
from lib import MiniGridEmojiWrapper, load_emoji_map_from_json
# Actual paths: lib.vlm.vlm_wrapper, lib.vlm.vlm_postprocessor
from lib import ChatGPT4oVLMWrapper, VLMResponsePostProcessor
import numpy as np
import cv2
from typing import Union, Tuple, Dict, Optional

# Register MiniGrid environments
register_minigrid_envs()

# VLM configuration
VLM_MODEL = "gpt-4o"
VLM_TEMPERATURE = 0.0
VLM_MAX_TOKENS = 1000


class AbsoluteDirectionEmojiWrapper(MiniGridEmojiWrapper):
    """
    Emoji Wrapper supporting absolute direction (up/down/left/right) movement
    
    Extends MiniGridEmojiWrapper to provide action space for direct up/down/left/right movement.
    Can move based on absolute coordinate system regardless of robot's current direction.
    
    This class is simply a MiniGridEmojiWrapper with use_absolute_movement=True.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize absolute direction wrapper"""
        # Force use_absolute_movement to True
        kwargs['use_absolute_movement'] = True
        super().__init__(*args, **kwargs)


def get_system_prompt() -> str:
    """Generate System Prompt (absolute coordinate version - emoji environment)"""
    return """You are a robot operating on a grid map.

## Environment
Grid world with walls which must step on and detour (black, brick emoji 🧱)
robot is represented as emoji 🤖, which is your avatar.
you can move in ANY direction regardless of the robot's current heading.

## Action Space, Coordinate System
**CRITICAL**: All movements are in ABSOLUTE directions (UP/DOWN/LEFT/RIGHT).
- "up": Move UP (upward on the image)
- "down": Move DOWN (downward on the image)
- "left": Move LEFT (leftward on the image)
- "right" :Move RIGHT (rightward on the image)
- "pickup": Pick up object
- "drop": Drop object
- "toggle": Interact with objects

## Movement Rules
- You cannot step on some objects, you should detour around it.
- When you step on an emoji object, the block will glow green

## Response Format
Respond in JSON format:
```json
{
    "environment_info": "<description of current state with spatial relationships in relative to robot(🤖)'s position (UP/DOWN/LEFT/RIGHT)>",
    "action": "<action_name>",
    "reasoning": "<explanation of why you chose this action>"
}
```

**Important**: 
- Valid JSON format required
- Actions must be from the list above
- Complete mission from user prompt
- Use absolute directions (up/down/left/right)
"""


def create_scenario2_environment():
    """Create Scenario 2 environment (emoji version)"""
    size = 10
    
    # Create outer walls
    walls = []
    for i in range(size):
        walls.append((i, 0))
        walls.append((i, size-1))
        walls.append((0, i))
        walls.append((size-1, i))
    
    # Blue pillar: 2x2 Grid -> 🧱(brick) emoji, can step on
    blue_pillar_positions = [(3, 4), (4, 4), (3, 5), (4, 5)]
    
    # Table: Purple 1x3 Grid -> 🖥️📱 (modified to 1x2), can step on
    # Modified to 1x2: (5, 1), (6, 1) -> desktop and workstation
    table_positions = [(5, 1), (6, 1)]
    
    # Start and goal positions
    start_pos = (1, 8)
    goal_pos = (8, 1)
    
    # Create emoji objects
    objects = []
    
    # 🧱(brick) emoji: blue, can step on
    for pos in blue_pillar_positions:
        objects.append({
            'type': 'emoji',
            'pos': pos,
            'emoji_name': 'brick',
            'color': 'blue',
            'can_pickup': False,
            'can_overlap': True,  # Can step on
            'use_emoji_color': True  # Use specified color (blue)
        })
    
    # 🖥️📱(desktop/workstation) emoji: purple, can step on
    objects.append({
        'type': 'emoji',
        'pos': (5, 1),
        'emoji_name': 'desktop',
        'color': 'purple',
        'can_pickup': False,
        'can_overlap': True,  # Can step on
        'use_emoji_color': True  # Use specified color (purple)
    })
    
    objects.append({
        'type': 'emoji',
        'pos': (6, 1),
        'emoji_name': 'workstation',
        'color': 'purple',
        'can_pickup': False,
        'can_overlap': True,  # Can step on
        'use_emoji_color': False  # Use specified color (purple)
    })
    
    room_config = {
        'start_pos': start_pos,
        'goal_pos': goal_pos,
        'walls': walls,
        'objects': objects,
        
        ## Robot marker settings
        'use_robot_emoji': True,  # Display robot as 🤖 emoji
        'robot_emoji_color': 'red',  # Robot emoji color (only used when use_robot_emoji_color=False)
        'use_robot_emoji_color': True  # Use original emoji color (True: original color, False: use robot_emoji_color)
    }
    
    return AbsoluteDirectionEmojiWrapper(size=size, room_config=room_config)


def visualize_grid_cli(wrapper: AbsoluteDirectionEmojiWrapper, state: dict):
    """Visualize grid as text in CLI"""
    env = wrapper.env
    size = wrapper.size
    
    # Agent position and direction
    agent_pos = state['agent_pos']
    if isinstance(agent_pos, np.ndarray):
        agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
    else:
        agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
    
    agent_dir = state['agent_dir']
    direction_symbols = {0: '>', 1: 'v', 2: '<', 3: '^'}
    agent_symbol = direction_symbols.get(agent_dir, 'A')
    
    # Create grid
    grid_chars = []
    for y in range(size):
        row = []
        for x in range(size):
            cell = env.grid.get(x, y)
            
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
            elif cell is not None and cell.type == 'emoji':
                # Display emoji object
                if hasattr(cell, 'emoji_name'):
                    emoji_map = {
                        'brick': '🧱',
                        'desktop': '🖥️',
                        'workstation': '📱',
                        'tree': '🌲',
                        'mushroom': '🍄',
                        'flower': '🌼',
                        'cat': '🐈',
                        'grass': '🌾',
                        'rock': '🗿',
                        'box': '📦',
                        'chair': '🪑',
                        'apple': '🍎'
                    }
                    emoji_char = emoji_map.get(cell.emoji_name, '❓')
                    # Special mark if robot is on top for green border display
                    if hasattr(cell, 'agent_on_top') and cell.agent_on_top:
                        row.append(f'[{emoji_char}]')  # Border mark
                    else:
                        row.append(emoji_char)
                else:
                    row.append('❓')
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
    
    # Print grid
    print("\n" + "=" * 60)
    print("Current Grid State:")
    print("=" * 60)
    for y in range(size):
        print(''.join(grid_chars[y]))
    print("=" * 60)
    print(f"Agent Position: ({agent_x}, {agent_y}), Direction: {agent_dir} ({agent_symbol})")
    print("=" * 60 + "\n")


def display_image(img, window_name="MiniGrid VLM Control (Absolute Emoji)", cell_size=32):
    """Display image using OpenCV"""
    if img is not None:
        try:
            img_bgr = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
            
            # Resize image
            height, width = img_bgr.shape[:2]
            max_size = 1200  # Increased from 800 to 1200 for larger window
            if height < max_size and width < max_size:
                scale = min(max_size // height, max_size // width, 6)  # Increased max scale from 4 to 6
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
    import sys
    from pathlib import Path
    
    # Get script directory for relative path resolution
    script_dir = Path(__file__).parent.resolve()
    
    # JSON map file path via command line argument
    json_map_path = None
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage:")
            print("  python VLM_interact_minigrid-absolute_emoji.py [json_map_path]")
            print("  Example: python VLM_interact_minigrid-absolute_emoji.py ./config/example_map.json")
            print("  Example: python VLM_interact_minigrid-absolute_emoji.py ../config/example_map.json")
            print("\nIf no JSON path is provided, uses hardcoded scenario2 environment.")
            return
        else:
            # Resolve path
            user_path = Path(sys.argv[1])
            if user_path.is_absolute():
                json_map_path = str(user_path)
            else:
                # Try relative to script directory first, then relative to current working directory
                script_relative = script_dir / user_path
                cwd_relative = Path.cwd() / user_path
                if script_relative.exists():
                    json_map_path = str(script_relative.resolve())
                elif cwd_relative.exists():
                    json_map_path = str(cwd_relative.resolve())
                else:
                    json_map_path = str(user_path)
    
    print("=" * 60)
    print("MiniGrid VLM Interaction (Absolute Movement Version - Emoji Environment)")
    print("=" * 60)
    
    # Create environment
    print("\n[1] Creating environment...")
    if json_map_path:
        print(f"  Loading map from: {json_map_path}")
        wrapper = load_emoji_map_from_json(json_map_path)
    else:
        print("  Using hardcoded scenario2 environment")
        print("  - 🧱(brick) emoji: 2x2 Grid, blue, can step on")
        print("  - 🖥️📱(desktop/workstation) emoji: 1x2 Grid, purple, can step on")
        print("  - Start position: (1, 8)")
        print("  - Goal position: (8, 1)")
        print("\nMission: Go to the blue pillar(🧱), turn right, then stop next to the table(🖥️📱)")
        wrapper = create_scenario2_environment()
    
    print("\nAction space: Direct movement up/down/left/right (absolute coordinates)")
    wrapper.reset()
    
    state = wrapper.get_state()
    print(f"Agent start position: {state['agent_pos']}")
    print(f"Agent direction: {state['agent_dir']}")
    
    # Print action space information
    action_space = wrapper.get_absolute_action_space()
    print(f"\nAbsolute direction action space:")
    print(f"  - Available actions: {action_space['actions']}")
    
    # Initialize VLM
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
    
    # Initialize PostProcessor
    postprocessor = VLMResponsePostProcessor(required_fields=["action", "environment_info"])
    
    # System Prompt
    SYSTEM_PROMPT = get_system_prompt()
    
    # Main loop
    step = 0
    done = False
    WINDOW_NAME = "MiniGrid VLM Control (Absolute Emoji)"
    
    print("\n" + "=" * 60)
    print("Experiment started")
    print("=" * 60)
    
    while not done:
        step += 1
        print("\n" + "=" * 80)
        print(f"STEP {step}")
        print("=" * 80)
        
        # Current state
        image = wrapper.get_image()
        state = wrapper.get_state()
        print(f"Position: {state['agent_pos']}, Direction: {state['agent_dir']}")
        
        # CLI visualization
        visualize_grid_cli(wrapper, state)
        
        # GUI visualization
        display_image(image, WINDOW_NAME)
        
        # Get user prompt input
        print("Enter command (Enter: default prompt):")
        user_prompt = input("> ").strip()
        if not user_prompt:
            user_prompt = "Based on the current image, choose the next action to complete the mission: Go to the blue brick emoji 🧱, turn right, then stop next to the desktop/workstation emoji 🖥️📱. Use absolute directions (up/down/left/right)."
        
        # Call VLM
        print("\n[3] Sending request to VLM...")
        try:
            vlm_response_raw = vlm.generate(
                image=image,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
            print(f"VLM response received")
        except Exception as e:
            print(f"VLM API call failed: {e}")
            break
        
        # Parse response
        print("[4] Parsing response...")
        try:
            vlm_response = postprocessor.process(vlm_response_raw, strict=True)
            action_str = vlm_response.get('action', 'up')
            print(f"Parsed action: {action_str}")
            print(f"Environment Info: {vlm_response.get('environment_info', 'N/A')}")
            print(f"Reasoning: {vlm_response.get('reasoning', 'N/A')}")
        except ValueError as e:
            print(f"Response parsing failed: {e}")
            print(f"Original response: {vlm_response_raw[:200]}...")
            action_str = 'up'  # Default: move up
        
        # Execute action
        print(f"\n[5] Executing action...")
        try:
            action_index = wrapper.parse_absolute_action(action_str)
            action_space = wrapper.get_absolute_action_space()
            action_name = action_space['action_mapping'].get(action_index, f"action_{action_index}")
            print(f"Action to execute: {action_name} (index: {action_index})")
            
            # step() handles absolute movement since use_absolute_movement=True
            _, reward, terminated, truncated, _ = wrapper.step(action_index)
            done = terminated or truncated
            
            print(f"Reward: {reward}, Done: {done}")
        except Exception as e:
            print(f"Action execution failed: {e}")
            import traceback
            traceback.print_exc()
            # Use default action
            try:
                _, reward, terminated, truncated, _ = wrapper.step(0)  # move up
                done = terminated or truncated
            except:
                break
        
        # Display updated state
        new_state = wrapper.get_state()
        visualize_grid_cli(wrapper, new_state)
        updated_image = wrapper.get_image()
        display_image(updated_image, WINDOW_NAME)
        
        # Check termination
        if done:
            print("\n" + "=" * 80)
            print("Goal reached! Terminating")
            print("=" * 80)
            break
        
        # Maximum step limit
        if step >= 100:
            print("\nMaximum step count (100) reached.")
            break
    
    # Clean up resources
    cv2.destroyAllWindows()
    wrapper.close()
    print("\nExperiment completed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
