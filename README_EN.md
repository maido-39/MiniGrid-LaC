# MiniGrid-LaC

A project for Language-conditioned Reinforcement Learning in MiniGrid environments.

## Overview

This project implements reinforcement learning agents that utilize language instructions in MiniGrid environments.

## Project Structure

```
multigrid-LaC/
├── src/                          # Source code directory
│   ├── lib/                      # Core library modules
│   │   ├── map_manager/          # Map and environment management
│   │   │   ├── minigrid_customenv_emoji.py    # Main environment wrapper (emoji support, absolute movement)
│   │   │   └── emoji_map_loader.py            # JSON map loader for emoji-based maps
│   │   └── vlm/                  # Vision Language Model modules
│   │       ├── vlm_wrapper.py                 # VLM API wrapper (OpenAI GPT-4o)
│   │       ├── vlm_postprocessor.py          # VLM response parser and validator
│   │       ├── vlm_controller.py             # Generic VLM controller for environment control
│   │       ├── vlm_manager.py                 # VLM handler manager (multi-provider support)
│   │       └── handlers/                      # VLM provider handlers (OpenAI, Qwen, Gemma, etc.)
│   ├── legacy/                   # Legacy code (maintained for backward compatibility)
│   │   ├── relative_movement/    # Relative movement-based control (deprecated)
│   │   │   └── custom_environment.py          # Legacy environment wrapper
│   │   └── vlm_rels/             # Legacy VLM-related modules
│   │       ├── minigrid_vlm_controller.py     # Legacy MiniGrid-specific VLM controller
│   │       └── minigrid_vlm_helpers.py        # Legacy visualization helpers
│   ├── dev-*/                    # Development branches (experimental features)
│   │   ├── dev-scenario_2/       # Scenario 2 development
│   │   └── dev-action_uncertainty/ # Action uncertainty estimation experiments
│   ├── test_script/              # Test and example scripts
│   │   ├── emoji_test/           # Emoji rendering tests
│   │   ├── keyboard_control/    # Keyboard control examples
│   │   ├── etc/                  # Miscellaneous test scripts
│   │   └── similarity_calculator/ # Text similarity utilities
│   ├── asset/                    # Resource files
│   │   ├── arrow.png             # Robot arrow marker image
│   │   └── fonts/                # Font files for emoji rendering
│   ├── config/                   # Configuration files (moved from root)
│   │   └── example_map.json      # Example emoji map configuration
│   ├── scenario2_test_absolutemove.py  # Main experiment script (absolute movement)
│   └── VLM_interact_minigrid-(absolute,emoji).py  # VLM interaction example
├── config/                        # Configuration files
│   └── example_map.json          # Example emoji map (JSON format)
├── logs/                         # Experiment logs (generated at runtime)
├── docs/                          # Documentation
└── requirements.txt              # Python dependencies
```

### Directory Purposes

- **`src/lib/`**: Core reusable library modules
  - **`map_manager/`**: Environment creation and map loading utilities
  - **`vlm/`**: VLM integration modules for robot control

- **`src/legacy/`**: Legacy code maintained for backward compatibility
  - **`relative_movement/`**: Old relative movement-based control (use `lib.map_manager` instead)
  - **`vlm_rels/`**: Legacy VLM modules (use `lib.vlm` instead)

- **`src/dev-*/`**: Experimental development branches
  - Active development features that may be merged into main library later

- **`src/test_script/`**: Test and example scripts
  - Various test scripts, examples, and utility scripts

- **`src/asset/`**: Static resource files
  - Images, fonts, and other assets used by the environment

- **`config/`**: Configuration files
  - JSON map files and other configuration data

### Import Usage

All modules can be imported using simplified paths thanks to `__init__.py`:

```python
# Simplified imports (recommended)
from lib import MiniGridEmojiWrapper, load_emoji_map_from_json
from lib import ChatGPT4oVLMWrapper, VLMResponsePostProcessor, VLMController
from legacy import CustomRoomWrapper, MiniGridVLMController

# Full paths are also available if needed
# from lib.map_manager.minigrid_customenv_emoji import MiniGridEmojiWrapper
# from lib.vlm.vlm_wrapper import ChatGPT4oVLMWrapper
```

## Documentation

Detailed documentation is available in the [`docs/`](docs/) folder:

### MiniGrid Basics
- [MiniGrid Environment List](docs/minigrid-environments.md) - All built-in MiniGrid environments
- [MiniGrid Objects and Attributes](docs/minigrid-objects.md) - Available object types and attributes in MiniGrid
- [Environment Creation Guide](docs/environment-creation.md) - How to create MiniGrid environments
- [Best Practices](docs/best-practices.md) - Recommended practices for MiniGrid environment creation

### API Documentation
- [Custom Environment API](docs/custom-environment-api.md) - CustomRoomEnv API documentation
- [Wrapper API](docs/wrapper-api.md) - CustomRoomWrapper API documentation (includes absolute movement)
- [Wrapper Methods Guide](docs/wrapper-methods.md) - All CustomRoomWrapper method descriptions

### Usage Guides
- [Keyboard Control Guide](docs/keyboard-control.md) - Keyboard control example explanation
- [VLM Test Script Guide](docs/test-vlm-guide.md) - VLM model testing and comparison guide
- [Emoji Map JSON Loader Guide](docs/emoji-map-loader.md) - Loading emoji maps from JSON files
- [SLAM-style FOV Mapping Guide](docs/slam-fov-mapping.md) - Exploration area tracking and field of view limitation
- [Emoji Usage Guide](docs/EMOJI_USAGE_GUIDE.md) - Using emoji objects

## Features

- MiniGrid environment integration
- Language-conditioned policy learning
- Reinforcement learning algorithm implementation

## Installation

### Requirements

- Python 3.8 or higher
- OpenAI API key (for VLM features)

### Installation with Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/maido-39/MiniGrid-LaC.git
cd MiniGrid-LaC

# Create Conda environment (Python 3.10 recommended)
conda create -n minigrid python=3.10 -y
conda activate minigrid

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key (create .env file)
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### Installation with pip

```bash
# Clone repository
git clone https://github.com/maido-39/MiniGrid-LaC.git
cd MiniGrid-LaC

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key (create .env file)
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### Verify Installation

```bash
# Check Python version
python --version  # Should be Python 3.8 or higher

# Verify package installation
python -c "import minigrid; import gymnasium; import openai; import cv2; print('All packages installed successfully!')"
```

## Usage

### Before Running Scripts

All scripts should be run from the `src/` directory, or set `PYTHONPATH` from the project root:

```bash
# Method 1: Run from src/ directory (recommended)
cd src
python scenario2_test_absolutemove.py

# Method 2: Set PYTHONPATH from project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python src/scenario2_test_absolutemove.py

# Method 3: Set sys.path in Python code
# Add the following code inside the script:
# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
```

### Example Scripts

This project provides several example scripts for language-based control using VLM (Vision Language Model) in MiniGrid environments.

#### 1. `test_script/keyboard_control/keyboard_control.py` - Keyboard Control Example

**Description**: A simple example script for directly controlling MiniGrid environments with keyboard input. Suitable for understanding and testing basic environment behavior.

**Features**:
- Agent control via keyboard input
- Real-time environment visualization with OpenCV
- Environment reset and exit functionality

**How to Run**:
```bash
cd src
python test_script/keyboard_control/keyboard_control.py
```

**Controls**:
- `w`: Move forward
- `a`: Turn left
- `d`: Turn right
- `s`: Move backward (only supported in some environments)
- `r`: Reset environment
- `q`: Quit

**Environment**: `MiniGrid-Empty-8x8-v0` (default empty environment)

---

#### 2. `VLM_interact_minigrid-(absolute,emoji).py` - VLM Interaction Example

**Description**: An example of controlling MiniGrid environments using VLM. Supports absolute coordinate movement and emoji maps.

**Features**:
- Automatic agent control via VLM
- Absolute coordinate movement (direct up/down/left/right movement)
- Emoji map support
- CLI and OpenCV visualization

**How to Run**:
```bash
# OpenAI API key required
export OPENAI_API_KEY=your-api-key

cd src
python VLM_interact_minigrid-\(absolute,emoji\).py
```

**Configuration**:
- VLM Model: `gpt-4o` (can be changed at top of code)
- Temperature: `0.0`
- Max Tokens: `1000`

**Environment**: Scenario 2 environment (blue pillar 2x2, purple table 1x3)

**Mission**: "Go to the blue pillar, turn right, then stop next to the table."

---

#### 3. `legacy/relative_movement/scenario2_test.py` - Scenario 2 Experiment (Full Features, Legacy)

**Note**: This script is legacy code. New projects should use `scenario2_test_absolutemove.py`.

**Description**: A complete control system using VLM in Scenario 2 experiment environment. A complete experiment script with all features including logging, permanent memory, grounding knowledge, and predicted path visualization.

**Features**:
- Automatic agent control via VLM
- Scenario 2 environment (blue pillar, purple table)
- **Permanent Memory System**: Summarizes previous actions and tracks progress
- **Grounding Knowledge System**: Learns from mistakes through user feedback and accumulates knowledge
- **Predicted Path Visualization**: Displays VLM-predicted action trajectories in CLI and OpenCV
- **Comprehensive Logging**: Saves images, JSON, CSV, and VLM I/O logs
- CLI and OpenCV visualization

**How to Run**:
```bash
# OpenAI API key required
export OPENAI_API_KEY=your-api-key

cd src
python legacy/relative_movement/scenario2_test.py
```

**Configuration** (can be changed at top of code):
```python
VLM_MODEL = "gpt-4o"  # Model name to use
VLM_TEMPERATURE = 0.0  # Generation temperature
VLM_MAX_TOKENS = 1000  # Maximum token count
ACTION_PREDICTION_COUNT = 5  # Number of actions for VLM to predict
```

**Environment**: Scenario 2 environment
- Size: 10x10
- Blue pillar: 2x2 Grid (impassable)
- Purple table: 1x3 Grid (impassable)
- Start position: (1, 8)
- Goal position: (8, 1)

**Mission**: "Go to the blue pillar, turn right, then stop next to the table."

**Log Output**:
- Saved in `logs/scenario2_YYYYMMDD_HHMMSS/` directory
  - `step_XXXX.png`: Environment image for each step
  - `experiment_log.json`: JSON log for all steps (cumulative)
  - `vlm_io_log.txt`: VLM input/output log (cumulative)
  - `experiment_log.csv`: Experiment data CSV (cumulative)
  - `system_prompt.txt`: Complete System Prompt content
  - `permanent_memory.txt`: Permanent memory and Grounding knowledge

**Features**:
- **Permanent Memory**: VLM summarizes previous actions and updates current progress at each step
- **Grounding Knowledge**: When user feedback is detected, VLM analyzes mistakes and records lessons (cumulative)
- **Predicted Trajectory**: VLM predicts multiple actions sequentially, executes only the first, and visualizes the rest
- **Feedback Detection**: Automatically detects natural language feedback from user prompts and updates Grounding

---

#### 4. `scenario2_test_absolutemove.py` - Scenario 2 Experiment (Absolute Movement Version) ⭐ **Recommended**

**Description**: VLM control system using absolute coordinate movement in Scenario 2 experiment environment. Loads maps from JSON files, enabling more intuitive control through absolute coordinate movement.

**Features**:
- Load emoji maps from JSON files (uses `lib.map_manager.emoji_map_loader`)
- Absolute coordinate movement (direct up/down/left/right movement)
- Automatic agent control via VLM
- Permanent memory system and grounding knowledge system
- Comprehensive logging (images, JSON, CSV, VLM I/O logs)

**How to Run**:
```bash
# OpenAI API key required
export OPENAI_API_KEY=your-api-key

cd src
# Use default map file (config/example_map.json)
python scenario2_test_absolutemove.py

# Specify specific JSON map file
python scenario2_test_absolutemove.py ../config/example_map.json
```

**Configuration** (can be changed at top of code):
```python
VLM_MODEL = "gpt-4o"
VLM_TEMPERATURE = 0.0
VLM_MAX_TOKENS = 1000
```

**Map File Format**: JSON file (see `example_map.json`)
- Define map layout with emojis
- Define type and attributes for each emoji
- Specify start and goal positions

**Features**:
- **Absolute Coordinate Movement**: Direct up/down/left/right movement regardless of robot direction
- **JSON Map Loading**: Create various maps by changing only JSON files without code modification
- **Emoji Support**: Visual map representation using emoji objects

**Detailed Guide**: [Emoji Map Loader Guide](docs/emoji-map-loader.md)

---

#### 5. `dev-scenario_2/scenario2_keyboard_control.py` - Scenario 2 Keyboard Control (Absolute Movement)

**Description**: Script for directly controlling Scenario 2 environment with keyboard. Enables more intuitive control using absolute coordinate movement.

**Features**:
- Load emoji maps from JSON files
- Absolute coordinate movement (w/a/s/d keys for up/down/left/right movement)
- OpenCV visualization
- Real-time status display

**How to Run**:
```bash
cd src
# Use default map file
python dev-scenario_2/scenario2_keyboard_control.py

# Specify specific JSON map file
python dev-scenario_2/scenario2_keyboard_control.py ../../config/example_map.json
```

**Controls**:
- `w`: Move up (North)
- `s`: Move down (South)
- `a`: Move left (West)
- `d`: Move right (East)
- `p`: pickup
- `x`: drop
- `t`: toggle
- `r`: Reset environment
- `q`: Quit

**Features**:
- Intuitive control with absolute coordinate movement
- Easy map changes with JSON map files
- Emoji object support

---

### Additional Scripts

#### `test_script/keyboard_control/keyboard_control_fov.py` - Field of View Limitation

A version of the keyboard control example with Field of View (FOV) limitation functionality. You can select MiniGrid built-in environments.

**How to Run**:
```bash
cd src
python test_script/keyboard_control/keyboard_control_fov.py
```

**Available Environments**:
1. FourRooms (4-room structure)
2. MultiRoom-N6 (6 rooms)
3. DoorKey-16x16 (door and key)
4. KeyCorridorS6R3 (corridor and key)
5. Playground
6. Empty-16x16 (empty environment)

**Additional Controls**:
- `f`: Toggle field of view limitation (on/off)
- `+`: Increase field of view range
- `-`: Decrease field of view range

---

#### `test_script/keyboard_control/keyboard_control_fov_mapping.py` - SLAM-style FOV Mapping

A version of the keyboard control example with SLAM (Simultaneous Localization and Mapping) style field of view limitation functionality.

**How to Run**:
```bash
cd src
python test_script/keyboard_control/keyboard_control_fov_mapping.py
```

**Key Features**:
- Track explored areas
- Current field of view range: displayed brightly
- Previously explored areas (outside field of view): displayed darkly (semi-transparent)
- Important objects (key, door, goal) locations: remain bright even if explored
- Unexplored areas: displayed in black

**Controls**: Same as `keyboard_control_fov.py`

**Detailed Guide**: [SLAM-style FOV Mapping Guide](docs/slam-fov-mapping.md)

---

#### 6. `test_script/etc/test_vlm.py` - VLM Model Testing and Comparison

**Description**: A script for testing and comparing various VLM (Vision Language Model) models. You can easily change images, prompts, and models for testing.

**Features**:
- Support for various VLM models (OpenAI, Qwen, Gemma)
- Flexible image input (URL, local file, auto-generated)
- Command-line interface for specifying images and prompts
- Multi-model simultaneous testing and result comparison

**How to Run**:
```bash
# Activate minigrid conda environment (required)
conda activate minigrid

cd src
# Use default image and default prompt
python test_script/etc/test_vlm.py

# Use local image file
python test_script/etc/test_vlm.py --image path/to/image.jpg

# Download image from URL
python test_script/etc/test_vlm.py --image https://picsum.photos/400/300

# Specify user prompt
python test_script/etc/test_vlm.py --prompt "What objects are in this image?"

# Specify both system and user prompts
python test_script/etc/test_vlm.py --system "You are an expert image analyst." --prompt "Analyze this image in detail."

# Specify both image and prompt
python test_script/etc/test_vlm.py -i path/to/image.jpg --command "Describe the colors in this image"
```

**Command-line Options**:
- `--image`, `-i`: Image file path or URL
- `--system-prompt`, `--system`: System prompt
- `--user-prompt`, `--prompt`, `--command`: User prompt/command
- `--help`, `-h`: Show help message

**Supported Models**:
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-4`, `gpt-5`
- **Qwen (Local)**: `Qwen/Qwen2-VL-2B-Instruct`, `Qwen/Qwen2-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-32B-Instruct`, etc.
- **Gemma (Local)**: `google/gemma-2-2b-it`, `google/gemma-2-9b-it`, `google/gemma-2-27b-it`

**Configuration**:
- Model settings can be modified in the `TEST_MODELS` list inside `test_vlm.py` file
- Default image URL and default prompt can also be changed at the top of the file

**Detailed Guide**: [VLM Test Script Guide](docs/test-vlm-guide.md)

---

## Quick Start Examples

### Simple Environment Creation and Control

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from minigrid import register_minigrid_envs
from lib import MiniGridEmojiWrapper, load_emoji_map_from_json

# Register MiniGrid environments
register_minigrid_envs()

# Load environment from JSON map file
map_data = load_emoji_map_from_json('config/example_map.json')
wrapper = MiniGridEmojiWrapper(**map_data)

# Reset environment
obs, info = wrapper.reset()

# Absolute coordinate movement (up/down/left/right)
obs, reward, done, truncated, info = wrapper.step_absolute('North')  # Move up
obs, reward, done, truncated, info = wrapper.step_absolute('East')   # Move right
```

### Automatic Control Using VLM

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from lib import MiniGridEmojiWrapper, ChatGPT4oVLMWrapper, VLMResponsePostProcessor

# Initialize environment and VLM
wrapper = MiniGridEmojiWrapper(...)
vlm = ChatGPT4oVLMWrapper()
postprocessor = VLMResponsePostProcessor()

# Generate action with VLM
image = wrapper.render()
response = vlm.generate(
    image=image,
    system_prompt="You are a robot controller.",
    user_prompt="Move to the goal."
)

# Parse and execute action
action = postprocessor.parse_action(response)
obs, reward, done, truncated, info = wrapper.step_absolute(action)
```

## License

MIT License

## Contributing

Issues and Pull Requests are welcome!

