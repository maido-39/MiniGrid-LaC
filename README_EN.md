# MiniGrid-LaC

A project for Language-conditioned Reinforcement Learning in MiniGrid environments.

## Overview

This project implements a language-based agent control system using Vision Language Models (VLM) in MiniGrid environments. It provides various features including absolute coordinate movement, emoji map system, grounding knowledge system, and entropy-based uncertainty analysis.

## Key Features

- **VLM-based Automatic Control**: Support for various VLM models (GPT-4o, Gemini, Qwen, Gemma)
- **Absolute Coordinate Movement**: Direct up/down/left/right movement regardless of robot direction
- **Emoji Map System**: Define and load emoji-based maps from JSON files
- **Grounding Knowledge System**: Learn from mistakes through user feedback and accumulate knowledge
- **Entropy Analysis**: Quantify VLM action uncertainty and calculate Trust values
- **Episode Management**: Episode-based logging and grounding generation
- **Multiple VLM Handlers**: Unified interface for OpenAI, Gemini, Qwen, Gemma
- **Logprobs Support**: Probability distribution analysis via Vertex AI Gemini

## Project Structure

```
multigrid-LaC/
├── src/                          # Source code directory
│   ├── utils/                    # Core utility modules
│   │   ├── map_manager/          # Map and environment management
│   │   │   ├── minigrid_customenv_emoji.py    # Main environment wrapper (emoji support, absolute movement)
│   │   │   └── emoji_map_loader.py            # JSON map loader for emoji-based maps
│   │   ├── vlm/                  # Vision Language Model modules
│   │   │   ├── vlm_wrapper.py                 # VLM API wrapper (unified interface)
│   │   │   ├── vlm_postprocessor.py          # VLM response parser and validator
│   │   │   ├── vlm_processor.py              # VLM processing logic
│   │   │   ├── vlm_controller.py             # Generic VLM controller for environment control
│   │   │   ├── vlm_manager.py                # VLM handler manager (multi-provider support)
│   │   │   └── handlers/                     # VLM provider handlers
│   │   │       ├── base.py                    # Base handler class
│   │   │       ├── openai_handler.py         # OpenAI GPT-4o handler
│   │   │       ├── gemini_handler.py         # Google Gemini handler
│   │   │       ├── qwen_handler.py           # Qwen VLM handler
│   │   │       └── gemma_handler.py          # Gemma handler
│   │   ├── miscellaneous/                    # Miscellaneous utilities
│   │   │   ├── scenario_runner.py            # ScenarioExperiment class (main experiment runner)
│   │   │   ├── episode_manager.py           # Episode management and logging
│   │   │   ├── grounding_file_manager.py     # Grounding knowledge file management
│   │   │   ├── visualizer.py                # Visualization utilities
│   │   │   ├── global_variables.py          # Global configuration variables
│   │   │   └── safe_minigrid_registration.py # Safe MiniGrid environment registration
│   │   ├── prompt_manager/                   # Prompt management
│   │   │   ├── prompt_organizer.py          # System/user prompt organization
│   │   │   ├── prompt_interp.py             # Prompt interpolation
│   │   │   └── terminal_formatting_utils.py # Terminal formatting utilities
│   │   ├── user_manager/                     # User interaction
│   │   │   └── user_interact.py             # User interaction handler
│   │   ├── prompts/                          # Prompt templates
│   │   │   ├── system_prompt_start.txt      # System prompt template
│   │   │   ├── task_prompt.txt               # Task prompt template
│   │   │   ├── grounding_generation_prompt.txt    # Grounding generation prompt
│   │   │   ├── reflexion_prompt.txt         # Reflexion prompt
│   │   │   └── feedback_prompt.txt          # Feedback prompt
│   │   └── scripts/                          # Utility scripts
│   │       └── json_to_csv_converter.py     # JSON to CSV converter
│   ├── legacy/                   # Legacy code (maintained for backward compatibility)
│   │   ├── scenario2_test_absolutemove.py   # Legacy scenario 2 script
│   │   └── VLM_interact_minigrid-absolute_emoji.py  # Legacy VLM interaction
│   ├── dev-*/                    # Development branches (experimental features)
│   │   ├── dev-scenario_2/       # Scenario 2 development
│   │   └── dev-action_uncertainty/ # Action uncertainty estimation experiments
│   ├── test_script/              # Test and example scripts
│   │   ├── emoji_test/           # Emoji rendering tests
│   │   ├── keyboard_control/    # Keyboard control examples
│   │   ├── action_entropy/      # Action entropy analysis
│   │   ├── etc/                  # Miscellaneous test scripts
│   │   └── similarity_calculator/ # Text similarity utilities
│   ├── asset/                    # Resource files
│   │   ├── arrow.png             # Robot arrow marker image
│   │   └── fonts/                # Font files for emoji rendering
│   ├── config/                   # Configuration files
│   │   ├── example_map.json      # Example emoji map configuration
│   │   ├── scenario135_example_map.json  # Scenario 135 example map
│   │   └── test_pickup_map.json  # Test pickup map
│   ├── minigrid_lac.py          # Main entry point (recommended)
│   ├── scenario2_test_entropy_comparison.py  # Entropy comparison experiment
│   └── scenario2_test_absolutemove_modularized.py  # Modularized scenario 2 script
├── logs/                         # Experiment logs (generated at runtime)
├── docs/                         # Documentation
└── requirements.txt             # Python dependencies
```

### Directory Purposes

- **`src/utils/`**: Core reusable utility modules
  - **`map_manager/`**: Environment creation and map loading utilities
  - **`vlm/`**: VLM integration modules for robot control
  - **`miscellaneous/`**: Experiment runner, episode management, grounding system
  - **`prompt_manager/`**: Prompt organization and formatting
  - **`user_manager/`**: User interaction handling

- **`src/legacy/`**: Legacy code maintained for backward compatibility
  - Old scripts and modules (use new modularized versions instead)

- **`src/dev-*/`**: Experimental development branches
  - Active development features that may be merged into main library later

- **`src/test_script/`**: Test and example scripts
  - Various test scripts, examples, and utility scripts

- **`src/asset/`**: Static resource files
  - Images, fonts, and other assets used by the environment

- **`src/config/`**: Configuration files
  - JSON map files and other configuration data

### Import Usage

All modules can be imported using the `utils` path:

```python
# Recommended imports
from utils.map_manager.emoji_map_loader import load_emoji_map_from_json
from utils.map_manager.minigrid_customenv_emoji import MiniGridEmojiWrapper
from utils.vlm.vlm_controller import VLMController
from utils.vlm.vlm_wrapper import VLMWrapper
from utils.miscellaneous.scenario_runner import ScenarioExperiment
from utils.miscellaneous.episode_manager import EpisodeManager
from utils.miscellaneous.grounding_file_manager import GroundingFileManager
```

## Documentation

Detailed documentation is available in the [`docs/`](docs/) folder:

### MiniGrid Basics
- [MiniGrid Environment List](docs/minigrid-environments.md) - All built-in MiniGrid environments
- [MiniGrid Objects and Attributes](docs/minigrid-objects.md) - Available object types and attributes in MiniGrid
- [Environment Creation Guide](docs/environment-creation.md) - How to create MiniGrid environments
- [Best Practices](docs/best-practices.md) - Recommended practices for MiniGrid environment creation

### API Documentation
- [Wrapper API](docs/wrapper-api.md) - MiniGridEmojiWrapper API documentation (includes absolute movement)
- [Wrapper Methods Guide](docs/wrapper-methods.md) - All Wrapper method descriptions
- [VLM Handlers System Guide](docs/vlm-handlers.md) - Using various VLM models (OpenAI, Qwen, Gemma, Gemini)
- [Similarity Calculator API](docs/similarity-calculator-api.md) - Word2Vec and SBERT similarity calculation API

### Usage Guides
- [API Key Setup Guide](docs/LLM-API/api-key-setup.md) - OpenAI, Gemini, Vertex AI API Key setup methods
- [Keyboard Control Guide](docs/keyboard-control.md) - Keyboard control example explanation
- [VLM Test Script Guide](docs/test-vlm-guide.md) - VLM model testing and comparison guide
- [Emoji Map JSON Loader Guide](docs/emoji-map-loader.md) - Loading emoji maps from JSON files
- [SLAM-style FOV Mapping Guide](docs/slam-fov-mapping.md) - Exploration area tracking and field of view limitation
- [Emoji Usage Guide](docs/EMOJI_USAGE_GUIDE.md) - Using emoji objects
- [Entropy and Trust Calculation Guide](docs/entropy-trust-calculation.md) - VLM action uncertainty analysis
- [VLM Action Uncertainty Guide](docs/vlm-action-uncertainty.md) - Action uncertainty measurement and visualization

### LLM API Documentation
- [Gemini Thinking Feature Guide](docs/LLM-API/gemini-thinking.md) - Using Thinking feature in Gemini 2.5/3 series

## Installation

### Requirements

- Python 3.8 or higher (Python 3.10 recommended)
- API keys (for VLM features):
  - OpenAI API key (for GPT-4o, etc.)
  - Gemini API key (for Gemini models)
  - Vertex AI setup (for logprobs feature)
  - DashScope API key (for Qwen models)

**📖 API Key Setup**: See [API Key Setup Guide](docs/LLM-API/api-key-setup.md)

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

# Set API keys (create .env file)
# See docs/LLM-API/api-key-setup.md for detailed setup
echo "OPENAI_API_KEY=your-api-key-here" > .env
echo "GEMINI_API_KEY=your-api-key-here" >> .env  # For Gemini
echo "DASHSCOPE_API_KEY=your-api-key-here" >> .env  # For Qwen
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

# Set API keys (create .env file)
# See docs/LLM-API/api-key-setup.md for detailed setup
echo "OPENAI_API_KEY=your-api-key-here" > .env
echo "GEMINI_API_KEY=your-api-key-here" >> .env  # For Gemini
echo "DASHSCOPE_API_KEY=your-api-key-here" >> .env  # For Qwen
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
python minigrid_lac.py

# Method 2: Set PYTHONPATH from project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python src/minigrid_lac.py
```

### Main Execution Scripts

#### `minigrid_lac.py` - Main Entry Point ⭐ **Recommended**

**Description**: Main execution script using the modularized experiment system. Uses the ScenarioExperiment class to provide all features.

**Features**:
- Load emoji maps from JSON files
- Absolute coordinate movement
- Automatic agent control via VLM
- Permanent memory system and grounding knowledge system
- Episode management and logging
- Comprehensive logging (images, JSON, CSV, VLM I/O logs)

**How to Run**:
```bash
cd src
# Use default map file (MAP_FILE_NAME from global_variables.py)
python minigrid_lac.py

# Specify specific JSON map file
python minigrid_lac.py config/example_map.json

# Show help
python minigrid_lac.py --help
```

**Configuration**: Can be changed in `src/utils/miscellaneous/global_variables.py`
- `VLM_MODEL`: VLM model to use (default: "gemini-2.5-flash-vertex")
- `VLM_TEMPERATURE`: Generation temperature (default: 0.5)
- `VLM_MAX_TOKENS`: Maximum token count (default: 3000)
- `LOGPROBS_ENABLED`: Enable logprobs (default: True)
- `MAP_FILE_NAME`: Default map file name (default: "example_map.json")
- `USE_NEW_GROUNDING_SYSTEM`: Use new grounding system (default: True)

**Log Output**:
- Saved in `logs/scenario2_absolute_<map_name>_<timestamp>/` directory
  - `episode_<N>_<timestamp>_<script_name>/`: Directory for each episode
    - `step_XXXX.png`: Environment image for each step
    - `episode_<N>.json`: Episode JSON log
    - `grounding_episode_<N>.json`: Grounding knowledge (JSON)
    - `grounding_episode_<N>.txt`: Grounding knowledge (TXT)
  - `grounding/`: Latest grounding files
    - `grounding_latest.json`: Latest grounding (JSON)
    - `grounding_latest.txt`: Latest grounding (TXT)
  - `experiment_log.json`: Complete experiment JSON log (cumulative)
  - `experiment_log.csv`: Experiment data CSV (cumulative)

---

#### `scenario2_test_entropy_comparison.py` - Entropy Comparison Experiment

**Description**: Entropy comparison experiment script for analyzing VLM action uncertainty. Calls VLM with 3 conditions (H(X), H(X|S), H(X|L,S)) to calculate Trust values.

**Features**:
- Simultaneous VLM calls with 3 conditions
- Entropy calculation and Trust value calculation
- Logprobs-based probability distribution analysis
- CSV logging (including Entropy and Trust values)

**How to Run**:
```bash
cd src
# Use default map file
python scenario2_test_entropy_comparison.py

# Specify specific JSON map file
python scenario2_test_entropy_comparison.py config/scenario135_example_map.json

# Show help
python scenario2_test_entropy_comparison.py --help
```

**Requirements**:
- `LOGPROBS_ENABLED = True` (in global_variables.py)
- Vertex AI Gemini model (logprobs support)

**Detailed Guide**: [Entropy and Trust Calculation Guide](docs/entropy-trust-calculation.md)

---

### Example Scripts

#### 1. `test_script/keyboard_control/keyboard_control.py` - Keyboard Control Example

**Description**: Simple example script for directly controlling MiniGrid environments with keyboard input.

**How to Run**:
```bash
cd src
python test_script/keyboard_control/keyboard_control.py
```

**Controls**:
- `w`: Move forward
- `a`: Turn left
- `d`: Turn right
- `s`: Move backward
- `r`: Reset environment
- `q`: Quit

---

#### 2. `dev-scenario_2/scenario2_keyboard_control.py` - Scenario 2 Keyboard Control (Absolute Movement)

**Description**: Script for directly controlling Scenario 2 environment with keyboard. Uses absolute coordinate movement.

**How to Run**:
```bash
cd src
python dev-scenario_2/scenario2_keyboard_control.py
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

---

#### 3. `test_script/etc/test_vlm.py` - VLM Model Testing and Comparison

**Description**: Script for testing and comparing various VLM models.

**How to Run**:
```bash
cd src
# Use default image and default prompt
python test_script/etc/test_vlm.py

# Use local image file
python test_script/etc/test_vlm.py --image path/to/image.jpg

# Specify user prompt
python test_script/etc/test_vlm.py --prompt "What objects are in this image?"
```

**Detailed Guide**: [VLM Test Script Guide](docs/test-vlm-guide.md)

---

## Quick Start Examples

### Simple Environment Creation and Control

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.map_manager.emoji_map_loader import load_emoji_map_from_json
from utils.map_manager.minigrid_customenv_emoji import MiniGridEmojiWrapper
from utils.miscellaneous.safe_minigrid_registration import safe_minigrid_reg

# Register MiniGrid environments
safe_minigrid_reg()

# Load environment from JSON map file
wrapper = load_emoji_map_from_json('config/example_map.json')

# Reset environment
obs, info = wrapper.reset()

# Absolute coordinate movement (up/down/left/right)
obs, reward, done, truncated, info = wrapper.step_absolute('move up')    # Move up
obs, reward, done, truncated, info = wrapper.step_absolute('move right') # Move right
obs, reward, done, truncated, info = wrapper.step_absolute(0)            # Move up (index)
obs, reward, done, truncated, info = wrapper.step_absolute('north')       # Move up (alias)

# Get current state
state = wrapper.get_state()
print(f"Agent position: {state['agent_pos']}")
print(f"Agent direction: {state['agent_dir']}")

# Get environment image (for VLM input)
image = wrapper.get_image()
print(f"Image shape: {image.shape}")  # (height, width, 3)
```

### Automatic Control Using VLM

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.map_manager.emoji_map_loader import load_emoji_map_from_json
from utils.vlm.vlm_controller import VLMController
from utils.miscellaneous.safe_minigrid_registration import safe_minigrid_reg

# Register MiniGrid environments
safe_minigrid_reg()

# Create environment
wrapper = load_emoji_map_from_json('config/example_map.json')
wrapper.reset()

# Create VLM controller
controller = VLMController(
    env=wrapper,
    model="gpt-4o",
    temperature=0.0
)

# Generate and execute action with VLM
obs, reward, done, truncated, info, vlm_response = controller.step(
    mission="Go to the blue pillar"
)

print(f"Action: {vlm_response['action']}")
print(f"Reasoning: {vlm_response.get('reasoning', 'N/A')}")
```

### Complete Experiment Using ScenarioExperiment

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.miscellaneous.scenario_runner import ScenarioExperiment
from utils.miscellaneous.safe_minigrid_registration import safe_minigrid_reg

# Register MiniGrid environments
safe_minigrid_reg()

# Create and run experiment
experiment = ScenarioExperiment(
    json_map_path="config/example_map.json"
)
experiment.run()
```

## Key Features Details

### 1. Absolute Coordinate Movement

Move directly up/down/left/right regardless of robot's current direction:

```python
wrapper.step_absolute('move up')      # Move up
wrapper.step_absolute('move down')    # Move down
wrapper.step_absolute('move left')    # Move left
wrapper.step_absolute('move right')   # Move right
```

### 2. Emoji Map System

Define and load emoji-based maps from JSON files:

```json
{
  "size": 10,
  "map": [
    "🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦",
    "🟦🟫🟫🟫🟫🟫🟫🟫🟫🟦",
    "🟦🟫🟫🟫🟫🟫🟫🟫🟫🟦",
    ...
  ],
  "objects": {
    "🟦": {"type": "wall", "color": "blue"},
    "🟫": {"type": "floor", "color": "brown"}
  }
}
```

### 3. Grounding Knowledge System

Learn from mistakes through user feedback and accumulate knowledge:

- Automatic grounding generation at episode end
- Save in JSON/TXT format
- Automatically applied from next episode

### 4. Entropy and Trust Calculation

Quantify VLM action uncertainty:

- **H(X)**: Entropy without Language Instruction and Grounding
- **H(X|S)**: Entropy with only Grounding
- **H(X|L,S)**: Entropy with both Grounding and Language Instruction
- **Trust T**: `(H(X) - H(X|S)) / (H(X) - H(X|L,S))`

### 5. Episode Management

Manage logs and generate grounding per episode:

- Create directory for each episode
- Save episode JSON log
- Automatically generate grounding files

## License

MIT License

## Contributing

Issues and Pull Requests are welcome!
