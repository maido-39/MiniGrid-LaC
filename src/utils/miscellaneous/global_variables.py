######################################################
#                                                    #
#                        GLOBAL                      #
#                      VARIABLES                     #
#                                                    #
######################################################


"""
Global variables for MiniGrid-LaC project.

This module defines global configuration variables used across the MiniGrid-LaC codebase, including:
- settings for Vision Language Models (VLM),
- map configurations,
- mission setups,
- grounding generation,
- and environment rendering options.
"""




######################################################
#                                                    #
#                      VARIABLES                     #
#                                                    #
######################################################


# VLM Configuration
# VLM_MODEL: Select the Vision Language Model to use
#   OpenAI: "gpt-4o" (Basic, Balanced), "gpt-4o-mini" (Fast), "gpt-4", "gpt-4-turbo", "gpt-5"
#   Gemini: "gemini-2.5-flash" (Speed, thinking support), "gemini-1.5-pro" (Accurate), "gemini-1.5-flash" (Fast)
#   Vertex AI: "gemini-2.5-flash-vertex" (logprobs support)
#   Qwen: "qwen2-vl-2b", "qwen2-vl-7b", "qwen2-vl-72b", "qwen2.5-vl-3b", "qwen2.5-vl-7b", "qwen2.5-vl-32b"
#   Gemma: "gemma-2-2b", "gemma-2-9b", "gemma-2-27b"
VLM_MODEL = "gemini-2.5-flash"
# VLM_TEMPERATURE: Generation Randomness Control (0.0 = Deterministic, 1.0 = Random, Recommended: 0.0)
VLM_TEMPERATURE = 0.5
# VLM_MAX_TOKENS: Maximum Output Token Count (Response Length Limit)
VLM_MAX_TOKENS = 3000
# VLM_THINKING_BUDGET: Gemini 2.5 Flash Thinking Budget (Token Units, None=Default, 0 = Inactive)
VLM_THINKING_BUDGET = 0
# LOGPROBS_ENABLED: logprobs activation status (Vertex AI Gemini only)
#   - True: logprobs active (only works when the model name is gemini-*-vertex / -logprobs)
#   - False: logprobs disabled (automatically disabled in OpenAI models or the standard Gemini API)
LOGPROBS_ENABLED = False
# LOGPROBS_TOPK: Number of top-k logprobs (returns the top k probabilities for each token; recommended: 5)
LOGPROBS_TOPK = 5
# DEBUG: Enable or Disable VLM Debug Output
#   - True: Output detailed debug information when calling the VLM API (response, token usage, inference time, etc.)
#   - False: Do not output debug information
DEBUG = False

# Gemini Authentication Configuration
# USE_GCP_KEY: Whether GCP keys are used in standard Gemini models (e.g., gemini-2.5-flash)
#   - True: Using GCP Service Account Key (GOOGLE_APPLICATION_CREDENTIALS Environment Variable)
#   - False: Using the Google AI Studio API Key (GEMINI_API_KEY or GOOGLE_API_KEY environment variable)
#   Note: Vertex AI models (gemini-*-vertex) always use GCP keys (regardless of this setting).
USE_GCP_KEY = True  # True: GCP Keys Usage, False: Google AI Studio Key Usage

# Verbalized Entropy Mode (Based on Tian et al. 2023)
# - True: Verbalized Confidence Method Used (Calculating entropy using step1/step2/step3 probability distributions)
#        → System Prompt: system_prompt_verbalized_entropy.txt
# - False: Use the existing logprobs-based entropy calculation method
#         → System Prompt: system_prompt_start.txt
# Scenario runner and similar tools switch the system prompt based on this value.
USE_VERBALIZED_ENTROPY = False

# Map Configuration
# MAP_FILE_NAME: Name of the map file to use (must be located in the config/ directory)
#   Good examples: "example_map.json", "scenario135_example_map.json"
MAP_FILE_NAME = "scenario4_example_map.json"

# Mission/Task Setup
DEFAULT_INITIAL_MISSION = "Explore your surroundings and gather as much information as possible."
DEFAULT_MISSION = "Pursue your current mission!"

# Path Directory to Prompts Text Files
PROMPT_DIR = "utils/prompts"

# Grounding Generation VLM Configuration
# Grounding Generation Model Configuration (Use VLM_MODEL if None)
GROUNDING_VLM_MODEL = None  # None | "gpt-4o" | "gemini-2.5-flash" | etc.
GROUNDING_VLM_TEMPERATURE = 0.3  # Grounding generation temperature (default: 0.3)
GROUNDING_VLM_MAX_TOKENS = 2000  # max_tokens for grounding generation (default: 2000)
# Note: Grounding is processed in batches at the end of the episode, so the asynchronous option is removed.

# Grounding Merge Format Settings
# GROUNDING_MERGE_FORMAT: Output format when merging multiple Grounding files
#   - "txt": Render in Markdown text format (default)
#   - "json": Merge in JSON format
#   - "both": Both formats provided (currently only “txt” supported)
GROUNDING_MERGE_FORMAT = "txt"  # "txt" | "json" | "both"

# Enable Grounding
# Path to the Grounding file to use (If None, do not use Grounding)
# Single file: "logs/grounding/grounding_latest.txt" or "logs/grounding/grounding_latest.json"
# Multiple files: ["logs/grounding/grounding_latest.txt", "logs/grounding/custom_grounding.txt"] or comma-separated strings
# Example JSON File (RECOMMENDED): "logs/grounding/grounding_latest.json,logs/grounding/episode1_grounding.json"
# Mixed Examples: "logs/grounding/grounding_latest.txt,logs/grounding/grounding_latest.json"
# GROUNDING_FILE_PATH = "logs/scenario2_absolute_example_map_20260123_170550/episode_1_20260123_170552_scenario2_test_absolutemove_modularized/grounding_episode_1.txt,logs/scenario2_absolute_example_map_20260123_165532/episode_1_20260123_165535_scenario2_test_absolutemove_modularized/grounding_episode_1.txt"  # None | str | List[str] | "file1.txt,file2.txt"
GROUNDING_FILE_PATH = "logs_good/Stan/Predefined_Grounding_Scenario_2/grounding/predefined_grounding.txt"

# Grounding created in the current episode will be available starting from the next episode (not applied immediately).
# Multiple file support: comma-separated strings or list format
# JSON files are automatically merged and rendered in Markdown format.

# Environment Rendering Configuration
# RENDER_GOAL: Goal (green target point) rendering status
#   - True: Goal Rendering (highlighted in green)
#   - False: Goal not rendered (not visually displayed)
RENDER_GOAL = True

#
ENV_ID = "MyCustomEnv-v0"



