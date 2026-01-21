"""
MiniGrid VLM Interaction Script (Absolute Movement Version - Emoji Environment)
Modularized Version using ScenarioExperiment

This is a modularized version of VLM_interact_minigrid-absolute_emoji.py that uses
the ScenarioExperiment class from utils.miscellaneous.scenario_runner.

Usage:
    python VLM_interact_minigrid-absolute_emoji_modularized.py [json_map_path]
"""

import utils.prompt_manager.terminal_formatting_utils as tfu
from utils.miscellaneous.scenario_runner import ScenarioExperiment
from utils.miscellaneous.safe_minigrid_registration import safe_minigrid_reg
from utils.prompt_manager.prompt_organizer import PromptOrganizer
from utils.vlm.vlm_processor import VLMProcessor
from utils.vlm.vlm_postprocessor import VLMResponsePostProcessor
from utils.miscellaneous.global_variables import MAP_FILE_NAME, DEBUG, LOGPROBS_ENABLED

safe_minigrid_reg()


# ============================================================================
# Configuration (Edit these settings)
# ============================================================================
# Note: DEBUG and USE_LOGPROBS are now imported from global_variables.py
# To change these settings, edit global_variables.py instead
USE_LOGPROBS = LOGPROBS_ENABLED  # Use global LOGPROBS_ENABLED setting
# ============================================================================


# Hardcoded system prompt (exact same as VLM_interact_minigrid-absolute_emoji.py)
SYSTEM_PROMPT = """You are a robot operating on a grid map.

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

# Hardcoded default user prompt
DEFAULT_USER_PROMPT = "Based on the current image, choose the next action to complete the mission: Go to the blue brick emoji 🧱, turn right, then stop next to the desktop/workstation emoji 🖥️📱. Use absolute directions (up/down/left/right)."


class VLMInteractPromptOrganizer(PromptOrganizer):
    """Prompt Organizer with hardcoded prompts"""
    
    def get_system_prompt(self, wrapper=None, last_action_result=None) -> str:
        return SYSTEM_PROMPT
    
    def get_user_prompt(self, default_prompt: str = None, init_step: bool = False) -> str:
        if default_prompt is None:
            default_prompt = DEFAULT_USER_PROMPT
        print("Enter command (Enter: default prompt):")
        user_input = input("> ").strip()
        self._raw_user_input = user_input
        return default_prompt if not user_input else user_input


class VLMInteractExperiment(ScenarioExperiment):
    """VLM Interact Experiment with custom configuration"""
    
    def __init__(self, log_dir=None, json_map_path=None, debug=None, use_logprobs=None):
        # Set attributes before super().__init__() calls _create_vlm_processor()
        # Use global DEBUG if debug is not provided
        self.debug = debug if debug is not None else DEBUG
        # Use global LOGPROBS_ENABLED if use_logprobs is not provided
        self.use_logprobs = use_logprobs if use_logprobs is not None else LOGPROBS_ENABLED
        self.custom_postprocessor = VLMResponsePostProcessor(required_fields=["action", "environment_info"])
        
        # Use global MAP_FILE_NAME if json_map_path is not provided
        if json_map_path is None:
            json_map_path = f"config/{MAP_FILE_NAME}"
        
        prompt_organizer = VLMInteractPromptOrganizer()
        super().__init__(
            log_dir=log_dir,
            json_map_path=json_map_path,
            prompt_organizer=prompt_organizer,
            use_logprobs=self.use_logprobs,
            debug=self.debug
        )
        
        # Override postprocessor after initialization
        self.vlm_processor.postprocessor_action = VLMResponsePostProcessor(required_fields=["action", "environment_info"])
    
    def vlm_gen_action(self, image, system_prompt: str, user_prompt: str, use_logprobs: bool = True):
        tfu.cprint("\n[3] Sending request to VLM...", tfu.LIGHT_BLACK)
        
        use_lp = self.use_logprobs if use_logprobs is None else (use_logprobs and self.logprobs_active)
        
        if use_lp and self.logprobs_active:
            raw_response, logprobs_metadata = self.vlm_processor.requester_with_logprobs(
                image=image, system_prompt=system_prompt, user_prompt=user_prompt, debug=self.debug
            )
            self.logprobs_metadata = logprobs_metadata
            
            if self.debug:
                tfu.cprint(f"[Debug] logprobs_metadata keys: {list(logprobs_metadata.keys()) if logprobs_metadata else 'empty'}", tfu.LIGHT_BLACK)
            
            tfu.cprint("VLM response received", tfu.LIGHT_GREEN)
            tfu.cprint("[4] Parsing response...", tfu.LIGHT_BLACK)
            parsed = self.custom_postprocessor.process(raw_response, strict=True)
            
            if self.logprobs_metadata:
                self.action_logprobs_info = self.custom_postprocessor.get_action_logprobs(
                    self.logprobs_metadata, action_field="action"
                )
                
                # Print logprobs info if debug is enabled
                if self.debug:
                    if self.action_logprobs_info:
                        self.custom_postprocessor.print_action_logprobs_info(self.action_logprobs_info)
                    else:
                        tfu.cprint("[Debug] action_logprobs_info is empty", tfu.LIGHT_RED)
            else:
                if self.debug:
                    tfu.cprint("[Debug] logprobs_metadata is empty, skipping logprobs processing", tfu.LIGHT_RED)
            
            # Ensure compatibility with parent class's _log_step() expectations
            # Add missing fields that parent class expects
            if 'memory' not in parsed:
                parsed['memory'] = {
                    "spatial_description": "",
                    "task_process": {"goal": "", "status": "", "blocked_reason": ""},
                    "previous_action": ""
                }
            if 'grounding' not in parsed:
                parsed['grounding'] = ""
            if 'reasoning' not in parsed:
                parsed['reasoning'] = parsed.get('environment_info', '')
            
            action_str = parsed.get('action', 'up')
            tfu.cprint(f"Parsed action: {action_str}", tfu.LIGHT_BLACK)
            tfu.cprint(f"Environment Info: {parsed.get('environment_info', 'N/A')}", tfu.LIGHT_BLACK)
            tfu.cprint(f"Reasoning: {parsed.get('reasoning', 'N/A')}", tfu.LIGHT_BLACK)
            self.vlm_response_raw = raw_response
            return parsed
        else:
            raw_response = self.vlm_processor.requester(
                image=image, system_prompt=system_prompt, user_prompt=user_prompt, debug=self.debug
            )
            tfu.cprint("VLM response received", tfu.LIGHT_GREEN)
            tfu.cprint("[4] Parsing response...", tfu.LIGHT_BLACK)
            parsed = self.custom_postprocessor.process(raw_response, strict=True)
            
            # Ensure compatibility with parent class's _log_step() expectations
            # Add missing fields that parent class expects
            if 'memory' not in parsed:
                parsed['memory'] = {
                    "spatial_description": "",
                    "task_process": {"goal": "", "status": "", "blocked_reason": ""},
                    "previous_action": ""
                }
            if 'grounding' not in parsed:
                parsed['grounding'] = ""
            if 'reasoning' not in parsed:
                parsed['reasoning'] = parsed.get('environment_info', '')
            
            action_str = parsed.get('action', 'up')
            tfu.cprint(f"Parsed action: {action_str}", tfu.LIGHT_BLACK)
            tfu.cprint(f"Environment Info: {parsed.get('environment_info', 'N/A')}", tfu.LIGHT_BLACK)
            tfu.cprint(f"Reasoning: {parsed.get('reasoning', 'N/A')}", tfu.LIGHT_BLACK)
            self.vlm_response_raw = raw_response
            return parsed


def main():
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h"]:
            print("Usage: python VLM_interact_minigrid-absolute_emoji_modularized.py [json_map_path]")
            print(f"Example: python VLM_interact_minigrid-absolute_emoji_modularized.py config/{MAP_FILE_NAME}")
            print(f"Default: Uses MAP_FILE_NAME from global_variables.py (currently: {MAP_FILE_NAME})")
            return
        json_map_path = sys.argv[1]
    else:
        # Use global MAP_FILE_NAME
        json_map_path = None  # Will be set to config/{MAP_FILE_NAME} in __init__
    
    experiment = VLMInteractExperiment(
        json_map_path=json_map_path,
        debug=DEBUG,
        use_logprobs=USE_LOGPROBS
    )
    experiment.run()
    experiment.cleanup()


if __name__ == "__main__":
    main()
