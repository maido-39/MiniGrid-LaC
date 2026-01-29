######################################################
#                                                    #
#                     PROMPT INPUTS                  #
#                    INTERPRETATION                  #
#                                                    #
######################################################


"""
Module for interpreting prompt inputs and performing variable substitution in prompt templates.

This module provides functions to load prompt templates from files, perform security checks, and substitute variables dynamically.
It is designed to be flexible and future-proof, allowing easy addition or removal of variables without changing the core logic.

Functions
---------
- file_checking: Perform security and existence checks on prompt file paths.
- mission_input_interp: Interpret user input for mission prompts, handling direct text or file paths.
- system_prompt_interp: Load and render prompt templates with variable substitution.
"""




######################################################
#                                                    #
#                      LIBRARIES                     #
#                                                    #
######################################################


import re
from pathlib import Path
from string import Template

from utils.miscellaneous.global_variables import PROMPT_DIR
import utils.prompt_manager.terminal_formatting_utils as tfu




######################################################
#                                                    #
#                     FUNCTIONS                      #
#                                                    #
######################################################


def file_checking(base_dir: Path, prompt_path: Path, terminal_msg: bool = False):
    """
    
    """
    
    # 🔒 Security check (path traversal)
    if not prompt_path.is_relative_to(base_dir):
        raise ValueError("Invalid prompt path (path traversal detected)")

    # 📁 Existence check
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    if terminal_msg:
        tfu.cprint("[Terminal] File Path found!", tfu.LIGHT_GREEN)

def mission_input_interp(user_input, actual_default) -> str:
    """
    
    """
    
    # Case 1: user pressed Enter → fallback
    if not user_input:
        mission = actual_default
    
    # Case 2: user entered a .txt file path
    elif "/" in user_input or user_input.endswith(".txt"):
        base_dir = Path(PROMPT_DIR).resolve()
        # Build the full path using the global prompt directory
        prompt_path = (base_dir / user_input).resolve()
        
        # Security check
        file_checking(base_dir, prompt_path)
        
        # 📖 Read file
        mission = prompt_path.read_text(encoding="utf-8").strip()
        if not mission:
            raise ValueError("Mission file is empty!")
    
    # Case 3: user typed a mission directly
    else:
        mission = user_input
    
    # Final prompt sent to VLM
    return system_prompt_interp(
        file_name="task_prompt.txt",
        strict=True,
        mission=mission,
    )

def system_prompt_interp(file_name: str,
                         strict: bool = False,
                         **variables
                        ) -> str:
    """
    Load a prompt template from the global prompt directory and
    substitute variables dynamically.

    This function is designed to be future-proof:
    - You only pass the prompt file_name
    - The base path is defined globally (PROMPT_DIR)
    - Any variables can be added or removed without changing this function
    - Safe for JSON-heavy prompts

    Parameters
    ----------
    file_name : str
        Name of the prompt file (e.g. "system_prompt_start.txt").

    strict : bool, optional
        If True, raise an error when a variable required by the prompt
        is missing from `variables`.
        If False (default), missing variables are left untouched.

    **variables : dict
        Arbitrary variables used in the prompt template.
        Example: grounding_content="...", last_action_str="..."

    Returns
    -------
    str
        Fully rendered system prompt ready to be sent to the LLM.
    """
    
    base_dir = Path(PROMPT_DIR).resolve()
    # Build the full path using the global prompt directory
    prompt_path = (base_dir / file_name).resolve()
    
    # Security check
    file_checking(base_dir, prompt_path)
    
    # Read the entire prompt template file into a string
    template_text = prompt_path.read_text(encoding="utf-8")
    
    # Detect which variables are used in the template ($var or ${var})
    used_vars = set(re.findall(r"\$(\w+)|\$\{(\w+)\}", template_text))
    used_vars = {v for pair in used_vars for v in pair if v}
    
    # In strict mode, fail fast if required variables are missing
    if strict:
        missing = sorted(used_vars - variables.keys())
        if missing:
            raise ValueError(
                f"Missing variables for prompt '{file_name}': {missing}"
            )

    # Create a Template object for safe substitution
    template = Template(template_text)

    # Substitute provided variables
    # Missing variables remain unchanged unless strict=True
    filled_prompt = template.safe_substitute(**variables)

    return filled_prompt



