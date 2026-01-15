######################################################
#                                                    #
#                     PROMPT INPUTS                  #
#                    INTERPRETATION                  #
#                                                    #
######################################################


""""""




######################################################
#                                                    #
#                      LIBRARIES                     #
#                                                    #
######################################################


import os
import re
from pathlib import Path
from string import Template

from utils.global_variables import PROMPT_DIR




######################################################
#                                                    #
#                     FUNCTIONS                      #
#                                                    #
######################################################


def mission_input_interp(user_input, actual_default) -> str:
    # Case 1: user pressed Enter → fallback
    if not user_input:
        mission = actual_default
    
    # Case 2: user entered a file path
    elif os.path.isfile(user_input) and user_input.endswith(".txt"):
        with open(user_input, "r", encoding="utf-8") as f:
            mission = f.read().strip()
        if not mission:
            raise ValueError("Mission file is empty.")
    
    # Case 3: user typed a mission directly
    else:
        mission = user_input
    
    # Final prompt sent to VLM
    return (
        f"Task: {mission}\n\n"
        "Based on the current image, choose the next action to complete this task. "
        "Use absolute directions (up/down/left/right)."
    )

def system_prompt_interp(file_name: str, strict: bool = False, **variables) -> str:
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
    
    prompt_path_str = PROMPT_DIR + "/" + file_name
    # Build the full path using the global prompt directory
    prompt_path = Path(prompt_path_str)
    
    # Read the entire prompt template file into a string
    template_text = prompt_path.read_text(encoding="utf-8")
    
    # Detect which variables are used in the template ($var or ${var})
    used_vars = set(re.findall(r"\$(\w+)|\$\{(\w+)\}", template_text))
    used_vars = {v for pair in used_vars for v in pair if v}
    
    # In strict mode, fail fast if required variables are missing
    if strict:
        missing = used_vars - variables.keys()
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



