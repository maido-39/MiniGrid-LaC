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


import logging
import re
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional

from utils.miscellaneous.global_variables import PROMPT_DIR
import utils.prompt_manager.terminal_formatting_utils as tfu
from utils.prompt_manager.memory_renderer import render_memory_value

logger = logging.getLogger(__name__)




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

# Pattern for $memory[key] or $memory[key][subkey]... (keys: alphanumeric, underscore, hyphen)
_MEMORY_BRACKET_PATTERN = re.compile(r"\$memory(?:\[[a-zA-Z0-9_-]+\])+")


def _resolve_memory_path(memory_dict: Dict[str, Any], keys: List[str]) -> Any:
    """Resolve keys like ['task_process', 'goal'] to memory_dict['task_process']['goal']."""
    obj: Any = memory_dict
    for key in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def _substitute_memory_brackets(
    template_text: str,
    memory_dict: Optional[Dict[str, Any]],
    default_for_missing: str = "None",
    strict: bool = False,
    file_name: str = "",
) -> str:
    """
    Replace all $memory[key] and $memory[key][subkey]... with rendered values.
    Keys are restricted to [a-zA-Z0-9_-]+. Uses dict.get() only; no eval.
    When strict=True and a key is missing, logs a warning and still substitutes default.
    """
    if memory_dict is None or not isinstance(memory_dict, dict):
        memory_dict = {}

    def replacer(match: re.Match) -> str:
        placeholder = match.group(0)
        keys = re.findall(r"\[([a-zA-Z0-9_-]+)\]", placeholder)
        if not keys:
            return default_for_missing
        value = _resolve_memory_path(memory_dict, keys)
        if value is None:
            msg = (
                f"Missing memory key: template uses '{placeholder}' but key path {keys} is not in memory dict. "
                f"Prompt file: '{file_name}'. Substituting default '{default_for_missing}'."
            )
            logger.warning(msg)
            tfu.cprint("[Prompt] MISSING MEMORY KEY", tfu.LIGHT_RED, bold=True)
            tfu.cprint(f"  Placeholder: {placeholder}", tfu.LIGHT_RED)
            tfu.cprint(f"  Key path: {keys}", tfu.LIGHT_RED)
            tfu.cprint(f"  File: {file_name}", tfu.LIGHT_RED)
            tfu.cprint(f"  Using default: {default_for_missing}", tfu.LIGHT_YELLOW)
        return render_memory_value(value, default_for_empty=default_for_missing)

    return _MEMORY_BRACKET_PATTERN.sub(replacer, template_text)


def system_prompt_interp(file_name: str,
                         strict: bool = False,
                         **variables
                        ) -> str:
    """
    Load a prompt template from the global prompt directory and
    substitute variables dynamically.

    Supports:
    - Flat variables: $var_name, ${var_name}
    - Memory bracket syntax: $memory[key], $memory[key][subkey] (resolved from
      variables['memory'] dict, then rendered via MemoryRenderer)

    Parameters
    ----------
    file_name : str
        Name of the prompt file (e.g. "system_prompt_start.txt").

    strict : bool, optional
        If True, raise an error when a variable required by the prompt
        is missing from `variables`.
        If False (default), missing variables are left untouched.
        If template contains $memory[...], memory must be in variables and
        be a dict (or normalized to {}).

    **variables : dict
        Arbitrary variables. Pass memory=<dict> for $memory[key] substitution.

    Returns
    -------
    str
        Fully rendered system prompt ready to be sent to the LLM.
    """
    
    base_dir = Path(PROMPT_DIR).resolve()
    prompt_path = (base_dir / file_name).resolve()
    file_checking(base_dir, prompt_path)
    
    template_text = prompt_path.read_text(encoding="utf-8")
    
    # Normalize memory: must be dict for bracket substitution
    memory_dict = variables.get("memory")
    if memory_dict is None or not isinstance(memory_dict, dict):
        memory_dict = {}
    if strict and _MEMORY_BRACKET_PATTERN.search(template_text):
        if "memory" not in variables:
            raise ValueError(
                f"Missing variable for prompt '{file_name}': template uses $memory[...] but 'memory' not provided"
            )
        if not isinstance(variables.get("memory"), dict):
            raise ValueError(
                f"Variable 'memory' for prompt '{file_name}' must be a dict (got {type(variables.get('memory'))})"
            )
    
    # 1) Substitute $memory[key] and $memory[key][subkey]... first (missing key → default + warning if strict)
    template_text = _substitute_memory_brackets(
        template_text, memory_dict, strict=strict, file_name=file_name
    )
    
    # 2) Detect flat variables ($var or ${var}) for strict check
    used_vars = set(re.findall(r"\$(\w+)|\$\{(\w+)\}", template_text))
    used_vars = {v for pair in used_vars for v in pair if v}
    
    vars_for_template = {k: v for k, v in variables.items() if k != "memory"}
    if strict:
        # Require all flat variables; 'memory' is handled by bracket substitution, so exclude from Template requirements
        required_flat = used_vars - {"memory"}
        missing = sorted(required_flat - vars_for_template.keys())
        if missing:
            raise ValueError(
                f"Missing variables for prompt '{file_name}': {missing}"
            )
    
    # 3) Substitute flat variables ($var); pass variables without 'memory' so $memory is not replaced with dict repr
    template = Template(template_text)
    filled_prompt = template.safe_substitute(**vars_for_template)

    return filled_prompt



