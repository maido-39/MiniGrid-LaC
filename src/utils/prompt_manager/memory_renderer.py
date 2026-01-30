######################################################
#                                                    #
#                    MEMORY RENDERER                 #
#                                                    #
######################################################


"""
Renders memory_dict values (str, list, dict) to prompt-safe strings.

- str: returned as-is; empty string → "None" (or configurable).
- list: one bullet per item, e.g. "- a\n- b".
- dict: "key: value" per line; values recursively rendered with depth limit.
- Other types (int, bool, None): str(value) or "None" to avoid crashes.
"""




######################################################
#                                                    #
#                      LIBRARIES                     #
#                                                    #
######################################################


from typing import Any




######################################################
#                                                    #
#                      VARIABLES                     #
#                                                    #
######################################################


# Allowed key chars for $memory[key]: alphanumeric, underscore, hyphen
MEMORY_KEY_PATTERN = r"[a-zA-Z0-9_-]+"

# Max recursion depth for dict/list rendering (safety)
DEFAULT_MAX_DEPTH = 8




######################################################
#                                                    #
#                      FUNCTIONS                     #
#                                                    #
######################################################


def render_memory_value(
    value: Any,
    default_for_empty: str = "None",
    max_depth: int = DEFAULT_MAX_DEPTH,
    _depth: int = 0,
) -> str:
    """
    Render a single memory value to a string for prompt substitution.

    - str: as-is; empty → default_for_empty.
    - list: "- item\\n" per element (item rendered recursively).
    - dict: "key: value\\n" per pair (value rendered recursively).
    - int, float, bool, None: str(value) or default_for_empty for None.

    Args:
    value : Any
        The value to render (from memory_dict[key] or nested).
    default_for_empty : str, optional
        String to use for empty str or None. Default "None".
    max_depth : int, optional
        Maximum recursion depth for dict/list. Default 8.
    _depth : int, optional
        Internal recursion depth; do not pass.
    Returns:
        str
            Rendered string safe for prompt insertion.
    """
    
    if _depth > max_depth:
        return default_for_empty

    if value is None:
        return default_for_empty

    if isinstance(value, str):
        return value.strip() if value.strip() else default_for_empty

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, list):
        parts = []
        for item in value:
            part = render_memory_value(
                item,
                default_for_empty=default_for_empty,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            parts.append("- " + part.replace("\n", "\n  "))
        return "\n".join(parts) if parts else default_for_empty

    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            v_str = render_memory_value(
                v,
                default_for_empty=default_for_empty,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            # Indent continuation lines for nested list/dict
            if "\n" in v_str:
                v_str = v_str.replace("\n", "\n  ")
            parts.append(f"{k}: {v_str}")
        if not parts:
            return default_for_empty
        body = "\n".join(parts)
        # Leading newline + indent each line so "- Key: $memory[key]" renders as:
        # - Key:
        #   k1: v1
        #   k2: v2
        return "\n" + "\n".join("  " + line for line in body.split("\n"))

    # Fallback for unexpected types
    return str(value) if value is not None else default_for_empty



