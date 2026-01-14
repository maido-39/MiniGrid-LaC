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




######################################################
#                                                    #
#                      FUNCTION                      #
#                                                    #
######################################################


def prompt_input_interp(user_input, actual_default):
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