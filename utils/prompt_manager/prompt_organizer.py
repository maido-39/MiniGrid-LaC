######################################################
#                                                    #
#                        PROMPT                      #
#                      ORGANIZER                     #
#                                                    #
######################################################


""""""




######################################################
#                                                    #
#                      LIBRARIES                     #
#                                                    #
######################################################


from utils.prompt_manager.prompt_interp import *
import utils.prompt_manager.terminal_formatting_utils as tfu

from utils.global_variables import DEFAULT_MISSION




######################################################
#                                                    #
#                       CLASS                        #
#                                                    #
######################################################


class PromptOrganizer:
    """Prompt Management Class (Absolute Coordinate Version)"""
    
    def __init__(self):
        self.grounding = ""
        self.previous_action = ""
        self.task_process = {"goal": "", "status": ""}  # status: pending | in_progress | completed | blocked
    
    def get_system_prompt(self, wrapper=None, last_action_result=None) -> str:
        """Generate Entire System Prompt (Absolute Coordinate Version)"""
        
        ## For handling prompt errors
        # Grounding Content (Always displayed; if empty, displays an empty string)
        grounding_content = self.grounding if self.grounding else ""
        
        # Previous Action (Always displayed, “None” if empty)
        previous_action = self.previous_action if self.previous_action else "None"
        
        # Task Process (Always display, default if empty)
        task_goal = self.task_process.get("goal", "") if self.task_process.get("goal") else "None"
        task_status = self.task_process.get("status", "") if self.task_process.get("status") else "None"
        task_process_str = f"Goal: {task_goal}, Status: {task_status}"
        
        # Last Action Result (Failure Information)
        if last_action_result and last_action_result.get("action"):
            action_result = last_action_result.get("action", "None")
            result_status = "success" if last_action_result.get("success", True) else "failed"
            failure_reason = last_action_result.get("failure_reason", "")
            position_changed = "yes" if last_action_result.get("position_changed", True) else "no"
            last_action_str = f"Action: {action_result}, Result: {result_status}"
            if not last_action_result.get("success", True):
                last_action_str += f", Failure Reason: {failure_reason}"
            last_action_str += f", Position Changed: {position_changed}"
        else:
            last_action_str = "None"
        
        ## Actual Application Prompt Start (Absolute Coordinate Version)
        return system_prompt_interp(file_name="system_prompt_start.txt",
                             grounding_content=grounding_content,
                             last_action_str=last_action_str,
                             previous_action=previous_action,
                             task_process_str=task_process_str)
    
    def get_feedback_system_prompt(self) -> str:
        """System Prompt for Feedback Generation (Absolute Coordinate Version)"""
        
        return system_prompt_interp("feedback_prompt.txt")
    
    def update_grounding(self, new_grounding: str):
        """Grounding Knowledge Accumulation Update"""
        
        if new_grounding and new_grounding.strip():
            if self.grounding:
                self.grounding = f"{self.grounding}\n\n{new_grounding.strip()}"
            else:
                self.grounding = new_grounding.strip()
    
    def get_user_prompt(self, default_prompt: str = None) -> str:
        """Receive user prompt input (mission) either from terminal input or from a .txt file"""
        
        # Default prompt determination (if no default_prompt is specified, use DEFAULT_MISSION)
        actual_default = default_prompt if default_prompt else DEFAULT_MISSION
        
        if default_prompt:
            tfu.cprint(f"Task Hint: {default_prompt}", tfu.LIGHT_BLUE)
        tfu.cprint("Enter your command:", tfu.LIGHT_GREEN)
        tfu.cprint("- Directly input mission", tfu.LIGHT_GREEN, italic=True)
        tfu.cprint("- or enter the .txt file path", tfu.LIGHT_GREEN, italic=True)
        tfu.cprint(f"(Enter: {actual_default})", tfu.LIGHT_GREEN, italic=True)
        
        user_input = input("> ").strip()
        
        return mission_input_interp(user_input, actual_default)



