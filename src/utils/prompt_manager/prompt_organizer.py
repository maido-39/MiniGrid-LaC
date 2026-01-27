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

from utils.miscellaneous.global_variables import DEFAULT_INITIAL_MISSION, DEFAULT_MISSION, USE_NEW_GROUNDING_SYSTEM, USE_VERBALIZED_ENTROPY




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
        # User prompt input cache (for reusing previous input with Enter key)
        self.user_prompt_cache = None
    
    def get_system_prompt(self, wrapper=None, last_action_result=None, grounding_file_path=None) -> str:
        """Generate Entire System Prompt (Absolute Coordinate Version)"""
        
        ## For handling prompt errors
        # Grounding Content (Always displayed; if empty, displays an empty string)
        if USE_NEW_GROUNDING_SYSTEM:
            # 새 Grounding 시스템 사용 시: grounding_file_path에서 파일 내용 읽기
            grounding_content = ""
            if grounding_file_path:
                try:
                    from pathlib import Path
                    # 여러 파일 지원: 리스트 또는 쉼표로 구분된 문자열
                    if isinstance(grounding_file_path, list):
                        file_paths = [Path(f) for f in grounding_file_path]
                    elif isinstance(grounding_file_path, str) and ',' in grounding_file_path:
                        file_paths = [Path(f.strip()) for f in grounding_file_path.split(',')]
                    else:
                        file_paths = [Path(grounding_file_path)]
                    
                    # 모든 파일 읽기 및 병합
                    grounding_contents = []
                    for grounding_path in file_paths:
                        if grounding_path.exists():
                            content = grounding_path.read_text(encoding='utf-8')
                            grounding_contents.append(content)
                    
                    if grounding_contents:
                        grounding_content = "\n\n---\n\n".join(grounding_contents)
                except Exception as e:
                    # 파일 읽기 실패 시 빈 문자열 사용
                    grounding_content = ""
        else:
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
                                    strict=True,
                                    last_action_str=last_action_str,
                                    grounding_content=grounding_content,
                                    task_process_str=task_process_str,
                                    previous_action=previous_action
                                   )
    
    def get_verbalized_entropy_system_prompt(self, wrapper=None, last_action_result=None, grounding_file_path=None) -> str:
        """
        Generate System Prompt for Verbalized Entropy mode
        Uses step1/step2/step3 probability distributions instead of action array
        """
        
        # Grounding Content
        if USE_NEW_GROUNDING_SYSTEM:
            # 새 Grounding 시스템 사용 시: grounding_file_path에서 파일 내용 읽기
            grounding_content = ""
            if grounding_file_path:
                try:
                    from pathlib import Path
                    # 여러 파일 지원: 리스트 또는 쉼표로 구분된 문자열
                    if isinstance(grounding_file_path, list):
                        file_paths = [Path(f) for f in grounding_file_path]
                    elif isinstance(grounding_file_path, str) and ',' in grounding_file_path:
                        file_paths = [Path(f.strip()) for f in grounding_file_path.split(',')]
                    else:
                        file_paths = [Path(grounding_file_path)]
                    
                    # 모든 파일 읽기 및 병합
                    grounding_contents = []
                    for grounding_path in file_paths:
                        if grounding_path.exists():
                            content = grounding_path.read_text(encoding='utf-8')
                            grounding_contents.append(content)
                    
                    if grounding_contents:
                        grounding_content = "\n\n---\n\n".join(grounding_contents)
                except Exception as e:
                    # 파일 읽기 실패 시 빈 문자열 사용
                    grounding_content = ""
        else:
            grounding_content = self.grounding if self.grounding else ""
        
        # Previous Action
        previous_action = self.previous_action if self.previous_action else "None"
        
        # Task Process
        task_goal = self.task_process.get("goal", "") if self.task_process.get("goal") else "None"
        task_status = self.task_process.get("status", "") if self.task_process.get("status") else "None"
        task_process_str = f"Goal: {task_goal}, Status: {task_status}"
        
        return system_prompt_interp(file_name="system_prompt_verbalized_entropy.txt",
                                    strict=True,
                                    grounding_content=grounding_content,
                                    task_process_str=task_process_str,
                                    previous_action=previous_action
                                   )
    
    def get_system_prompt_by_mode(self, wrapper=None, last_action_result=None, use_verbalized: bool = None, grounding_file_path=None) -> str:
        """
        Get system prompt based on mode (verbalized entropy or standard)
        
        Args:
            wrapper: Environment wrapper
            last_action_result: Last action result dictionary
            use_verbalized: Override for USE_VERBALIZED_ENTROPY (None = use global setting)
            grounding_file_path: Path to grounding file(s) (for USE_NEW_GROUNDING_SYSTEM=True)
        
        Returns:
            System prompt string
        """
        if use_verbalized is None:
            use_verbalized = USE_VERBALIZED_ENTROPY
        
        if use_verbalized:
            return self.get_verbalized_entropy_system_prompt(wrapper, last_action_result, grounding_file_path)
        else:
            return self.get_system_prompt(wrapper, last_action_result, grounding_file_path)
    
    def get_feedback_system_prompt(self) -> str:
        """
        System Prompt for Feedback Generation (Absolute Coordinate Version)
        """
        
        return system_prompt_interp(file_name="feedback_prompt.txt",
                                    strict=True
                                   )
    
    def update_grounding(self, new_grounding: str):
        """
        Grounding Knowledge Accumulation Update
        """
        
        if new_grounding and new_grounding.strip():
            if self.grounding:
                self.grounding = f"{self.grounding}\n\n{new_grounding.strip()}"
            else:
                self.grounding = new_grounding.strip()
    
    def get_user_prompt(self, default_prompt: str = None, init_step: bool = False) -> str:
        """
        Receive user prompt input (mission) either from terminal input or from a .txt file
        Returns tuple: (processed_prompt, raw_input) if called with return_raw=True
        """
        
        # Default prompt determination (if no default_prompt is specified, use DEFAULT_MISSION)
        if init_step:
            actual_default = default_prompt if default_prompt else DEFAULT_INITIAL_MISSION
        else:
            actual_default = default_prompt if default_prompt else DEFAULT_MISSION
        
        tfu.cprint("\n" + "=" * 80, bold=True)
        
        #if default_prompt:
        #    tfu.cprint(f"\nTask Hint: {default_prompt}", tfu.LIGHT_BLUE)
        
        tfu.cprint("\nEnter your command:", tfu.LIGHT_WHITE)
        tfu.cprint("- Directly input mission", tfu.LIGHT_WHITE, italic=True)
        tfu.cprint("- or enter the .txt file path", tfu.LIGHT_WHITE, italic=True)
        tfu.cprint(f"(Example >> {actual_default})\n", tfu.LIGHT_BLACK, italic=True)
        
        # Show cached value if available
        if self.user_prompt_cache:
            tfu.cprint(f"Previous input (press Enter to reuse): {self.user_prompt_cache}", tfu.LIGHT_BLACK, italic=True)
        
        user_input = input("> ").strip()
        
        # Empty input and cache exists: use cache
        if not user_input and self.user_prompt_cache:
            user_input = self.user_prompt_cache
            tfu.cprint(f"Using cached input: {user_input}", tfu.LIGHT_BLACK, italic=True)
        elif user_input:
            # New input: update cache
            self.user_prompt_cache = user_input
        
        # Store raw input for feedback detection
        self._raw_user_input = user_input
        
        return mission_input_interp(user_input, actual_default)



