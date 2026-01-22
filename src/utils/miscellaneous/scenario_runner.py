######################################################
#                                                    #
#                      SCENARIO                      #
#                       RUNNER                       #
#                                                    #
######################################################


""""""




######################################################
#                                                    #
#                      LIBRARIES                     #
#                                                    #
######################################################


import csv
import cv2
import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional  # Union is used in visualize_grid_cli
from datetime import datetime

from utils.miscellaneous.visualizer import Visualizer
from utils.vlm.vlm_processor import VLMProcessor
from utils.user_manager.user_interact import UserInteraction
import utils.prompt_manager.terminal_formatting_utils as tfu
from utils.prompt_manager.prompt_organizer import PromptOrganizer
from utils.prompt_manager.prompt_interp import system_prompt_interp
from utils.map_manager.emoji_map_loader import load_emoji_map_from_json

from utils.miscellaneous.global_variables import (
    DEFAULT_INITIAL_MISSION,
    DEFAULT_MISSION,
    VLM_MODEL,
    LOGPROBS_ENABLED,
    LOGPROBS_TOPK,
    MAP_FILE_NAME,
    DEBUG,
)




######################################################
#                                                    #
#                        CLASS                       #
#                                                    #
######################################################


class ScenarioExperiment:
    """Scenario 2 Experiment Main Class (Runner) - Absolute Coordinate Version"""
    
    def __init__(self,
                 log_dir: Path = None,
                 json_map_path: str = None,
                 prompt_organizer: Optional[PromptOrganizer] = None,
                 use_logprobs: bool = None,
                 debug: bool = None
                ):
        """
        Args:
            log_dir: Log directory path
            json_map_path: JSON map file path (default: None, uses MAP_FILE_NAME from global_variables)
            prompt_organizer: Custom PromptOrganizer instance (default: None, uses default PromptOrganizer)
            use_logprobs: Enable logprobs (default: None, uses LOGPROBS_ENABLED from global_variables)
            debug: Enable debug output (default: None, uses DEBUG from global_variables)
        """
        
        self.wrapper = None
        # Use global MAP_FILE_NAME if json_map_path is not provided
        if json_map_path is None:
            json_map_path = f"config/{MAP_FILE_NAME}"
        self.json_map_path = json_map_path
        self.prompt_organizer = prompt_organizer if prompt_organizer is not None else PromptOrganizer()
        # logprobs / VLM 설정
        self.vlm_model = VLM_MODEL
        # Use use_logprobs parameter if provided, otherwise use global LOGPROBS_ENABLED
        self.logprobs_enabled_cfg = use_logprobs if use_logprobs is not None else LOGPROBS_ENABLED
        self.logprobs_topk = LOGPROBS_TOPK
        # Use debug parameter if provided, otherwise use global DEBUG
        self.debug = debug if debug is not None else DEBUG
        self.logprobs_active = False  # 실제 활성 여부 (모델/설정에 따라 결정)
        self.logprobs_metadata = {}
        self.action_logprobs_info = {}
        # VLMProcessor 생성 (모델/설정에 따라 logprobs 활성화 여부 결정)
        self.vlm_processor = self._create_vlm_processor()
        self.visualizer = Visualizer()
        self.user_interaction = UserInteraction()
        
        if log_dir is None:
            map_name = Path(json_map_path).stem
            log_dir = Path("logs") / f"scenario2_absolute_{map_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.step = 0
        self.done = False
        self.state = None
        self.image = None
        self.user_prompt = ""
        self.vlm_response_raw = ""
        self.vlm_response_parsed = {}
        self.action_index = 0  # Default value: move up
        self.action_name = "move up"
        self.reward = 0.0
        
        # Last action result tracking
        self.last_action_result = {
            "action": "",
            "success": True,
            "failure_reason": "",
            "position_changed": True
        }
        self.previous_position = None
        
        self.csv_file = None
        self.csv_writer = None
        self._init_csv_logging()
    
    def _create_vlm_processor(self) -> VLMProcessor:
        """
        모델/설정에 따라 logprobs를 활성화한 VLMProcessor 생성.
        logprobs는 Vertex AI Gemini(-vertex/-logprobs) 모델에서만 사용.
        """
        model_lower = (self.vlm_model or "").lower()
        logprobs_allowed = (
            model_lower.startswith("gemini")
            and ("-vertex" in model_lower or "-logprobs" in model_lower)
        )
        logprobs_value = self.logprobs_topk if (self.logprobs_enabled_cfg and logprobs_allowed) else None
        self.logprobs_active = logprobs_value is not None

        if self.logprobs_enabled_cfg and not logprobs_allowed:
            tfu.cprint(
                "[Info] logprobs 비활성화: 모델이 Vertex AI Gemini(-vertex/-logprobs) 형태가 아닙니다.",
                tfu.LIGHT_BLACK,
            )

        return VLMProcessor(
            model=self.vlm_model,
            logprobs=logprobs_value,
            debug=self.debug,
        )

    def _evaluate_feedback(self, user_prompt: str) -> bool:
        """
        Feedback Evaluation (Internal Method)
        """
        
        if not user_prompt or not isinstance(user_prompt, str):
            return False
        
        ## Change this one to the feedback keywords you want to use (list of strings)
        feedback_keywords = [
            "wrong",
            "feedback :"
        ]
        
        user_lower = user_prompt.lower()
        for keyword in feedback_keywords:
            if keyword in user_lower:
                return True
        
        return False
    
    def _format_carrying_object(self, carrying_obj) -> str:
        """
        Format carrying object information for terminal display.
        Uses JSON map file to get the correct emoji character for emoji objects.
        Supports both single object and list of objects.
        
        Args:
            carrying_obj: The object(s) being carried by the agent (can be single object or list)
            
        Returns:
            Formatted string describing the carrying object(s)
        """
        if carrying_obj is None:
            return "None"
        
        # Handle list of objects
        if isinstance(carrying_obj, list):
            if len(carrying_obj) == 0:
                return "None"
            # Format all objects in the list
            formatted_objects = []
            for obj in carrying_obj:
                formatted_objects.append(self._format_single_carrying_object(obj))
            return f"[{', '.join(formatted_objects)}]"
        
        # Handle single object
        return self._format_single_carrying_object(carrying_obj)
    
    def _format_single_carrying_object(self, carrying_obj) -> str:
        """
        Format a single carrying object information for terminal display.
        
        Args:
            carrying_obj: A single object being carried by the agent
            
        Returns:
            Formatted string describing the carrying object
        """
        if carrying_obj is None:
            return "None"
        
        # Try to get emoji character from JSON map file
        emoji_char = None
        if hasattr(self, 'json_map_path') and self.json_map_path:
            try:
                import json
                with open(self.json_map_path, 'r', encoding='utf-8') as f:
                    map_data = json.load(f)
                    if 'map' in map_data and 'emoji_objects' in map_data['map']:
                        emoji_objects = map_data['map']['emoji_objects']
                        # Search for emoji_name in emoji_objects
                        for emoji_key, emoji_def in emoji_objects.items():
                            if emoji_def.get('emoji_name') == getattr(carrying_obj, 'emoji_name', None):
                                emoji_char = emoji_key
                                break
            except Exception:
                # If JSON loading fails, fall back to default mapping
                pass
        
        # Fallback emoji mapping for common objects (if JSON lookup failed)
        if emoji_char is None:
            emoji_map = {
                'box': '📦',
                'apple': '🍎',
                'key': '🔑',
                'ball': '⚽',
                'chair': '🪑',
                'tree': '🌲',
                'mushroom': '🍄',
                'flower': '🌼',
                'cat': '🐈',
                'grass': '🌾',
                'rock': '🗿',
                'desktop': '🖥️',
                'workstation': '📱',
                'brick': '🧱'
            }
            if hasattr(carrying_obj, 'emoji_name'):
                emoji_char = emoji_map.get(carrying_obj.emoji_name, '❓')
            else:
                emoji_char = '❓'
        
        if hasattr(carrying_obj, 'type'):
            obj_type = carrying_obj.type
            
            # Handle emoji objects
            if obj_type == 'emoji' and hasattr(carrying_obj, 'emoji_name'):
                emoji_name = carrying_obj.emoji_name
                color = getattr(carrying_obj, 'color', 'N/A')
                return f"{emoji_char} {emoji_name} (color: {color})"
            
            # Handle standard MiniGrid objects
            elif obj_type == 'key':
                color = getattr(carrying_obj, 'color', 'N/A')
                return f"🔑 Key (color: {color})"
            elif obj_type == 'ball':
                color = getattr(carrying_obj, 'color', 'N/A')
                return f"⚽ Ball (color: {color})"
            elif obj_type == 'box':
                color = getattr(carrying_obj, 'color', 'N/A')
                return f"📦 Box (color: {color})"
            else:
                # Generic object with color if available
                color = getattr(carrying_obj, 'color', None)
                if color:
                    return f"{obj_type} (color: {color})"
                else:
                    return f"{obj_type}"
        else:
            # Fallback: just return string representation
            return str(carrying_obj)
    
    def vlm_gen_action(self,
                       image: np.ndarray,
                       system_prompt: str,
                       user_prompt: str,
                       use_logprobs: bool = None
                      ) -> dict:
        """
        VLM call for Action creation.
        use_logprobs=True 이고 logprobs가 활성 상태일 때만 logprobs 호출.
        """
        use_lp = self.logprobs_active if use_logprobs is None else (use_logprobs and self.logprobs_active)

        tfu.cprint("\n[3] Sending Action creation request to VLM...")

        if use_lp and self.vlm_processor.logprobs:
            try:
                raw_response, logprobs_metadata = self.vlm_processor.requester_with_logprobs(
                    image=image,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    debug=self.debug
                )

                if not raw_response:
                    tfu.cprint("The VLM response is empty.")
                    return {}

                tfu.cprint("VLM response received")
                tfu.cprint("[4] Parsing response with logprobs...")
                parsed = self.vlm_processor.parser_action_with_logprobs(
                    raw_response,
                    logprobs_metadata,
                    action_field="action",
                    remove_logprobs=False
                )

                # logprobs 정보 저장 (로깅용)
                self.logprobs_metadata = logprobs_metadata
                self.action_logprobs_info = parsed.get('action_logprobs_info', {})
                
                # Print logprobs info if debug is enabled
                if self.debug and self.action_logprobs_info:
                    self.vlm_processor.postprocessor_action.print_action_logprobs_info(self.action_logprobs_info)
                
                return parsed
            except Exception as e:
                tfu.cprint(f"[Warning] logprobs 호출 실패, 일반 모드로 재시도: {e}", tfu.LIGHT_RED)
                # fallthrough to non-logprobs

        # 일반 모드
        raw_response = self.vlm_processor.requester(
            image=image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            debug=self.debug
        )

        if not raw_response:
            tfu.cprint("The VLM response is empty.")
            return {}

        tfu.cprint("VLM response received")
        tfu.cprint("[4] Parsing response...")
        parsed = self.vlm_processor.parser_action(raw_response)
        # 일반 모드에서는 logprobs 정보 초기화
        self.logprobs_metadata = {}
        self.action_logprobs_info = {}
        return parsed
    
    def vlm_gen_feedback(self, system_prompt: str, user_feedback: str) -> str:
        """
        VLM call for feedback generation
        """
        
        tfu.cprint("\n[3-F] Sending Feedback Analysis Request to VLM...")
        
        feedback_system_prompt = self.prompt_organizer.get_feedback_system_prompt()
        
        file_name = "feedback_user_prompt.txt"
        
        feedback_user_prompt = system_prompt_interp(file_name,
                                                    strict=True,
                                                    system_prompt=system_prompt,
                                                    user_feedback=user_feedback
                                                   )
        
        raw_response = self.vlm_processor.requester(
            image=None,
            system_prompt=feedback_system_prompt,
            user_prompt=feedback_user_prompt,
            debug=self.debug
        )
        
        if not raw_response:
            tfu.cprint("[Warning] Feedback VLM response is empty!", tfu.LIGHT_RED)
            return ""
        
        tfu.cprint("Feedback VLM Response Received", tfu.LIGHT_GREEN, indent=8)
        tfu.cprint("\n[4-F] Parsing Feedback Response...\n", tfu.LIGHT_BLACK)
        parsed = self.vlm_processor.parser_feedback(raw_response)
        knowledge = parsed.get('knowledge', '')
        
        if knowledge:
            tfu.cprint(f"\n[4-F-1] Generated Knowledge: {knowledge}", tfu.LIGHT_BLACK)
            self.prompt_organizer.update_grounding(knowledge)
            tfu.cprint("\n[4-F-2] Grounding update complete", tfu.LIGHT_GREEN)
            tfu.cprint("\n" + "=" * 80 + "\n", bold=True)
            tfu.cprint("Updated Grounding Information:")
            tfu.cprint("-" * 80)
            tfu.cprint(knowledge + "\n", tfu.LIGHT_BLUE)
            tfu.cprint("\nComplete Grounding Information:", tfu.BLUE, True)
            tfu.cprint("-" * 80)
            if self.prompt_organizer.grounding:
                tfu.cprint(self.prompt_organizer.grounding, tfu.BLUE, True)
            else:
                tfu.cprint("(None)")
            tfu.cprint("\n" + "=" * 80, bold=True)
        
        return knowledge
    
    def _get_system_prompt_without_grounding(self, wrapper=None, last_action_result=None) -> str:
        """
        Generate system prompt without grounding content.
        Temporarily sets grounding to empty string, then restores original value.
        
        Args:
            wrapper: Environment wrapper (optional)
            last_action_result: Last action result dict (optional)
        
        Returns:
            System prompt string without grounding content
        """
        # Save original grounding
        original_grounding = self.prompt_organizer.grounding
        
        # Temporarily set grounding to empty
        self.prompt_organizer.grounding = ""
        
        try:
            # Get system prompt (will have empty grounding_content)
            system_prompt = self.prompt_organizer.get_system_prompt(wrapper, last_action_result)
        finally:
            # Restore original grounding
            self.prompt_organizer.grounding = original_grounding
        
        return system_prompt
    
    def _calculate_entropy_from_logprobs(self, action_logprobs_info: dict) -> Optional[float]:
        """
        Calculate entropy from action_logprobs_info.
        Uses the first action's entropy value.
        
        Args:
            action_logprobs_info: Dictionary from get_action_logprobs() containing:
                - 'action_entropies': List[float] - List of entropies for each action
        
        Returns:
            float: First action's entropy, or None if not available
        """
        if not action_logprobs_info:
            return None
        
        action_entropies = action_logprobs_info.get('action_entropies', [])
        if not action_entropies or len(action_entropies) == 0:
            return None
        
        # Use first action's entropy
        first_entropy = action_entropies[0]
        return first_entropy if first_entropy is not None else None
    
    def vlm_gen_action_H_X(self,
                           image: np.ndarray,
                           system_prompt: str,
                           user_prompt: str
                          ) -> dict:
        """
        VLM call for H(X) calculation - without Language Instruction and Grounding.
        
        This method calls VLM with:
        - system_prompt: Grounding removed (empty string)
        - user_prompt: Empty string
        
        Args:
            image: Input image array
            system_prompt: Original system prompt (grounding will be removed)
            user_prompt: Original user prompt (will be set to empty string)
        
        Returns:
            Dictionary containing:
                - 'parsed': Parsed VLM response dict
                - 'logprobs_metadata': Logprobs metadata dict (if available)
                - 'action_logprobs_info': Action logprobs info dict (if available)
        """
        # Get system prompt without grounding
        system_prompt_no_grounding = self._get_system_prompt_without_grounding(
            self.wrapper, self.last_action_result
        )
        
        # Set user_prompt to empty string
        user_prompt_empty = ""
        
        tfu.cprint("\n[H(X)] Sending VLM request (no Language Instruction, no Grounding)...")
        
        if self.logprobs_active and self.vlm_processor.logprobs:
            try:
                raw_response, logprobs_metadata = self.vlm_processor.requester_with_logprobs(
                    image=image,
                    system_prompt=system_prompt_no_grounding,
                    user_prompt=user_prompt_empty,
                    debug=self.debug
                )
                
                if not raw_response:
                    tfu.cprint("[H(X)] VLM response is empty.", tfu.LIGHT_RED)
                    return {
                        'parsed': {},
                        'logprobs_metadata': {},
                        'action_logprobs_info': {}
                    }
                
                tfu.cprint("[H(X)] VLM response received")
                parsed = self.vlm_processor.parser_action_with_logprobs(
                    raw_response,
                    logprobs_metadata,
                    action_field="action",
                    remove_logprobs=False
                )
                
                action_logprobs_info = parsed.get('action_logprobs_info', {})
                
                # Debug output
                if self.debug:
                    action_chunk = parsed.get('action', [])
                    tfu.cprint(f"[H(X)] Action: {action_chunk}", tfu.LIGHT_CYAN)
                    if action_logprobs_info:
                        self.vlm_processor.postprocessor_action.print_action_logprobs_info(action_logprobs_info)
                
                return {
                    'parsed': parsed,
                    'logprobs_metadata': logprobs_metadata,
                    'action_logprobs_info': action_logprobs_info
                }
            except Exception as e:
                tfu.cprint(f"[H(X)] Warning: logprobs call failed: {e}", tfu.LIGHT_RED)
                return {
                    'parsed': {},
                    'logprobs_metadata': {},
                    'action_logprobs_info': {}
                }
        else:
            tfu.cprint("[H(X)] Warning: logprobs not active, cannot calculate entropy", tfu.LIGHT_RED)
            return {
                'parsed': {},
                'logprobs_metadata': {},
                'action_logprobs_info': {}
            }
    
    def vlm_gen_action_H_X_given_S(self,
                                    image: np.ndarray,
                                    system_prompt: str,
                                    user_prompt: str
                                   ) -> dict:
        """
        VLM call for H(X|S) calculation - without Language Instruction (Grounding included).
        
        This method calls VLM with:
        - system_prompt: Original system prompt (with grounding)
        - user_prompt: Empty string
        
        Args:
            image: Input image array
            system_prompt: System prompt with grounding
            user_prompt: Original user prompt (will be set to empty string)
        
        Returns:
            Dictionary containing:
                - 'parsed': Parsed VLM response dict
                - 'logprobs_metadata': Logprobs metadata dict (if available)
                - 'action_logprobs_info': Action logprobs info dict (if available)
        """
        # Use original system_prompt (with grounding)
        # Set user_prompt to empty string
        user_prompt_empty = ""
        
        tfu.cprint("\n[H(X|S)] Sending VLM request (no Language Instruction, with Grounding)...")
        
        if self.logprobs_active and self.vlm_processor.logprobs:
            try:
                raw_response, logprobs_metadata = self.vlm_processor.requester_with_logprobs(
                    image=image,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt_empty,
                    debug=self.debug
                )
                
                if not raw_response:
                    tfu.cprint("[H(X|S)] VLM response is empty.", tfu.LIGHT_RED)
                    return {
                        'parsed': {},
                        'logprobs_metadata': {},
                        'action_logprobs_info': {}
                    }
                
                tfu.cprint("[H(X|S)] VLM response received")
                parsed = self.vlm_processor.parser_action_with_logprobs(
                    raw_response,
                    logprobs_metadata,
                    action_field="action",
                    remove_logprobs=False
                )
                
                action_logprobs_info = parsed.get('action_logprobs_info', {})
                
                # Debug output
                if self.debug:
                    action_chunk = parsed.get('action', [])
                    tfu.cprint(f"[H(X|S)] Action: {action_chunk}", tfu.LIGHT_CYAN)
                    if action_logprobs_info:
                        self.vlm_processor.postprocessor_action.print_action_logprobs_info(action_logprobs_info)
                
                return {
                    'parsed': parsed,
                    'logprobs_metadata': logprobs_metadata,
                    'action_logprobs_info': action_logprobs_info
                }
            except Exception as e:
                tfu.cprint(f"[H(X|S)] Warning: logprobs call failed: {e}", tfu.LIGHT_RED)
                return {
                    'parsed': {},
                    'logprobs_metadata': {},
                    'action_logprobs_info': {}
                }
        else:
            tfu.cprint("[H(X|S)] Warning: logprobs not active, cannot calculate entropy", tfu.LIGHT_RED)
            return {
                'parsed': {},
                'logprobs_metadata': {},
                'action_logprobs_info': {}
            }
    
    def vlm_gen_action_H_X_given_LS(self,
                                     image: np.ndarray,
                                     system_prompt: str,
                                     user_prompt: str
                                    ) -> dict:
        """
        VLM call for H(X|L,S) calculation - with both Language Instruction and Grounding.
        
        This method wraps the existing vlm_gen_action() method to provide consistent
        return format with logprobs information.
        
        Args:
            image: Input image array
            system_prompt: System prompt with grounding
            user_prompt: User prompt with language instruction
        
        Returns:
            Dictionary containing:
                - 'parsed': Parsed VLM response dict
                - 'logprobs_metadata': Logprobs metadata dict (if available)
                - 'action_logprobs_info': Action logprobs info dict (if available)
        """
        tfu.cprint("\n[H(X|L,S)] Sending VLM request (with Language Instruction and Grounding)...")
        
        # Use existing vlm_gen_action method
        parsed = self.vlm_gen_action(
            image=image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            use_logprobs=self.logprobs_active
        )
        
        # Extract logprobs information if available
        logprobs_metadata = getattr(self, 'logprobs_metadata', {})
        action_logprobs_info = getattr(self, 'action_logprobs_info', {})
        
        # Debug output
        if self.debug:
            action_chunk = parsed.get('action', [])
            tfu.cprint(f"[H(X|L,S)] Action: {action_chunk}", tfu.LIGHT_CYAN)
            if action_logprobs_info:
                self.vlm_processor.postprocessor_action.print_action_logprobs_info(action_logprobs_info)
        
        return {
            'parsed': parsed,
            'logprobs_metadata': logprobs_metadata,
            'action_logprobs_info': action_logprobs_info
        }
    
    def _init_csv_logging(self):
        """
        CSV Logging Initialization
        """
        
        csv_path = self.log_dir / "experiment_log.csv"
        file_exists = csv_path.exists()
        
        self.csv_file = open(csv_path, 'a', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        
        if not file_exists:
            self.csv_writer.writerow([
                "step", "timestamp", "agent_x", "agent_y", "agent_dir",
                "action_index", "action_name", "user_prompt",
                "vlm_action_chunk", "vlm_reasoning", "vlm_grounding",
                "memory_spatial_description", "memory_task_goal", "memory_task_status", "memory_task_blocked_reason", "memory_previous_action",
                "last_action_result_action", "last_action_result_success", "last_action_result_failure_reason", "last_action_result_position_changed",
                "reward", "done", "image_path", "vlm_action_logprobs_info",
                "carrying_object", "is_pickup", "is_drop"
            ])
    
    def _log_step(self):
        """
        Current step logging
        """
        
        timestamp = datetime.now().isoformat()
        
        agent_pos = self.state['agent_pos']
        if isinstance(agent_pos, np.ndarray):
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        else:
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        
        image_path = f"step_{self.step:04d}.png"
        
        # Memory Parsing
        memory = self.vlm_response_parsed.get('memory', {})
        if isinstance(memory, str):
            try:
                memory = json.loads(memory)
            except Exception:
                memory = {"spatial_description": "", "task_process": {"goal": "", "status": ""}, "previous_action": ""}
        elif not isinstance(memory, dict):
            memory = {"spatial_description": "", "task_process": {"goal": "", "status": ""}, "previous_action": ""}
        
        # task_process parsing
        task_process = memory.get('task_process', {})
        if not isinstance(task_process, dict):
            task_process = {"goal": "", "status": "", "blocked_reason": ""}
        
        action_chunk = self.vlm_response_parsed.get('action', [])
        if isinstance(action_chunk, str):
            try:
                action_chunk = json.loads(action_chunk)
            except Exception:
                action_chunk = [action_chunk] if action_chunk else []
        if not isinstance(action_chunk, list):
            action_chunk = [str(action_chunk)]
        
        # Get last_action_result
        last_action_result = self.last_action_result if hasattr(self, 'last_action_result') else {
            "action": "",
            "success": True,
            "failure_reason": "",
            "position_changed": True
        }
        
        # Get carrying object information
        carrying_object = "None"
        env = self.wrapper.env
        if hasattr(env, 'carrying'):
            if isinstance(env.carrying, list):
                if len(env.carrying) > 0:
                    carrying_object = self._format_carrying_object(env.carrying)
            elif env.carrying is not None:
                carrying_object = self._format_carrying_object(env.carrying)
        
        # Check if action is pickup or drop
        is_pickup = (self.action_name.lower() in ['pickup', 'pick up'] or self.action_index == 4)
        is_drop = (self.action_name.lower() in ['drop'] or self.action_index == 5)
        
        self.csv_writer.writerow([
            self.step,
            timestamp,
            agent_x,
            agent_y,
            int(self.state['agent_dir']),
            self.action_index,
            self.action_name,
            self.user_prompt,
            json.dumps(action_chunk, ensure_ascii=False),
            self.vlm_response_parsed.get('reasoning', ''),
            self.vlm_response_parsed.get('grounding', ''),
            memory.get('spatial_description', ''),
            task_process.get('goal', ''),
            task_process.get('status', ''),
            task_process.get('blocked_reason', ''),
            memory.get('previous_action', ''),
            last_action_result.get('action', ''),
            bool(last_action_result.get('success', True)),
            last_action_result.get('failure_reason', ''),
            bool(last_action_result.get('position_changed', True)),
            float(self.reward),
            bool(self.done),
            image_path,
            json.dumps(self.action_logprobs_info, ensure_ascii=False),
            carrying_object,
            bool(is_pickup),
            bool(is_drop)
        ])
        self.csv_file.flush()
        
        json_path = self.log_dir / "experiment_log.json"
        
        # Get carrying object information for JSON log
        carrying_object_json = None
        env = self.wrapper.env
        if hasattr(env, 'carrying'):
            if isinstance(env.carrying, list):
                if len(env.carrying) > 0:
                    carrying_object_json = self._format_carrying_object(env.carrying)
            elif env.carrying is not None:
                carrying_object_json = self._format_carrying_object(env.carrying)
        
        # Check if action is pickup or drop for JSON log
        is_pickup_json = (self.action_name.lower() in ['pickup', 'pick up'] or self.action_index == 4)
        is_drop_json = (self.action_name.lower() in ['drop'] or self.action_index == 5)
        
        json_data = {
            "step": self.step,
            "timestamp": timestamp,
            "state": {
                "agent_pos": [agent_x, agent_y],
                "agent_dir": int(self.state['agent_dir']),
                "mission": str(self.state.get('mission', ''))
            },
            "action": {
                "index": self.action_index,
                "name": self.action_name
            },
            "user_prompt": self.user_prompt,
            "vlm_response": self.vlm_response_parsed,
            "memory": memory,
            "grounding": self.prompt_organizer.grounding,
            "last_action_result": last_action_result,
            "reward": float(self.reward),
            "done": bool(self.done),
            "image_path": image_path,
            "action_logprobs_info": self.action_logprobs_info,
            "carrying_object": carrying_object_json,
            "is_pickup": bool(is_pickup_json),
            "is_drop": bool(is_drop_json)
        }
        
        all_data = []
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    all_data = json.load(f)
                    if not isinstance(all_data, list):
                        all_data = [all_data]
                except json.JSONDecodeError:
                    all_data = []
        
        all_data.append(json_data)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        image_path_full = self.log_dir / image_path
        img_pil = Image.fromarray(self.image)
        img_pil.save(image_path_full)
    
    def initialize(self):
        """
        Reset experiment
        """
        
        tfu.cprint("\n\n" + "=" * 80 + "\n", bold=True)
        tfu.cprint("Scenario 2: VLM Control Experiment (Absolute Coordinate Movement Version)", bold=True)
        tfu.cprint("\n" + "=" * 80 + "\n", bold=True)
        tfu.cprint(f"\nMission: {DEFAULT_MISSION}", tfu.LIGHT_BLACK, italic=True)
        tfu.cprint("\nAction Space: Direct movement possible up/down/left/right (absolute coordinates)", tfu.LIGHT_BLACK, italic=True)
        tfu.cprint(f"\nLog directory: {self.log_dir}", tfu.LIGHT_BLACK, italic=True)
        
        tfu.cprint("\n[1] Creating environment...")
        tfu.cprint(f"Map file: {self.json_map_path}", tfu.LIGHT_BLACK, italic=True, indent=4)
        self.wrapper = load_emoji_map_from_json(self.json_map_path)
        self.wrapper.reset()
        
        self.state = self.wrapper.get_state()
        tfu.cprint(f"Agent Start Position: {self.state['agent_pos']}", tfu.LIGHT_BLACK, italic=True, indent=4)
        tfu.cprint(f"Agent Direction: {self.state['agent_dir']}", tfu.LIGHT_BLACK, italic=True, indent=4)
        
        # Save initial position
        initial_pos = tuple(self.state['agent_pos'])
        if isinstance(initial_pos, np.ndarray):
            initial_pos = (int(initial_pos[0]), int(initial_pos[1]))
        self.previous_position = initial_pos
        
        # Initial last_action_result setting
        self.last_action_result = {
            "action": "",
            "success": True,
            "failure_reason": "",
            "position_changed": True
        }
        
        # Action Spatial Information Output
        action_space = self.wrapper.get_absolute_action_space()
        tfu.cprint(f"\nAbsolute Direction Action Space:", tfu.LIGHT_BLACK, italic=True, indent=4)
        tfu.cprint(f"  - Available actions: {action_space['actions']}", tfu.LIGHT_BLACK, italic=True, indent=4)
        
        tfu.cprint("\n[2] VLM initialization complete!", tfu.LIGHT_GREEN, True)
        tfu.cprint("\nExperiment Start...\n", tfu.LIGHT_BLACK, italic=True)
    
    def run_step(self, init_step: bool = False):
        """
        Execute one step
        """
        
        self.step += 1
        tfu.cprint("\n" + "=" * 80 + "\n", bold=True)
        tfu.cprint(f"STEP {self.step}", bold=True, indent=8)
        tfu.cprint("\n" + "=" * 80 + "\n", bold=True)
        
        self.image = self.wrapper.get_image()
        self.state = self.wrapper.get_state()
        heading = self.wrapper.get_heading()
        heading_desc = self.wrapper.get_heading_description()
        tfu.cprint(f"Location: {self.state['agent_pos']}, Direction: {self.state['agent_dir']} ({heading})")
        tfu.cprint(f"Current Heading: {heading_desc}")
        
        self.visualizer.visualize_grid_cli(self.wrapper, self.state)
        self.visualizer.display_image(self.image)
        
        if init_step:
            default_prompt = f"{DEFAULT_INITIAL_MISSION}"
        else:
            default_prompt = f"{DEFAULT_MISSION}"
        
        # Use PromptOrganizer (supports file paths, templates, etc.)
        self.user_prompt = self.prompt_organizer.get_user_prompt(default_prompt, init_step=init_step)
        
        tfu.cprint("\n" + "=" * 80, bold=True)
        tfu.cprint(f"{self.user_prompt}", tfu.YELLOW, True)
        tfu.cprint("=" * 80 + "\n", bold=True)
        
        if self.user_prompt is None:
            tfu.cprint("[Warning] No user prompt provided. Using empty prompt.", tfu.LIGHT_RED)
            self.user_prompt = default_prompt
        
        # Feedback Evaluation - use raw user input, not template-processed prompt
        # This avoids false positives from template text (e.g., "don't" in task_prompt.txt)
        raw_user_input = getattr(self.prompt_organizer, '_raw_user_input', '')
        has_feedback = self._evaluate_feedback(raw_user_input)
        
        if has_feedback:
            # Feedback processing: If it starts with “feedback : ”
            if self.user_prompt.lower().startswith("feedback :"):
                feedback_text = self.user_prompt[10:].strip()  # Remove “feedback : ”
            else:
                feedback_text = self.user_prompt
            
            # Feedback Generation VLM Call
            system_prompt = self.prompt_organizer.get_system_prompt(self.wrapper)
            self.vlm_gen_feedback(system_prompt, feedback_text)
            
            # Skip proceeding to create a general action after processing feedback.
            tfu.cprint("\n[4-1] Feedback processing complete! Proceeding to the next step.", tfu.LIGHT_GREEN, True)
            return True
        
        # Create a General Action
        system_prompt = self.prompt_organizer.get_system_prompt(self.wrapper, self.last_action_result)
        
        self.vlm_response_parsed = self.vlm_gen_action(
            image=self.image,
            system_prompt=system_prompt,
            user_prompt=self.user_prompt,
            use_logprobs=self.logprobs_active
        )
        
        if not self.vlm_response_parsed:
            return False
        
        # Extract only the first action from the action chunk
        action_chunk = self.vlm_response_parsed.get('action', [])
        if isinstance(action_chunk, str):
            try:
                action_chunk = json.loads(action_chunk)
            except Exception:
                action_chunk = [action_chunk] if action_chunk else []
        if not isinstance(action_chunk, list):
            action_chunk = [str(action_chunk)]
        
        if len(action_chunk) == 0:
            action_str = '0'  # Default value: move up
        else:
            action_str = str(action_chunk[0])
        
        # Memory Parsing
        memory = self.vlm_response_parsed.get('memory', {})
        if isinstance(memory, str):
            try:
                memory = json.loads(memory)
            except Exception:
                memory = {}
        if not isinstance(memory, dict):
            memory = {}
        
        # task_process parsing
        task_process = memory.get('task_process', {})
        if not isinstance(task_process, dict):
            task_process = {"goal": "", "status": "", "blocked_reason": ""}
        
        # Parsing last_action_result (in VLM response)
        vlm_last_action_result = memory.get('last_action_result', {})
        if not isinstance(vlm_last_action_result, dict):
            vlm_last_action_result = {}
        
        # Memory Update
        if isinstance(memory, dict):
            self.prompt_organizer.previous_action = memory.get('previous_action', action_str)
            self.prompt_organizer.task_process = {
                "goal": task_process.get('goal', ''),
                "status": task_process.get('status', ''),
                "blocked_reason": task_process.get('blocked_reason', '')
            }
            
            # When VLM is set to blocked status, it is reflected.
            if task_process.get('status') == 'blocked':
                blocked_reason = task_process.get('blocked_reason', '')
                if blocked_reason:
                    tfu.cprint(f"\n[Memory] Task marked as blocked: {blocked_reason}")
        
        # Grounding Update (If from a response)
        grounding_update = self.vlm_response_parsed.get('grounding', '')
        grounding_updated = False
        if grounding_update and grounding_update.strip():
            self.prompt_organizer.update_grounding(grounding_update)
            grounding_updated = True
        
        # CLI output: Action, Reasoning, Memory, Grounding
        tfu.cprint("\n" + "=" * 80)
        tfu.cprint("[VLM Response Information]")
        tfu.cprint("=" * 80)
        
        # Action Chunk Output
        tfu.cprint("\n[Action Chunk]")
        tfu.cprint("-" * 80)
        if len(action_chunk) > 0:
            for i, action in enumerate(action_chunk, 1):
                marker = "→ Execution" if i == 1 else "  Prediction"
                tfu.cprint(f"{marker} [{i}] {action}", indent=4)
        else:
            tfu.cprint("(No action)", indent=4)
        
        # Reasoning Output
        reasoning = self.vlm_response_parsed.get('reasoning', '')
        tfu.cprint("\n[Reasoning...]")
        tfu.cprint("-" * 80)
        if reasoning:
            tfu.cprint(f"{reasoning}", indent=4)
        else:
            tfu.cprint("(None)", indent=4)
        
        # Memory Output
        tfu.cprint("\n[Memory]")
        tfu.cprint("-" * 80)
        spatial_desc = memory.get('spatial_description', '')
        task_goal = task_process.get('goal', '')
        task_status = task_process.get('status', '')
        prev_action = memory.get('previous_action', '')
        
        tfu.cprint("Spatial Description:", indent=4)
        if spatial_desc:
            tfu.cprint(f"{spatial_desc}", indent=8)
        else:
            tfu.cprint("(None)", indent=8)
        
        tfu.cprint("Task Process:", indent=4)
        if task_goal or task_status:
            tfu.cprint(f"Goal: {task_goal if task_goal else '(None)'}", indent=8)
            tfu.cprint(f"Status: {task_status if task_status else '(None)'}", indent=8)
        else:
            tfu.cprint("(None)", indent=8)
        
        tfu.cprint("Previous Action:", indent=4)
        if prev_action:
            tfu.cprint(f"{prev_action}", indent=8)
        else:
            tfu.cprint("(None)", indent=8)
        
        # Grounding Output (Only if updated)
        if grounding_updated:
            tfu.cprint("\n[Grounding Update]")
            tfu.cprint("-" * 80)
            tfu.cprint(f"{grounding_update}", indent=4)
        
        tfu.cprint("=" * 80)
        
        tfu.cprint("\n[5] Action in progress...")
        
        # Save Current Location (Before Action Execution)
        current_pos_before = tuple(self.state['agent_pos'])
        if isinstance(current_pos_before, np.ndarray):
            current_pos_before = (int(current_pos_before[0]), int(current_pos_before[1]))
        
        try:
            self.action_index = self.wrapper.parse_absolute_action(action_str)
            action_space = self.wrapper.get_absolute_action_space()
            self.action_name = action_space['action_mapping'].get(self.action_index, f"action_{self.action_index}")
            tfu.cprint(f"Action to execute: {self.action_name} (Index: {self.action_index})")
            
            # Since use_absolute_movement=True, step() handles absolute movement.
            _, self.reward, terminated, truncated, _ = self.wrapper.step(self.action_index)
            self.done = terminated or truncated
            
            # Confirm location after action execution
            new_state = self.wrapper.get_state()
            current_pos_after = tuple(new_state['agent_pos'])
            if isinstance(current_pos_after, np.ndarray):
                current_pos_after = (int(current_pos_after[0]), int(current_pos_after[1]))
            
            # Confirming position changes
            position_changed = (current_pos_before != current_pos_after)
            
            # Check if this is a movement action (0=up, 1=down, 2=left, 3=right)
            is_movement_action = (self.action_index in [0, 1, 2, 3])
            
            # Action Result Determination
            if is_movement_action:
                # For movement actions: success if position changed
                action_success = position_changed or self.reward > 0
                failure_reason = ""
                if not action_success:
                    # Reasoning about Failure Causes (Based on Visible Information in the Image)
                    if not position_changed:
                        failure_reason = "blocked_by_obstacle"
                    else:
                        failure_reason = "unknown"
            else:
                # For non-movement actions (pickup, drop, toggle): don't check reward
                # These actions don't change position, so we don't check position_changed or reward
                action_success = True  # Always consider as executed (not failed due to obstacle)
                failure_reason = ""
            
            # Check pickup failure: if pickup action was executed but nothing was picked up
            if self.action_index == 4:  # pickup action
                env = self.wrapper.env
                carrying_count = 0
                if hasattr(env, 'carrying'):
                    if isinstance(env.carrying, list):
                        carrying_count = len(env.carrying)
                    elif env.carrying is not None:
                        carrying_count = 1
                # Note: We can't easily detect if pickup failed since we always allow multiple pickups
                # This check is less useful now, but kept for compatibility
            
            # Last action result update
            self.last_action_result = {
                "action": self.action_name,
                "success": action_success,
                "failure_reason": failure_reason,
                "position_changed": position_changed
            }
            
            tfu.cprint(f"Reward: {self.reward}, End: {self.done}")
            tfu.cprint(f"Action Result: {'Success' if action_success else 'Failure'} (Position Change: {'Yes' if position_changed else 'No'})")
            if not action_success:
                print(f"Reasons for Failure: {failure_reason}")
            
            # Display carrying object information
            env = self.wrapper.env
            if hasattr(env, 'carrying'):
                if isinstance(env.carrying, list):
                    if len(env.carrying) > 0:
                        carrying_info = self._format_carrying_object(env.carrying)
                        tfu.cprint(f"Carrying Objects ({len(env.carrying)}): {carrying_info}", color=tfu.CYAN, bold=True)
                    else:
                        tfu.cprint("Carrying Objects: None", color=tfu.LIGHT_BLACK)
                elif env.carrying is not None:
                    carrying_info = self._format_carrying_object(env.carrying)
                    tfu.cprint(f"Carrying Object: {carrying_info}", color=tfu.CYAN, bold=True)
                else:
                    tfu.cprint("Carrying Object: None", color=tfu.LIGHT_BLACK)
            else:
                tfu.cprint("Carrying Object: None", color=tfu.LIGHT_BLACK)
                
        except Exception as e:
            tfu.cprint(f"Action execution failed: {e}")
            import traceback
            traceback.print_exc()
            self.action_index = 0
            self.action_name = "move up"
            try:
                _, self.reward, terminated, truncated, _ = self.wrapper.step(0)
                self.done = terminated or truncated
            except:
                pass
            
            # Update last_action_result even when an exception occurs
            self.last_action_result = {
                "action": self.action_name,
                "success": False,
                "failure_reason": "exception",
                "position_changed": False
            }
            
            # Display carrying object information even on exception
            env = self.wrapper.env
            if hasattr(env, 'carrying'):
                if isinstance(env.carrying, list):
                    if len(env.carrying) > 0:
                        carrying_info = self._format_carrying_object(env.carrying)
                        tfu.cprint(f"Carrying Objects ({len(env.carrying)}): {carrying_info}", color=tfu.CYAN, bold=True)
                    else:
                        tfu.cprint("Carrying Objects: None", color=tfu.LIGHT_BLACK)
                elif env.carrying is not None:
                    carrying_info = self._format_carrying_object(env.carrying)
                    tfu.cprint(f"Carrying Object: {carrying_info}", color=tfu.CYAN, bold=True)
                else:
                    tfu.cprint("Carrying Object: None", color=tfu.LIGHT_BLACK)
            else:
                tfu.cprint("Carrying Object: None", color=tfu.LIGHT_BLACK)
        
        # Previous action update (action actually executed)
        self.prompt_organizer.previous_action = self.action_name
        
        # new_state has already been retrieved above, so it is reused.
        if 'new_state' not in locals():
            new_state = self.wrapper.get_state()
        self.state = new_state
        self.visualizer.visualize_grid_cli(self.wrapper, new_state)
        updated_image = self.wrapper.get_image()
        self.image = updated_image
        self.visualizer.display_image(updated_image)
        
        self._log_step()
        
        return True
    
    def run(self):
        """
        Main Loop Execution
        """
        
        self.initialize()
        
        if self.step == 0:
            init_step = True
        else:
            raise ValueError("Initially, step isn't 0!")
        
        while not self.done:
            if not self.run_step(init_step):
                break
            
            if self.done:
                tfu.cprint("\n" + "=" * 80)
                tfu.cprint("Goal scored! Game ended.")
                tfu.cprint("=" * 80)
                break
            
            if self.step >= 100:
                tfu.cprint("\nThe maximum number of steps (100) has been reached..")
                break
            
            if self.step >= 1:
                init_step = False
            else:
                raise ValueError("[Error] Step is still 0!")
    
    def cleanup(self):
        """
        Resource Cleanup
        """
        
        self.visualizer.cleanup()
        if self.wrapper:
            self.wrapper.close()
        if self.csv_file:
            self.csv_file.close()
        tfu.cprint(f"\nExperiment complete. Logs are {self.log_dir}. It has been saved.")



