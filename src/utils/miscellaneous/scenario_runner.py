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
from typing import Union            # Union is used in visualize_grid_cli
from datetime import datetime

from utils.miscellaneous.visualizer import Visualizer
from utils.vlm_manager.vlm_processor import VLMProcessor
from utils.user_manager.user_interact import UserInteraction
import utils.prompt_manager.terminal_formatting_utils as tfu
from utils.prompt_manager.prompt_organizer import PromptOrganizer
from utils.prompt_manager.prompt_interp import system_prompt_interp
from utils.map_manager.emoji_map_loader import load_emoji_map_from_json

from utils.miscellaneous.global_variables import DEFAULT_INITIAL_MISSION, DEFAULT_MISSION




######################################################
#                                                    #
#                        CLASS                       #
#                                                    #
######################################################


class ScenarioExperiment:
    """Scenario 2 Experiment Main Class (Runner) - Absolute Coordinate Version"""
    
    def __init__(self,
                 log_dir: Path = None,
                 json_map_path: str = "scenario135_example_map.json"
                ):
        """
        Args:
            log_dir: Log directory path
            json_map_path: JSON map file path
        """
        
        self.wrapper = None
        self.json_map_path = json_map_path
        self.prompt_organizer = PromptOrganizer()
        self.vlm_processor = VLMProcessor()
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
    
    def _evaluate_feedback(self, user_prompt: str) -> bool:
        """
        Feedback Evaluation (Internal Method)
        """
        
        if not user_prompt or not isinstance(user_prompt, str):
            return False
        
        feedback_keywords = [
            "wrong", "incorrect", "that's wrong", "no", "not that", "don't", 
            "shouldn't", "error", "mistake", "why did you", "why didn't you",
            "what are you doing", "where are you going", "not feasible", 
            "cannot", "should not",
            "feedback :"
        ]
        
        user_lower = user_prompt.lower()
        for keyword in feedback_keywords:
            if keyword in user_lower:
                return True
        
        return False
    
    def vlm_gen_action(self,
                       image: np.ndarray,
                       system_prompt: str,
                       user_prompt: str
                      ) -> dict:
        """
        VLM call for Action creation
        """
        
        tfu.cprint("\n[3] Sending Action creation request to VLM...")
        raw_response = self.vlm_processor.requester(
            image=image,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        if not raw_response:
            tfu.cprint("The VLM response is empty.")
            return {}
        
        tfu.cprint("VLM response received")
        tfu.cprint("[4] Parsing response...")
        parsed = self.vlm_processor.parser_action(raw_response)
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
            user_prompt=feedback_user_prompt
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
                "reward", "done", "image_path"
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
            image_path
        ])
        self.csv_file.flush()
        
        json_path = self.log_dir / "experiment_log.json"
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
            "image_path": image_path
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
        self.user_prompt = self.prompt_organizer.get_user_prompt(default_prompt)
        
        tfu.cprint("\n" + "=" * 80, bold=True)
        tfu.cprint(f"{self.user_prompt}", tfu.YELLOW, True)
        tfu.cprint("=" * 80 + "\n", bold=True)
        
        if self.user_prompt is None:
            tfu.cprint("[Warning] No user prompt provided. Using empty prompt.", tfu.LIGHT_RED)
            self.user_prompt = default_prompt
        
        # Feedback Evaluation
        has_feedback = self._evaluate_feedback(self.user_prompt)
        
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
            user_prompt=self.user_prompt
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
            
            # Action Result Determination
            action_success = position_changed or self.reward > 0
            failure_reason = ""
            if not action_success:
                # Reasoning about Failure Causes (Based on Visible Information in the Image)
                if not position_changed:
                    failure_reason = "blocked_by_obstacle"
                else:
                    failure_reason = "unknown"
            
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



