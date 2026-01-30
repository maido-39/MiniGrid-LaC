"""
Scenario 2 experiment with entropy comparison (absolute movement version)

This script performs entropy comparison experiments by calling VLM with different
combinations of Grounding and Language Instructions:
- H(X): No Language Instruction, No Grounding
- H(X|S): No Language Instruction, With Grounding
- H(X|L,S): With Language Instruction, With Grounding

Trust value is calculated as: T = (H(X) - H(X|S)) / (H(X) - H(X|L,S))

Usage:
    # Run from src/ directory
    cd src/
    python scenario2_test_entropy_comparison.py [json_map_path]
    
    Examples:
    python scenario2_test_entropy_comparison.py config/example_map.json
    python scenario2_test_entropy_comparison.py config/scenario135_example_map.json
    
    # Show help
    python scenario2_test_entropy_comparison.py --help
"""

import sys
import json
import math
import csv
import numpy as np
from typing import Optional
from datetime import datetime
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import modularized components
import utils.prompt_manager.terminal_formatting_utils as tfu
from utils.miscellaneous.scenario_runner import ScenarioExperiment
from utils.prompt_manager.feedback_utils import strip_feedback_prefix
from utils.miscellaneous.safe_minigrid_registration import safe_minigrid_reg
from utils.miscellaneous.global_variables import MAP_FILE_NAME, LOGPROBS_ENABLED, DEBUG, DEFAULT_INITIAL_MISSION, DEFAULT_MISSION

# MiniGrid Environment Safe Registration
safe_minigrid_reg()


class EntropyComparisonExperiment(ScenarioExperiment):
    """
    Experiment class for entropy comparison.
    Overrides run_step() to perform 3 simultaneous VLM calls and calculate entropy/trust.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize entropy comparison experiment"""
        super().__init__(*args, **kwargs)
        
        # Entropy values storage
        self.entropy_H_X: Optional[float] = None
        self.entropy_H_X_given_S: Optional[float] = None
        self.entropy_H_X_given_LS: Optional[float] = None
        self.trust_T: Optional[float] = None
    
    def _calculate_trust(self, H_X: Optional[float], H_X_given_S: Optional[float], 
                        H_X_given_LS: Optional[float]) -> Optional[float]:
        """
        Calculate Trust value: T = (H(X) - H(X|S)) / (H(X) - H(X|L,S))
        
        Args:
            H_X: Entropy H(X)
            H_X_given_S: Entropy H(X|S)
            H_X_given_LS: Entropy H(X|L,S)
        
        Returns:
            Trust value, or None/NaN if calculation is not possible
        """
        # Check if all values are available
        if H_X is None or H_X_given_S is None or H_X_given_LS is None:
            return None
        
        # Check for division by zero
        denominator = H_X - H_X_given_LS
        if abs(denominator) < 1e-10:  # Very small value, treat as zero
            return float('nan')
        
        numerator = H_X - H_X_given_S
        trust = numerator / denominator
        
        return trust
    
    def _init_csv_logging(self):
        """
        CSV Logging Initialization (override to add entropy fields)
        """
        csv_path = self.log_dir / "experiment_log.csv"
        file_exists = csv_path.exists()
        
        self.csv_file = open(csv_path, 'a', newline='', encoding='utf-8')
        # CSV에 특수문자(줄바꿈, 쉼표 등)가 포함된 경우를 위해 QUOTE_ALL 사용
        self.csv_writer = csv.writer(self.csv_file, quoting=csv.QUOTE_ALL)
        
        if not file_exists:
            self.csv_writer.writerow([
                "step", "timestamp", "agent_x", "agent_y", "agent_dir",
                "action_index", "action_name", "user_prompt",
                "vlm_action_chunk", "vlm_reasoning", "vlm_grounding",
                "memory_spatial_description", "memory_task_goal", "memory_task_status", "memory_task_blocked_reason", "memory_previous_action",
                "last_action_result_action", "last_action_result_success", "last_action_result_failure_reason", "last_action_result_position_changed",
                "reward", "done", "image_path", "vlm_action_logprobs_info",
                "entropy_H_X", "entropy_H_X_given_S", "entropy_H_X_given_LS", "trust_T"
            ])
    
    def _log_step(self):
        """
        Current step logging (override to include entropy values)
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
        
        # Format entropy values for CSV (None/NaN -> empty string)
        entropy_H_X_str = "" if self.entropy_H_X is None else ("" if math.isnan(self.entropy_H_X) else str(self.entropy_H_X))
        entropy_H_X_given_S_str = "" if self.entropy_H_X_given_S is None else ("" if math.isnan(self.entropy_H_X_given_S) else str(self.entropy_H_X_given_S))
        entropy_H_X_given_LS_str = "" if self.entropy_H_X_given_LS is None else ("" if math.isnan(self.entropy_H_X_given_LS) else str(self.entropy_H_X_given_LS))
        trust_T_str = "" if self.trust_T is None else ("" if math.isnan(self.trust_T) else str(self.trust_T))
        
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
            entropy_H_X_str,
            entropy_H_X_given_S_str,
            entropy_H_X_given_LS_str,
            trust_T_str
        ])
        self.csv_file.flush()
        
        # JSON logging (also include entropy values)
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
            "image_path": image_path,
            "action_logprobs_info": self.action_logprobs_info if self.action_logprobs_info else None,
            "entropy_H_X": self.entropy_H_X,
            "entropy_H_X_given_S": self.entropy_H_X_given_S,
            "entropy_H_X_given_LS": self.entropy_H_X_given_LS,
            "trust_T": self.trust_T
        }
        
        # Optional fields 추가 (있을 수도, 없을 수도 있음)
        if self.vlm_response_parsed.get('reasoning'):
            json_data["reasoning"] = self.vlm_response_parsed.get('reasoning')
        
        if hasattr(self, 'logprobs_metadata') and self.logprobs_metadata:
            json_data["logprobs_metadata"] = self.logprobs_metadata
        
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
        
        # Convert non-serializable types before saving
        from utils.miscellaneous.episode_manager import convert_numpy_types
        all_data_serializable = convert_numpy_types(all_data)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_data_serializable, f, indent=2, ensure_ascii=False)
        
        image_path_full = self.log_dir / image_path
        img_pil = Image.fromarray(self.image)
        img_pil.save(image_path_full)
    
    def run_step(self, init_step: bool = False):
        """
        Execute one step with entropy comparison (override)
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
        raw_user_input = getattr(self.prompt_organizer, '_raw_user_input', '')
        has_feedback = self._evaluate_feedback(raw_user_input)
        
        if has_feedback:
            feedback_text = strip_feedback_prefix(self.user_prompt)
            # Feedback Generation VLM Call
            system_prompt = self.prompt_organizer.get_system_prompt(self.wrapper)
            self.vlm_gen_feedback(system_prompt, feedback_text)
            
            # Skip proceeding to create a general action after processing feedback.
            tfu.cprint("\n[4-1] Feedback processing complete! Proceeding to the next step.", tfu.LIGHT_GREEN, True)
            return True
        
        # Get system prompt for all VLM calls
        system_prompt = self.prompt_organizer.get_system_prompt(self.wrapper, self.last_action_result)
        
        # ===== ENTROPY COMPARISON: 3 Parallel VLM Calls with Retry =====
        tfu.cprint("\n" + "=" * 80, bold=True)
        tfu.cprint("[Entropy Comparison] Performing 3 parallel VLM calls (with retry logic)...", tfu.LIGHT_CYAN, bold=True)
        tfu.cprint("=" * 80 + "\n", bold=True)
        
        # Retry configuration
        max_retries = 3
        retry_delay = 1.0  # seconds, with exponential backoff
        
        # Prepare arguments for each VLM call (with retry parameters)
        vlm_calls = [
            ("H(X)", self.vlm_gen_action_H_X, {
                "image": self.image,
                "system_prompt": system_prompt,
                "user_prompt": self.user_prompt,
                "max_retries": max_retries,
                "retry_delay": retry_delay
            }),
            ("H(X|S)", self.vlm_gen_action_H_X_given_S, {
                "image": self.image,
                "system_prompt": system_prompt,
                "user_prompt": self.user_prompt,
                "max_retries": max_retries,
                "retry_delay": retry_delay
            }),
            ("H(X|L,S)", self.vlm_gen_action_H_X_given_LS, {
                "image": self.image,
                "system_prompt": system_prompt,
                "user_prompt": self.user_prompt,
                "max_retries": max_retries,
                "retry_delay": retry_delay
            })
        ]
        
        # Execute all VLM calls in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all tasks
            future_to_name = {
                executor.submit(func, **kwargs): name 
                for name, func, kwargs in vlm_calls
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result(timeout=120)  # 2 minute timeout per call
                    results[name] = result
                    
                    # Check if result is valid
                    action_logprobs_info = result.get('action_logprobs_info', {})
                    if action_logprobs_info and action_logprobs_info.get('action_logprobs'):
                        tfu.cprint(f"[{name}] VLM call completed successfully", tfu.LIGHT_GREEN)
                    else:
                        tfu.cprint(f"[{name}] VLM call completed but with empty/invalid logprobs", tfu.LIGHT_YELLOW)
                except Exception as e:
                    tfu.cprint(f"[{name}] VLM call failed after all retries: {e}", tfu.LIGHT_RED, bold=True)
                    # Return empty result on failure
                    results[name] = {
                        'parsed': {},
                        'logprobs_metadata': {},
                        'action_logprobs_info': {}
                    }
        
        # Extract results with default empty dict
        empty_result = {
            'parsed': {},
            'logprobs_metadata': {},
            'action_logprobs_info': {}
        }
        H_X_result = results.get("H(X)", empty_result)
        H_X_given_S_result = results.get("H(X|S)", empty_result)
        H_X_given_LS_result = results.get("H(X|L,S)", empty_result)
        
        # Report success/failure status
        success_count = sum(1 for r in [H_X_result, H_X_given_S_result, H_X_given_LS_result] 
                          if r.get('action_logprobs_info', {}).get('action_logprobs'))
        tfu.cprint(f"\n[Entropy Comparison] {success_count}/3 VLM calls returned valid data", 
                  tfu.LIGHT_GREEN if success_count == 3 else tfu.LIGHT_YELLOW, bold=True)
        if success_count == 0:
            tfu.cprint("[Hint] Entropy/Trust require logprobs. Use Vertex AI Gemini with LOGPROBS_ENABLED=True in global_variables.py.", tfu.LIGHT_BLACK, italic=True)
        
        # Calculate entropies
        self.entropy_H_X = self._calculate_entropy_from_logprobs(H_X_result.get('action_logprobs_info', {}))
        self.entropy_H_X_given_S = self._calculate_entropy_from_logprobs(H_X_given_S_result.get('action_logprobs_info', {}))
        self.entropy_H_X_given_LS = self._calculate_entropy_from_logprobs(H_X_given_LS_result.get('action_logprobs_info', {}))
        
        # Calculate Trust
        self.trust_T = self._calculate_trust(
            self.entropy_H_X,
            self.entropy_H_X_given_S,
            self.entropy_H_X_given_LS
        )
        
        # Display entropy comparison results
        tfu.cprint("\n" + "=" * 80, bold=True)
        tfu.cprint("[Entropy Comparison Results]", tfu.LIGHT_CYAN, bold=True)
        tfu.cprint("=" * 80)
        tfu.cprint(f"H(X):           {self.entropy_H_X if self.entropy_H_X is not None else 'N/A'}", tfu.LIGHT_BLUE)
        tfu.cprint(f"H(X|S):         {self.entropy_H_X_given_S if self.entropy_H_X_given_S is not None else 'N/A'}", tfu.LIGHT_BLUE)
        tfu.cprint(f"H(X|L,S):       {self.entropy_H_X_given_LS if self.entropy_H_X_given_LS is not None else 'N/A'}", tfu.LIGHT_BLUE)
        tfu.cprint(f"Trust T:        {self.trust_T if self.trust_T is not None and not (isinstance(self.trust_T, float) and math.isnan(self.trust_T)) else 'N/A'}", tfu.LIGHT_GREEN)
        tfu.cprint("=" * 80 + "\n", bold=True)
        
        # Use H(X|L,S) result for actual action execution
        self.vlm_response_parsed = H_X_given_LS_result.get('parsed', {})
        
        if not self.vlm_response_parsed:
            tfu.cprint("[Warning] H(X|L,S) result is empty, cannot proceed.", tfu.LIGHT_RED)
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
        
        # Memory Update: store full memory dict; prompts use $memory[key] / $memory[key][subkey]
        if isinstance(memory, dict):
            memory_to_store = dict(memory)
            if not memory_to_store.get('previous_action') and action_str:
                memory_to_store['previous_action'] = action_str if isinstance(action_str, str) else str(action_str)
            self.prompt_organizer.set_memory_dict(memory_to_store)
            
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
                if hasattr(env, 'carrying') and env.carrying is None:
                    tfu.cprint(f"[WARNING] Pickup action executed but no object was picked up. Front cell may be empty or object cannot be picked up.", tfu.LIGHT_RED, bold=True)
            
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
        
        # Previous action update (action actually executed) — update memory_dict for next prompt
        self.prompt_organizer.memory_dict['previous_action'] = self.action_name
        
        # new_state has already been retrieved above, so it is reused.
        if 'new_state' not in locals():
            new_state = self.wrapper.get_state()
        self.state = new_state
        self.visualizer.visualize_grid_cli(self.wrapper, new_state)
        updated_image = self.wrapper.get_image()
        self.image = updated_image
        self.visualizer.display_image(updated_image)
        
        # Update logprobs info from H(X|L,S) result for logging
        self.logprobs_metadata = H_X_given_LS_result.get('logprobs_metadata', {})
        self.action_logprobs_info = H_X_given_LS_result.get('action_logprobs_info', {})
        
        self._log_step()
        
        return True


def main():
    """Main function for entropy comparison experiment"""
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            tfu.cprint("-- Usage Instructions --", tfu.LIGHT_GREEN, bold=True)
            tfu.cprint("python scenario2_test_entropy_comparison.py [json_map_path]", tfu.LIGHT_RED, italic=True, indent=8)
            tfu.cprint(f"Example: python scenario2_test_entropy_comparison.py config/{MAP_FILE_NAME}", tfu.LIGHT_BLACK, italic=True)
            tfu.cprint(f"Default: Uses MAP_FILE_NAME from global_variables.py (currently: {MAP_FILE_NAME})", tfu.LIGHT_BLACK, italic=True)
            tfu.cprint("\nNote: Run from src/ directory", tfu.LIGHT_BLACK, italic=True)
            tfu.cprint("\nThis experiment performs entropy comparison with 3 VLM calls:", tfu.LIGHT_BLACK, italic=True)
            tfu.cprint("  - H(X): No Language Instruction, No Grounding", tfu.LIGHT_BLACK, italic=True)
            tfu.cprint("  - H(X|S): No Language Instruction, With Grounding", tfu.LIGHT_BLACK, italic=True)
            tfu.cprint("  - H(X|L,S): With Language Instruction, With Grounding", tfu.LIGHT_BLACK, italic=True)
            return
        else:
            json_map_path = sys.argv[1]
    else:
        # Use global MAP_FILE_NAME (will be set to config/{MAP_FILE_NAME} in ScenarioExperiment.__init__)
        json_map_path = None
    
    try:
        # Create and run experiment using EntropyComparisonExperiment
        # use_logprobs and debug use settings from global_variables.py
        experiment = EntropyComparisonExperiment(
            json_map_path=json_map_path,
            use_logprobs=LOGPROBS_ENABLED,  # Use global LOGPROBS_ENABLED setting
            debug=DEBUG  # Use global DEBUG setting
        )
        experiment.run()
        experiment.cleanup()
    except KeyboardInterrupt:
        tfu.cprint("\n\nInterrupted by user.", tfu.LIGHT_BLUE, bold=True)
        if 'experiment' in locals():
            experiment.cleanup()
    except Exception as e:
        tfu.cprint(f"\n\nError occurred: {e}", tfu.LIGHT_RED, bold=True)
        import traceback
        traceback.print_exc()
        if 'experiment' in locals():
            experiment.cleanup()


if __name__ == "__main__":
    main()
