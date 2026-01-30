"""
Scenario 2 experiment with Verbalized Entropy comparison (Tian et al. 2023 based)

This script performs entropy comparison experiments using Verbalized Confidence:
- Instead of logprobs, uses step1/step2/step3 probability distributions
- VLM outputs probability for each direction (north/south/west/east)
- Action is extracted via argmax from probability distributions

Entropy types:
- H(X): No Language Instruction, No Grounding
- H(X|S): No Language Instruction, With Grounding
- H(X|L,S): With Language Instruction, With Grounding

Trust value is calculated as: T = (H(X) - H(X|S)) / (H(X) - H(X|L,S))

Usage:
    # Run from src/ directory
    cd src/
    python scenario2_test_entropy_comparison_refined_entropy.py [json_map_path]
    
    Examples:
    python scenario2_test_entropy_comparison_refined_entropy.py config/example_map.json
    
    # Show help
    python scenario2_test_entropy_comparison_refined_entropy.py --help
"""

import sys
import json
import math
import csv
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import modularized components
import utils.prompt_manager.terminal_formatting_utils as tfu
from utils.miscellaneous.scenario_runner import ScenarioExperiment
from utils.prompt_manager.feedback_utils import strip_feedback_prefix
from utils.miscellaneous.safe_minigrid_registration import safe_minigrid_reg
from utils.miscellaneous.global_variables import (
    MAP_FILE_NAME, DEBUG, DEFAULT_INITIAL_MISSION, DEFAULT_MISSION,
    GROUNDING_FILE_PATH
)
from utils.vlm.vlm_postprocessor import VLMResponsePostProcessor

# MiniGrid Environment Safe Registration
safe_minigrid_reg()


class RefinedEntropyComparisonExperiment(ScenarioExperiment):
    """
    Experiment class for Verbalized Entropy comparison.
    Uses step1/step2/step3 probability distributions instead of logprobs.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize refined entropy comparison experiment"""
        # Disable logprobs since we use verbalized confidence
        kwargs['use_logprobs'] = False
        super().__init__(*args, **kwargs)
        
        # Entropy values storage
        self.entropy_H_X: Optional[float] = None
        self.entropy_H_X_given_S: Optional[float] = None
        self.entropy_H_X_given_LS: Optional[float] = None
        self.trust_T: Optional[float] = None
        
        # Step-wise entropy storage
        self.step_entropies: Dict[str, List[float]] = {
            'H_X': [],
            'H_X_given_S': [],
            'H_X_given_LS': []
        }
        
        # Step probability distributions storage
        self.step_probs: Dict[str, Dict[str, Dict[str, float]]] = {
            'H_X': {},
            'H_X_given_S': {},
            'H_X_given_LS': {}
        }
        
        # Postprocessor for verbalized entropy
        self.verbalized_postprocessor = VLMResponsePostProcessor(
            required_fields=['step1', 'step2', 'step3', 'reasoning']
        )
    
    def _calculate_trust(self, H_X: Optional[float], H_X_given_S: Optional[float], 
                        H_X_given_LS: Optional[float]) -> Optional[float]:
        """
        Calculate Trust value: T = (H(X) - H(X|S)) / (H(X) - H(X|L,S))
        """
        if H_X is None or H_X_given_S is None or H_X_given_LS is None:
            return None
        
        denominator = H_X - H_X_given_LS
        if abs(denominator) < 1e-10:
            return float('nan')
        
        numerator = H_X - H_X_given_S
        trust = numerator / denominator
        
        return trust
    
    def _init_csv_logging(self):
        """
        CSV Logging Initialization (override to add verbalized entropy fields)
        """
        csv_path = self.log_dir / "experiment_log.csv"
        file_exists = csv_path.exists()
        
        self.csv_file = open(csv_path, 'a', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file, quoting=csv.QUOTE_ALL)
        
        if not file_exists:
            self.csv_writer.writerow([
                "step", "timestamp", "agent_x", "agent_y", "agent_dir",
                "action_index", "action_name", "user_prompt",
                "vlm_action_chunk", "vlm_reasoning",
                "memory_task_status", "memory_previous_action",
                "reward", "done", "image_path",
                # Verbalized entropy fields
                "executability",
                "step1_probs", "step2_probs", "step3_probs",
                "step1_entropy", "step2_entropy", "step3_entropy",
                "weighted_entropy_H_X", "weighted_entropy_H_X_given_S", "weighted_entropy_H_X_given_LS",
                "trust_T"
            ])
    
    def _log_step(self):
        """
        Current step logging (override to include verbalized entropy values)
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
                memory = {}
        elif not isinstance(memory, dict):
            memory = {}
        
        task_process = memory.get('task_process', {})
        if not isinstance(task_process, dict):
            task_process = {"status": ""}
        
        action_chunk = self.vlm_response_parsed.get('action', [])
        if isinstance(action_chunk, str):
            try:
                action_chunk = json.loads(action_chunk)
            except Exception:
                action_chunk = [action_chunk] if action_chunk else []
        if not isinstance(action_chunk, list):
            action_chunk = [str(action_chunk)]
        
        # Get step probabilities and entropies from H(X|L,S) result (with safe defaults)
        step_probs_ls = self.step_probs.get('H_X_given_LS', {})
        step1_probs = step_probs_ls.get('step1', {}) if isinstance(step_probs_ls, dict) else {}
        step2_probs = step_probs_ls.get('step2', {}) if isinstance(step_probs_ls, dict) else {}
        step3_probs = step_probs_ls.get('step3', {}) if isinstance(step_probs_ls, dict) else {}
        
        step_entropies_ls = self.step_entropies.get('H_X_given_LS', [])
        if not isinstance(step_entropies_ls, list):
            step_entropies_ls = [0.0, 0.0, 0.0]
        step1_entropy = step_entropies_ls[0] if len(step_entropies_ls) > 0 else 0.0
        step2_entropy = step_entropies_ls[1] if len(step_entropies_ls) > 1 else 0.0
        step3_entropy = step_entropies_ls[2] if len(step_entropies_ls) > 2 else 0.0
        
        # Format entropy values
        def format_entropy(val):
            if val is None:
                return ""
            if isinstance(val, float) and math.isnan(val):
                return ""
            return str(val)
        
        try:
            self.csv_writer.writerow([
                self.step,
                timestamp,
                agent_x,
                agent_y,
                int(self.state['agent_dir']),
                getattr(self, 'action_index', 0),
                getattr(self, 'action_name', 'unknown'),
                getattr(self, 'user_prompt', ''),
                json.dumps(action_chunk, ensure_ascii=False),
                self.vlm_response_parsed.get('reasoning', ''),
                task_process.get('status', ''),
                memory.get('previous_action', ''),
                float(getattr(self, 'reward', 0.0)),
                bool(getattr(self, 'done', False)),
                image_path,
                self.vlm_response_parsed.get('executability', 0.5),
                json.dumps(step1_probs, ensure_ascii=False),
                json.dumps(step2_probs, ensure_ascii=False),
                json.dumps(step3_probs, ensure_ascii=False),
                step1_entropy,
                step2_entropy,
                step3_entropy,
                format_entropy(self.entropy_H_X),
                format_entropy(self.entropy_H_X_given_S),
                format_entropy(self.entropy_H_X_given_LS),
                format_entropy(self.trust_T)
            ])
            self.csv_file.flush()
        except Exception as e:
            tfu.cprint(f"[Warning] Failed to write CSV log: {e}", tfu.LIGHT_YELLOW)
        
        # JSON logging with error handling
        json_path = self.log_dir / "experiment_log.json"
        try:
            json_data = {
                "step": self.step,
                "timestamp": timestamp,
                "state": {
                    "agent_pos": [agent_x, agent_y],
                    "agent_dir": int(self.state['agent_dir']),
                    "mission": str(self.state.get('mission', ''))
                },
                "action": {
                    "index": getattr(self, 'action_index', 0),
                    "name": getattr(self, 'action_name', 'unknown')
                },
                "user_prompt": getattr(self, 'user_prompt', ''),
                "vlm_response": self.vlm_response_parsed,
                "verbalized_entropy": {
                    "executability": self.vlm_response_parsed.get('executability', 0.5),
                    # Backward compatibility: direct access to entropy values
                    "H_X": self.entropy_H_X,
                    "H_X_given_S": self.entropy_H_X_given_S,
                    "H_X_given_LS": self.entropy_H_X_given_LS,
                    "trust_T": self.trust_T,
                    # Detailed information for each entropy type
                    "H_X_details": {
                        "weighted_entropy": self.entropy_H_X,
                        "step_probs": self.step_probs.get('H_X', {}),
                        "step_entropies": self.step_entropies.get('H_X', [])
                    },
                    "H_X_given_S_details": {
                        "weighted_entropy": self.entropy_H_X_given_S,
                        "step_probs": self.step_probs.get('H_X_given_S', {}),
                        "step_entropies": self.step_entropies.get('H_X_given_S', [])
                    },
                    "H_X_given_LS_details": {
                        "weighted_entropy": self.entropy_H_X_given_LS,
                        "step_probs": self.step_probs.get('H_X_given_LS', {}),
                        "step_entropies": self.step_entropies.get('H_X_given_LS', [])
                    }
                },
                "reward": float(getattr(self, 'reward', 0.0)),
                "done": bool(getattr(self, 'done', False)),
                "image_path": image_path
            }
            
            all_data = []
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    try:
                        all_data = json.load(f)
                        if not isinstance(all_data, list):
                            all_data = [all_data]
                    except (json.JSONDecodeError, IOError) as e:
                        tfu.cprint(f"[Warning] Failed to read existing JSON log: {e}", tfu.LIGHT_YELLOW)
                        all_data = []
            
            all_data.append(json_data)
            
            from utils.miscellaneous.episode_manager import convert_numpy_types
            all_data_serializable = convert_numpy_types(all_data)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_data_serializable, f, indent=2, ensure_ascii=False)
        except Exception as e:
            tfu.cprint(f"[Warning] Failed to write JSON log: {e}", tfu.LIGHT_YELLOW)
        
        # Save image with error handling
        try:
            image_path_full = self.log_dir / image_path
            img_pil = Image.fromarray(self.image)
            img_pil.save(image_path_full)
        except Exception as e:
            tfu.cprint(f"[Warning] Failed to save image: {e}", tfu.LIGHT_YELLOW)
    
    def _vlm_call_verbalized(
        self,
        image: np.ndarray,
        system_prompt: str,
        user_prompt: str,
        include_grounding: bool = False,
        include_language: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        show_debug: bool = False
    ) -> Dict[str, Any]:
        """
        VLM call for verbalized entropy with retry logic for broken JSON
        
        Args:
            image: Input image
            system_prompt: System prompt (verbalized entropy format)
            user_prompt: User prompt
            include_grounding: Whether to include grounding in prompt
            include_language: Whether to include language instruction
            max_retries: Maximum number of retries for broken JSON
            retry_delay: Delay between retries in seconds
        
        Returns:
            Dictionary containing parsed response and entropy info
        """
        import time
        
        # System prompt is already modified based on grounding_file_path
        # (grounding content is included in System Prompt at generation time)
        modified_system_prompt = system_prompt
        
        # Modify user prompt based on conditions
        if include_language:
            modified_user_prompt = user_prompt
        else:
            # Generic prompt without specific language instruction
            modified_user_prompt = "Continue the mission."
        
        # Retry loop for broken JSON
        last_error = None
        last_raw_response = ''
        
        for attempt in range(max_retries):
            try:
                # Only show debug output for H(X|L,S) (when show_debug=True)
                raw_response = self.vlm_processor.requester(
                    image=image,
                    system_prompt=modified_system_prompt,
                    user_prompt=modified_user_prompt,
                    debug=show_debug
                )
                last_raw_response = raw_response
                
                # Check if response is empty
                if not raw_response or not raw_response.strip():
                    raise ValueError("Empty response from VLM")
                
                # Try to parse JSON - if it fails, retry
                # Extract JSON from response
                text = raw_response.strip()
                if "```json" in text:
                    start_idx = text.find("```json") + 7
                    end_idx = text.find("```", start_idx)
                    if end_idx == -1:
                        raise json.JSONDecodeError("No closing ``` found", text, len(text))
                    text = text[start_idx:end_idx].strip()
                elif "```" in text:
                    start_idx = text.find("```") + 3
                    end_idx = text.find("```", start_idx)
                    if end_idx == -1:
                        raise json.JSONDecodeError("No closing ``` found", text, len(text))
                    text = text[start_idx:end_idx].strip()
                
                # Check if extracted text is empty
                if not text:
                    raise ValueError("No JSON content found in response")
                
                # Attempt JSON parsing (will raise JSONDecodeError if broken)
                json.loads(text)
                
                # JSON is valid, now parse with our processor
                parsed = self.verbalized_postprocessor.parse_verbalized_entropy_response(
                    raw_response, strict=False
                )
                
                # Calculate step entropies
                step_entropies = []
                step_probs = {}
                
                for step_name in ['step1', 'step2', 'step3']:
                    step_data = parsed.get(step_name, {})
                    if isinstance(step_data, dict):
                        # Normalize probabilities
                        normalized = self.verbalized_postprocessor.normalize_step_probs(step_data)
                        step_probs[step_name] = normalized
                        
                        # Calculate entropy for this step
                        entropy = self.verbalized_postprocessor.calculate_step_entropy(normalized)
                        step_entropies.append(entropy)
                    else:
                        step_probs[step_name] = {'north': 0.25, 'south': 0.25, 'west': 0.25, 'east': 0.25}
                        step_entropies.append(2.0)  # Maximum entropy for uniform distribution
                
                # Calculate weighted entropy (50/30/20)
                weighted_entropy = self.verbalized_postprocessor.calculate_weighted_entropy(
                    step_probs.get('step1', {}),
                    step_probs.get('step2', {}),
                    step_probs.get('step3', {})
                )
                
                return {
                    'parsed': parsed,
                    'step_probs': step_probs,
                    'step_entropies': step_entropies,
                    'weighted_entropy': weighted_entropy,
                    'raw_response': raw_response,
                    'retry_count': attempt
                }
                
            except json.JSONDecodeError as e:
                last_error = e
                error_msg = getattr(e, 'msg', str(e))
                tfu.cprint(f"[Retry {attempt + 1}/{max_retries}] JSON parsing failed: {error_msg}, retrying...", tfu.LIGHT_YELLOW)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                continue
                
            except ValueError as e:
                last_error = e
                tfu.cprint(f"[Retry {attempt + 1}/{max_retries}] VLM response error: {e}, retrying...", tfu.LIGHT_YELLOW)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                continue
                
            except Exception as e:
                last_error = e
                tfu.cprint(f"[Retry {attempt + 1}/{max_retries}] VLM call error: {e}", tfu.LIGHT_YELLOW)
                if DEBUG:
                    import traceback
                    traceback.print_exc()
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                continue
        
        # All retries failed
        tfu.cprint(f"[VLM Call Failed] All {max_retries} attempts failed. Last error: {last_error}", tfu.LIGHT_RED)
        return {
            'parsed': {},
            'step_probs': {},
            'step_entropies': [2.0, 2.0, 2.0],
            'weighted_entropy': 2.0,
            'raw_response': last_raw_response,
            'retry_count': max_retries
        }
    
    def vlm_gen_action_verbalized_H_X(self, image, system_prompt, user_prompt, **kwargs):
        """H(X): No Language Instruction, No Grounding"""
        return self._vlm_call_verbalized(
            image=image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            include_grounding=False,
            include_language=False
        )
    
    def vlm_gen_action_verbalized_H_X_given_S(self, image, system_prompt, user_prompt, **kwargs):
        """H(X|S): No Language Instruction, With Grounding"""
        return self._vlm_call_verbalized(
            image=image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            include_grounding=True,
            include_language=False
        )
    
    def vlm_gen_action_verbalized_H_X_given_LS(self, image, system_prompt, user_prompt, **kwargs):
        """H(X|L,S): With Language Instruction, With Grounding"""
        return self._vlm_call_verbalized(
            image=image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            include_grounding=True,
            include_language=True,
            show_debug=DEBUG  # Respect global DEBUG for H(X|L,S) debug output
        )
    
    def run_step(self, init_step: bool = False):
        """
        Execute one step with verbalized entropy comparison (override)
        """
        self.step += 1
        tfu.cprint("\n" + "=" * 80 + "\n", bold=True)
        tfu.cprint(f"STEP {self.step} (Verbalized Entropy Mode)", bold=True, indent=8)
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
        
        self.user_prompt = self.prompt_organizer.get_user_prompt(default_prompt, init_step=init_step)
        
        tfu.cprint("\n" + "=" * 80, bold=True)
        tfu.cprint(f"{self.user_prompt}", tfu.YELLOW, True)
        tfu.cprint("=" * 80 + "\n", bold=True)
        
        if self.user_prompt is None:
            tfu.cprint("[Warning] No user prompt provided. Using empty prompt.", tfu.LIGHT_RED)
            self.user_prompt = default_prompt
        
        # Feedback Evaluation
        raw_user_input = getattr(self.prompt_organizer, '_raw_user_input', '')
        has_feedback = self._evaluate_feedback(raw_user_input)
        
        if has_feedback:
            feedback_text = strip_feedback_prefix(self.user_prompt)
            system_prompt = self.prompt_organizer.get_system_prompt(self.wrapper)
            self.vlm_gen_feedback(system_prompt, feedback_text)
            
            tfu.cprint("\n[4-1] Feedback processing complete! Proceeding to the next step.", tfu.LIGHT_GREEN, True)
            return True
        
        # Ensure last_action_result is initialized
        if not hasattr(self, 'last_action_result') or not self.last_action_result:
            self.last_action_result = {
                "action": "",
                "success": True,
                "failure_reason": "",
                "position_changed": True
            }
        
        # Grounding 파일 경로 가져오기 (GROUNDING_FILE_PATH 설정 시)
        # scenario_runner.py와 동일한 로직 사용
        grounding_file_path = None
        if GROUNDING_FILE_PATH:
            from pathlib import Path
            # 여러 파일 지원: 리스트 또는 쉼표로 구분된 문자열 처리
            if isinstance(GROUNDING_FILE_PATH, str):
                if ',' in GROUNDING_FILE_PATH:
                    file_paths = [p.strip() for p in GROUNDING_FILE_PATH.split(',')]
                else:
                    file_paths = [GROUNDING_FILE_PATH]
            elif isinstance(GROUNDING_FILE_PATH, list):
                file_paths = GROUNDING_FILE_PATH
            else:
                file_paths = []
            
            # 각 파일 경로를 절대 경로로 변환하고 존재 여부 확인
            resolved_paths = []
            for file_path in file_paths:
                file_path_str = str(file_path).strip()
                potential_path = None
                
                # 절대 경로인 경우
                if Path(file_path_str).is_absolute():
                    potential_path = Path(file_path_str)
                # logs/grounding/grounding_latest.txt 형식인 경우 (상대 경로)
                elif file_path_str.startswith("logs/"):
                    # 프로젝트 루트 기준
                    project_root = Path(__file__).parent.parent
                    potential_path = project_root / file_path_str
                    if not potential_path.exists():
                        src_root = project_root / "src"
                        potential_path = src_root / file_path_str
                else:
                    # 현재 log_dir 기준으로 찾기
                    if hasattr(self, 'log_dir'):
                        potential_path = self.log_dir.parent / file_path_str
                    if not potential_path or not potential_path.exists():
                        project_root = Path(__file__).parent.parent
                        potential_path = project_root / file_path_str
                    if not potential_path.exists():
                        src_root = project_root / "src"
                        potential_path = src_root / file_path_str
                
                if potential_path and potential_path.exists():
                    resolved_paths.append(str(potential_path.resolve()))
            
            # 여러 파일이 있으면 리스트로 전달
            if resolved_paths:
                if len(resolved_paths) == 1:
                    grounding_file_path = resolved_paths[0]
                else:
                    grounding_file_path = resolved_paths
        
        # ===== VERBALIZED ENTROPY COMPARISON: 3 Parallel VLM Calls =====
        tfu.cprint("\n" + "=" * 80, bold=True)
        tfu.cprint("[Verbalized Entropy] Performing 3 parallel VLM calls...", tfu.LIGHT_CYAN, bold=True)
        tfu.cprint("=" * 80 + "\n", bold=True)
        
        # Prepare arguments for each VLM call
        # 각 entropy 타입에 따라 다른 System Prompt 생성 (grounding 포함 여부)
        vlm_calls = [
            ("H(X)", self.vlm_gen_action_verbalized_H_X, {
                "image": self.image,
                "system_prompt": self.prompt_organizer.get_verbalized_entropy_system_prompt(
                    self.wrapper, self.last_action_result, grounding_file_path=None  # No grounding
                ),
                "user_prompt": self.user_prompt
            }),
            ("H(X|S)", self.vlm_gen_action_verbalized_H_X_given_S, {
                "image": self.image,
                "system_prompt": self.prompt_organizer.get_verbalized_entropy_system_prompt(
                    self.wrapper, self.last_action_result, grounding_file_path=grounding_file_path  # With grounding
                ),
                "user_prompt": self.user_prompt
            }),
            ("H(X|L,S)", self.vlm_gen_action_verbalized_H_X_given_LS, {
                "image": self.image,
                "system_prompt": self.prompt_organizer.get_verbalized_entropy_system_prompt(
                    self.wrapper, self.last_action_result, grounding_file_path=grounding_file_path  # With grounding
                ),
                "user_prompt": self.user_prompt
            })
        ]
        
        # Execute all VLM calls in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_name = {
                executor.submit(func, **kwargs): name 
                for name, func, kwargs in vlm_calls
            }
            
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result(timeout=120)
                    results[name] = result
                    
                    weighted_entropy = result.get('weighted_entropy', None)
                    if weighted_entropy is not None:
                        tfu.cprint(f"[{name}] Completed - Weighted Entropy: {weighted_entropy:.4f}", tfu.LIGHT_GREEN)
                    else:
                        tfu.cprint(f"[{name}] Completed but entropy calculation failed", tfu.LIGHT_YELLOW)
                except Exception as e:
                    tfu.cprint(f"[{name}] VLM call failed: {e}", tfu.LIGHT_RED, bold=True)
                    import traceback
                    if DEBUG:
                        traceback.print_exc()
                    results[name] = {
                        'parsed': {},
                        'step_probs': {},
                        'step_entropies': [2.0, 2.0, 2.0],
                        'weighted_entropy': 2.0,
                        'raw_response': ''
                    }
        
        # Extract results
        empty_result = {
            'parsed': {},
            'step_probs': {},
            'step_entropies': [2.0, 2.0, 2.0],
            'weighted_entropy': 2.0
        }
        H_X_result = results.get("H(X)", empty_result)
        H_X_given_S_result = results.get("H(X|S)", empty_result)
        H_X_given_LS_result = results.get("H(X|L,S)", empty_result)
        
        # Store step probabilities and entropies
        self.step_probs['H_X'] = H_X_result.get('step_probs', {})
        self.step_probs['H_X_given_S'] = H_X_given_S_result.get('step_probs', {})
        self.step_probs['H_X_given_LS'] = H_X_given_LS_result.get('step_probs', {})
        
        self.step_entropies['H_X'] = H_X_result.get('step_entropies', [])
        self.step_entropies['H_X_given_S'] = H_X_given_S_result.get('step_entropies', [])
        self.step_entropies['H_X_given_LS'] = H_X_given_LS_result.get('step_entropies', [])
        
        # Get weighted entropies
        self.entropy_H_X = H_X_result.get('weighted_entropy', None)
        self.entropy_H_X_given_S = H_X_given_S_result.get('weighted_entropy', None)
        self.entropy_H_X_given_LS = H_X_given_LS_result.get('weighted_entropy', None)
        
        # Calculate Trust
        self.trust_T = self._calculate_trust(
            self.entropy_H_X,
            self.entropy_H_X_given_S,
            self.entropy_H_X_given_LS
        )
        
        # Display entropy comparison results
        tfu.cprint("\n" + "=" * 80, bold=True)
        tfu.cprint("[Verbalized Entropy Comparison Results]", tfu.LIGHT_CYAN, bold=True)
        tfu.cprint("=" * 80)
        tfu.cprint(f"H(X):           {self.entropy_H_X:.4f}" if self.entropy_H_X is not None else "H(X):           N/A", tfu.LIGHT_BLUE)
        tfu.cprint(f"H(X|S):         {self.entropy_H_X_given_S:.4f}" if self.entropy_H_X_given_S is not None else "H(X|S):         N/A", tfu.LIGHT_BLUE)
        tfu.cprint(f"H(X|L,S):       {self.entropy_H_X_given_LS:.4f}" if self.entropy_H_X_given_LS is not None else "H(X|L,S):       N/A", tfu.LIGHT_BLUE)
        
        trust_display = "N/A"
        if self.trust_T is not None and not (isinstance(self.trust_T, float) and math.isnan(self.trust_T)):
            trust_display = f"{self.trust_T:.4f}"
        tfu.cprint(f"Trust T:        {trust_display}", tfu.LIGHT_GREEN)
        tfu.cprint("=" * 80 + "\n", bold=True)
        
        # Display step-wise probabilities for H(X|L,S)
        tfu.cprint("[Step-wise Probability Distribution (H(X|L,S))]", tfu.LIGHT_CYAN)
        step_names = ['step1', 'step2', 'step3']
        step_entropies_ls = self.step_entropies.get('H_X_given_LS', [])
        for idx, step_name in enumerate(step_names):
            probs = self.step_probs.get('H_X_given_LS', {}).get(step_name, {})
            entropy = step_entropies_ls[idx] if idx < len(step_entropies_ls) else 0.0
            tfu.cprint(f"  {step_name}: N={probs.get('north', 0):.3f} S={probs.get('south', 0):.3f} W={probs.get('west', 0):.3f} E={probs.get('east', 0):.3f} (H={entropy:.3f})", tfu.LIGHT_BLACK)
        
        # Use H(X|L,S) result for actual action execution
        self.vlm_response_parsed = H_X_given_LS_result.get('parsed', {})
        
        if not self.vlm_response_parsed or not isinstance(self.vlm_response_parsed, dict):
            tfu.cprint("[Warning] H(X|L,S) result is empty or invalid, cannot proceed.", tfu.LIGHT_RED)
            # Set default values to prevent crashes
            self.vlm_response_parsed = {
                'action': ['north'],
                'reasoning': '',
                'executability': 0.5,
                'memory': {'task_process': {'status': ''}, 'previous_action': ''}
            }
            tfu.cprint("[Warning] Using default action 'north' to continue.", tfu.LIGHT_YELLOW)
        
        # Extract action from parsed response (already computed via argmax)
        action_chunk = self.vlm_response_parsed.get('action', [])
        if isinstance(action_chunk, str):
            try:
                action_chunk = json.loads(action_chunk)
            except Exception:
                action_chunk = [action_chunk] if action_chunk else []
        if not isinstance(action_chunk, list):
            action_chunk = [str(action_chunk)]
        
        if len(action_chunk) == 0:
            action_str = 'north'  # Default action
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
        
        task_process = memory.get('task_process', {})
        if not isinstance(task_process, dict):
            task_process = {"status": ""}
        
        # Memory Update: store full memory dict; prompts use $memory[key] / $memory[key][subkey]
        if isinstance(memory, dict):
            memory_to_store = dict(memory)
            if not memory_to_store.get('previous_action') and action_str:
                memory_to_store['previous_action'] = action_str if isinstance(action_str, str) else str(action_str)
            self.prompt_organizer.set_memory_dict(memory_to_store)
        
        # CLI output
        tfu.cprint("\n" + "=" * 80)
        tfu.cprint("[VLM Response Information]")
        tfu.cprint("=" * 80)
        
        tfu.cprint("\n[Action (argmax from probabilities)]")
        tfu.cprint("-" * 80)
        if len(action_chunk) > 0:
            for i, action in enumerate(action_chunk, 1):
                marker = "→ Execution" if i == 1 else "  Prediction"
                tfu.cprint(f"{marker} [{i}] {action}", indent=4)
        else:
            tfu.cprint("(No action)", indent=4)
        
        reasoning = self.vlm_response_parsed.get('reasoning', '')
        tfu.cprint("\n[Reasoning]")
        tfu.cprint("-" * 80)
        if reasoning:
            tfu.cprint(f"{reasoning}", indent=4)
        else:
            tfu.cprint("(None)", indent=4)
        
        executability = self.vlm_response_parsed.get('executability', 0.5)
        tfu.cprint(f"\n[Executability]: {executability}")
        
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
            
            _, self.reward, terminated, truncated, _ = self.wrapper.step(self.action_index)
            self.done = terminated or truncated
            
            # Confirm location after action execution
            new_state = self.wrapper.get_state()
            current_pos_after = tuple(new_state['agent_pos'])
            if isinstance(current_pos_after, np.ndarray):
                current_pos_after = (int(current_pos_after[0]), int(current_pos_after[1]))
            
            position_changed = (current_pos_before != current_pos_after)
            is_movement_action = (self.action_index in [0, 1, 2, 3])
            
            if is_movement_action:
                action_success = position_changed or self.reward > 0
                failure_reason = ""
                if not action_success:
                    if not position_changed:
                        failure_reason = "wall"
                    else:
                        failure_reason = "unknown"
            else:
                action_success = True
                failure_reason = ""
            
            self.last_action_result = {
                "action": self.action_name,
                "success": action_success,
                "failure_reason": failure_reason,
                "position_changed": position_changed
            }
            
            tfu.cprint(f"Reward: {self.reward}, End: {self.done}")
            tfu.cprint(f"Action Result: {'Success' if action_success else 'Failure'} (Position Change: {'Yes' if position_changed else 'No'})")
            if not action_success:
                tfu.cprint(f"Failure Reason: {failure_reason}")
                
        except Exception as e:
            tfu.cprint(f"Action execution failed: {e}")
            import traceback
            traceback.print_exc()
            self.action_index = 0
            self.action_name = "north"
            try:
                _, self.reward, terminated, truncated, _ = self.wrapper.step(0)
                self.done = terminated or truncated
            except:
                pass
            
            self.last_action_result = {
                "action": self.action_name,
                "success": False,
                "failure_reason": "exception",
                "position_changed": False
            }
        
        # Previous action update (action actually executed) — update memory_dict for next prompt
        self.prompt_organizer.memory_dict['previous_action'] = self.action_name
        
        if 'new_state' not in locals():
            new_state = self.wrapper.get_state()
        self.state = new_state
        self.visualizer.visualize_grid_cli(self.wrapper, new_state)
        updated_image = self.wrapper.get_image()
        self.image = updated_image
        self.visualizer.display_image(updated_image)
        
        self._log_step()
        
        return True


def main():
    """Main function for refined entropy comparison experiment"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            tfu.cprint("-- Usage Instructions --", tfu.LIGHT_GREEN, bold=True)
            tfu.cprint("python scenario2_test_entropy_comparison_refined_entropy.py [json_map_path]", tfu.LIGHT_RED, italic=True, indent=8)
            tfu.cprint(f"Example: python scenario2_test_entropy_comparison_refined_entropy.py config/{MAP_FILE_NAME}", tfu.LIGHT_BLACK, italic=True)
            tfu.cprint(f"Default: Uses MAP_FILE_NAME from global_variables.py (currently: {MAP_FILE_NAME})", tfu.LIGHT_BLACK, italic=True)
            tfu.cprint("\nNote: Run from src/ directory", tfu.LIGHT_BLACK, italic=True)
            tfu.cprint("\nThis experiment uses Verbalized Confidence for entropy calculation:", tfu.LIGHT_BLACK, italic=True)
            tfu.cprint("  - H(X): No Language Instruction, No Grounding", tfu.LIGHT_BLACK, italic=True)
            tfu.cprint("  - H(X|S): No Language Instruction, With Grounding", tfu.LIGHT_BLACK, italic=True)
            tfu.cprint("  - H(X|L,S): With Language Instruction, With Grounding", tfu.LIGHT_BLACK, italic=True)
            tfu.cprint("\nEntropy is calculated from step1/step2/step3 probability distributions.", tfu.LIGHT_BLACK, italic=True)
            return
        else:
            json_map_path = sys.argv[1]
    else:
        json_map_path = None
    
    try:
        experiment = RefinedEntropyComparisonExperiment(
            json_map_path=json_map_path,
            use_logprobs=False,  # Verbalized entropy doesn't use logprobs
            debug=DEBUG
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
