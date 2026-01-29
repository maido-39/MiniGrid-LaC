######################################################
#                                                    #
#                      EPISODE                      #
#                      MANAGER                      #
#                                                    #
######################################################


"""
Episode Manager Module

This module provides the EpisodeManager class for managing episode data, including saving and loading episode information in JSON format.
"""




######################################################
#                                                    #
#                      LIBRARIES                     #
#                                                    #
######################################################


import json
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
import numpy as np
from PIL import Image

import utils.prompt_manager.terminal_formatting_utils as tfu


def convert_numpy_types(obj):
    """
    Recursively convert numpy types and other non-serializable objects to Python native types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types or other non-serializable objects
        
    Returns:
        Object with all non-serializable types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_types(item) for item in obj.tolist()]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Handle objects with __dict__ (like LogprobsResult)
        try:
            return convert_numpy_types(obj.__dict__)
        except:
            return str(obj)
    elif hasattr(obj, 'as_dict'):
        # Handle objects with as_dict() method
        try:
            return convert_numpy_types(obj.as_dict())
        except:
            return str(obj)
    else:
        # For other types, try to convert to string as fallback
        try:
            json.dumps(obj)  # Test if it's already serializable
            return obj
        except (TypeError, ValueError):
            return str(obj)


######################################################
#                                                    #
#                       CLASS                        #
#                                                    #
######################################################


class EpisodeManager:
    """Episode 데이터 구조 관리 및 저장/로드"""
    
    def __init__(self, episode_id: int, log_dir: Path):
        """
        Args:
            episode_id: Episode 번호
            log_dir: 로그 디렉토리 경로 (logs/)
        """
        self.episode_id = episode_id
        self.log_dir = log_dir
        
        # Episode 데이터 구조 초기화
        self.episode_data = {
            "episode_id": episode_id,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_steps": 0,
            "termination_reason": None,
            "initial_state_image_path": None,
            "steps": [],
            "step_grounding": {
                "user_preference": [],
                "spatial": [],
                "procedural": [],
                "general": []
            },
            "reflexion": None,
            "final_grounding": None
        }
        
        # Episode 폴더 생성
        self.episode_dir = self._create_episode_directory()
        
        # 초기 상태 이미지 저장 경로
        self.initial_state_image_path = None
    
    def _create_episode_directory(self) -> Path:
        """Episode별 폴더 생성"""
        from pathlib import Path
        import sys
        import os
        
        # 스크립트 이름 가져오기
        script_name = Path(sys.argv[0]).stem if sys.argv else "unknown"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 폴더명: episode_{episode_id}_{timestamp}_{script_name}
        folder_name = f"episode_{self.episode_id}_{timestamp}_{script_name}"
        episode_dir = self.log_dir / folder_name
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        # images 폴더 생성
        (episode_dir / "images").mkdir(exist_ok=True)
        
        return episode_dir
    
    def save_initial_state_image(self, image: np.ndarray):
        """
        초기 상태 이미지 저장
        
        Args:
            image: 초기 상태 이미지 (numpy array)
        """
        image_path = self.episode_dir / "images" / "initial_state.png"
        img_pil = Image.fromarray(image)
        img_pil.save(image_path)
        
        self.initial_state_image_path = f"images/initial_state.png"
        self.episode_data["initial_state_image_path"] = self.initial_state_image_path
    
    def add_step(self, 
                 step_id: int,
                 instruction: str,
                 status: str,  # "SUCCESS" | "FAILURE" | "IN_PROGRESS"
                 feedback: Dict[str, Optional[str]],  # {"user_preference": ..., "spatial": ..., "procedural": ..., "general": ...}
                 action: Dict[str, Any],
                 state: Dict[str, Any],
                 image_path: str):
        """
        Step 데이터 추가 및 step_grounding에 누적
        
        Args:
            step_id: Step 번호
            instruction: Instruction 내용
            status: Step 상태
            feedback: 타입별 feedback 딕셔너리
            action: 액션 정보
            state: 상태 정보
            image_path: 이미지 경로
        """
        # Step 데이터 생성
        step_data = {
            "step_id": step_id,
            "instruction": instruction,
            "status": status,
            "feedback": feedback,
            "action": action,
            "state": state,
            "image_path": image_path
        }
        
        # steps에 추가
        self.episode_data["steps"].append(step_data)
        self.episode_data["total_steps"] = len(self.episode_data["steps"])
        
        # step_grounding에 타입별로 누적 저장
        for grounding_type, content in feedback.items():
            if content and content.strip():
                self.episode_data["step_grounding"][grounding_type].append({
                    "step_id": step_id,
                    "content": content.strip()
                })
    
    def set_reflexion(self, reflexion: Dict[str, str]):
        """
        Reflexion 설정
        
        Args:
            reflexion: {"trajectory_summary": ..., "error_diagnosis": ..., "correction_plan": ...}
        """
        self.episode_data["reflexion"] = reflexion
    
    def set_final_grounding(self, final_grounding: Dict[str, Any]):
        """
        Final Grounding 설정
        
        Args:
            final_grounding: VLM이 생성한 최종 Grounding
        """
        self.episode_data["final_grounding"] = final_grounding
    
    def set_termination_reason(self, reason: str):
        """
        종료 이유 설정
        
        Args:
            reason: "done" | "max_steps" | "user_command"
        """
        self.episode_data["termination_reason"] = reason
        self.episode_data["end_time"] = datetime.now().isoformat()
    
    def save(self):
        """Episode 데이터를 JSON 파일로 저장"""
        episode_file = self.episode_dir / f"episode_{self.episode_id}.json"
        
        # Convert numpy types to Python native types for JSON serialization
        episode_data_serializable = convert_numpy_types(self.episode_data)
        
        with open(episode_file, 'w', encoding='utf-8') as f:
            json.dump(episode_data_serializable, f, indent=2, ensure_ascii=False)
        
        tfu.cprint(f"\n[Episode Saved] {episode_file}", tfu.LIGHT_GREEN)
    
    def load(self, episode_file: Path) -> Dict[str, Any]:
        """
        저장된 Episode 로드
        
        Args:
            episode_file: Episode JSON 파일 경로
            
        Returns:
            Episode 데이터 딕셔너리
        """
        with open(episode_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_episode_dir(self) -> Path:
        """Episode 디렉토리 경로 반환"""
        return self.episode_dir
    
    def get_step_grounding(self) -> Dict[str, List[Dict[str, Any]]]:
        """Step별 Grounding 데이터 반환"""
        return self.episode_data["step_grounding"]
    
    def get_all_steps(self) -> List[Dict[str, Any]]:
        """모든 Step 데이터 반환"""
        return self.episode_data["steps"]
    
    def get_initial_state_image_path(self) -> Optional[str]:
        """초기 상태 이미지 경로 반환"""
        return self.initial_state_image_path
