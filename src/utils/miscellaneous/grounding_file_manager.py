######################################################
#                                                    #
#                    GROUNDING                      #
#                   FILE MANAGER                    #
#                                                    #
######################################################


""""""




######################################################
#                                                    #
#                      LIBRARIES                     #
#                                                    #
######################################################


import json
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
import threading

import utils.prompt_manager.terminal_formatting_utils as tfu


######################################################
#                                                    #
#                       CLASS                        #
#                                                    #
######################################################


class GroundingFileManager:
    """Grounding 파일 관리 - 에피소드별 JSON/TXT 저장"""
    
    def __init__(self, episode_dir: Path, episode_id: int):
        """
        Args:
            episode_dir: Episode 디렉토리 경로
            episode_id: Episode 번호
        """
        self.episode_dir = episode_dir
        self.episode_id = episode_id
        
        # Grounding 파일 경로
        self.grounding_json_file = episode_dir / f"grounding_episode_{episode_id}.json"
        self.grounding_txt_file = episode_dir / f"grounding_episode_{episode_id}.txt"
        
        # 전역 최신 Grounding 파일 경로
        self.global_grounding_dir = episode_dir.parent / "grounding"
        self.global_grounding_dir.mkdir(parents=True, exist_ok=True)
        self.global_grounding_json = self.global_grounding_dir / "grounding_latest.json"
        self.global_grounding_txt = self.global_grounding_dir / "grounding_latest.txt"
        
        # 파일 쓰기 동기화를 위한 Lock
        self.file_lock = threading.Lock()
        
        # Grounding 데이터 초기화 (새 구조)
        self.grounding_data = {
            "expr_info": {
                "episode_id": episode_id
            },
            "grounding_per_step": [],
            "stacked_grounding": {
                "user_preference": [],
                "spatial": [],
                "procedural": [],
                "general": []
            },
            "final_grounding": None
        }
        
        # TXT 파일 초기화
        self._initialize_txt_file()
    
    def _initialize_txt_file(self):
        """TXT 파일 헤더 초기화"""
        header = "## Grounding Knowledge (Experience from Past Failures, Successes)\n\n"
        header += "### User Preference Grounding\n\n"
        header += "### Spatial Grounding\n\n"
        header += "### Procedural Grounding\n\n"
        header += "### General Grounding Rules\n\n"
        
        with open(self.grounding_txt_file, 'w', encoding='utf-8') as f:
            f.write(header)
    
    def append_step_feedback(self, 
                            step_id: int,
                            instruction: str,
                            status: str,
                            feedback: Dict[str, Optional[str]]):
        """
        Step별 feedback을 타입별로 누적 저장 (JSON 및 TXT)
        
        Args:
            step_id: Step 번호
            instruction: Instruction 내용
            status: Step 상태 ("SUCCESS" | "FAILURE" | "IN_PROGRESS")
            feedback: 타입별 feedback 딕셔너리
        """
        with self.file_lock:
            # Status를 표준 형식으로 변환 (SUCCESS -> Success, FAILURE -> Failure, IN_PROGRESS -> WiP)
            status_display = {
                "SUCCESS": "Success",
                "FAILURE": "Failure",
                "IN_PROGRESS": "WiP"
            }.get(status, status)
            
            # grounding_per_step에 추가
            step_entry = {
                "step_id": step_id,
                "instruction": instruction,
                "status": status_display,
                "feedback": feedback
            }
            self.grounding_data["grounding_per_step"].append(step_entry)
            
            # stacked_grounding에 타입별로 누적 (Status 포함)
            saved_count = 0
            for grounding_type, content in feedback.items():
                if content and content.strip():
                    stacked_entry = f"[ Step{step_id} - {status_display} ] : {content.strip()}"
                    self.grounding_data["stacked_grounding"][grounding_type].append(stacked_entry)
                    saved_count += 1
            
            # JSON 파일 저장
            with open(self.grounding_json_file, 'w', encoding='utf-8') as f:
                json.dump(self.grounding_data, f, indent=2, ensure_ascii=False)
            
            # TXT 파일에 append (status 포함)
            self._append_to_txt_file(step_id, status_display, feedback)
    
    def _append_to_txt_file(self, step_id: int, status: str, feedback: Dict[str, Optional[str]]):
        """
        TXT 파일에 Step별 feedback append (Status 포함)
        
        Args:
            step_id: Step 번호
            status: Step 상태 ("Success" | "Failure" | "WiP")
            feedback: 타입별 feedback 딕셔너리
        """
        # 섹션별 매핑
        section_mapping = {
            "user_preference": "### User Preference Grounding",
            "spatial": "### Spatial Grounding",
            "procedural": "### Procedural Grounding",
            "general": "### General Grounding Rules"
        }
        
        # 파일 읽기
        current_content = self.grounding_txt_file.read_text(encoding='utf-8')
        lines = current_content.split('\n')
        
        # 각 타입별로 해당 섹션에 추가
        for grounding_type, content in feedback.items():
            if content and content.strip():
                section_name = section_mapping[grounding_type]
                entry = f"[ Step{step_id} - {status} ] : {content.strip()}\n"
                
                # 해당 섹션 찾아서 추가
                new_lines = []
                section_found = False
                
                for i, line in enumerate(lines):
                    new_lines.append(line)
                    if line.strip() == section_name:
                        section_found = True
                        # 섹션 바로 다음에 새 엔트리 추가
                        new_lines.append(entry)
                
                # 섹션이 없었으면 추가
                if not section_found:
                    new_lines.append(f"\n{section_name}\n")
                    new_lines.append(entry)
                
                lines = new_lines
        
        # 파일 쓰기
        with open(self.grounding_txt_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def save_final_grounding(self, final_grounding: Dict[str, Any]):
        """
        Final Grounding 저장 (JSON 및 TXT)
        
        Args:
            final_grounding: VLM이 생성한 최종 Grounding
        """
        with self.file_lock:
            # JSON에 추가
            self.grounding_data["final_grounding"] = {
                "generation_timestamp": datetime.now().isoformat(),
                **final_grounding
            }
            
            # JSON 파일 저장
            with open(self.grounding_json_file, 'w', encoding='utf-8') as f:
                json.dump(self.grounding_data, f, indent=2, ensure_ascii=False)
            
            # TXT 파일에 Final Grounding 섹션 추가
            self._append_final_grounding_to_txt(final_grounding)
            
            # 전역 최신 파일 업데이트
            self._update_global_latest_files()
    
    def _append_final_grounding_to_txt(self, final_grounding: Dict[str, Any]):
        """
        TXT 파일에 Final Grounding 섹션 추가
        
        Args:
            final_grounding: VLM이 생성한 최종 Grounding
        """
        final_section = "\n---\n\n## Final Grounding (Generated by VLM)\n\n"
        
        # 각 타입별로 추가
        type_mapping = {
            "user_preference_grounding": "### User Preference Grounding",
            "spatial_grounding": "### Spatial Grounding",
            "procedural_grounding": "### Procedural Grounding",
            "general_grounding_rules": "### General Grounding Rules"
        }
        
        for key, section_name in type_mapping.items():
            if key in final_grounding:
                content = final_grounding[key].get("content", "")
                if content:
                    final_section += f"{section_name}\n{content}\n\n"
        
        # 파일에 append
        with open(self.grounding_txt_file, 'a', encoding='utf-8') as f:
            f.write(final_section)
    
    def _update_global_latest_files(self):
        """전역 최신 Grounding 파일 업데이트"""
        # JSON 복사
        if self.grounding_json_file.exists():
            import shutil
            shutil.copy2(self.grounding_json_file, self.global_grounding_json)
        
        # TXT 복사
        if self.grounding_txt_file.exists():
            import shutil
            shutil.copy2(self.grounding_txt_file, self.global_grounding_txt)
    
    def get_grounding_txt_path(self) -> Path:
        """Grounding TXT 파일 경로 반환"""
        return self.grounding_txt_file
    
    def get_global_grounding_txt_path(self) -> Path:
        """전역 최신 Grounding TXT 파일 경로 반환"""
        return self.global_grounding_txt
    
    def get_stacked_grounding(self) -> Dict[str, List[str]]:
        """Stacked Grounding 데이터 반환 (Status 포함)"""
        return self.grounding_data.get("stacked_grounding", {})
    
    def get_grounding_per_step(self) -> List[Dict[str, Any]]:
        """Step별 Grounding 데이터 반환"""
        return self.grounding_data.get("grounding_per_step", [])
