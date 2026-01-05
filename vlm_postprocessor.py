"""
VLM 응답 후처리 모듈

VLM Wrapper가 반환한 원본 응답을 파싱하고 검증하여 구조화된 데이터로 변환합니다.
로봇 제어 명령 추출을 위한 JSON 파싱 및 검증을 담당합니다.
"""

import json
from typing import Dict, Optional


class VLMResponsePostProcessor:
    """
    VLM 응답 후처리 클래스
    
    VLM이 반환한 원본 텍스트를 파싱하여 구조화된 데이터로 변환합니다.
    """
    
    def __init__(
        self,
        required_fields: Optional[list] = None,
        default_fields: Optional[Dict] = None
    ):
        """
        후처리기 초기화
        
        Args:
            required_fields: 필수 필드 리스트 (예: ["robot_action", "context"])
            default_fields: 기본값 딕셔너리 (필드가 없을 때 사용)
        """
        self.required_fields = required_fields or ["robot_action", "context"]
        self.default_fields = default_fields or {}
    
    def parse_json_response(
        self,
        response_text: str,
        strict: bool = True
    ) -> Dict[str, str]:
        """
        응답 텍스트에서 JSON을 파싱하여 추출
        
        Args:
            response_text: VLM이 반환한 원본 텍스트
            strict: True이면 필수 필드 검증, False이면 유연하게 처리
            
        Returns:
            파싱된 JSON 딕셔너리
            
        Raises:
            ValueError: JSON 파싱 실패 또는 필수 필드 누락
        """
        # 텍스트 정리
        response_text = response_text.strip()
        
        # JSON 코드 블록이 있는 경우 추출
        if "```json" in response_text:
            start_idx = response_text.find("```json") + 7
            end_idx = response_text.find("```", start_idx)
            if end_idx != -1:
                response_text = response_text[start_idx:end_idx].strip()
        elif "```" in response_text:
            start_idx = response_text.find("```") + 3
            end_idx = response_text.find("```", start_idx)
            if end_idx != -1:
                response_text = response_text[start_idx:end_idx].strip()
        
        # JSON 파싱 시도
        try:
            parsed = json.loads(response_text)
            
            if not isinstance(parsed, dict):
                raise ValueError(f"응답이 딕셔너리 형식이 아닙니다: {type(parsed)}")
            
            # 필수 필드 확인
            if strict:
                for field in self.required_fields:
                    if field not in parsed:
                        raise ValueError(f"응답에 필수 필드 '{field}'가 없습니다.")
            
            # 기본값 적용 (원본 타입 유지)
            result = {}
            for field in self.required_fields:
                if field in parsed:
                    # 리스트나 딕셔너리는 원본 타입 유지, 나머지는 문자열로 변환
                    value = parsed[field]
                    if isinstance(value, (list, dict)):
                        result[field] = value
                    else:
                        result[field] = str(value)
                elif field in self.default_fields:
                    default_value = self.default_fields[field]
                    if isinstance(default_value, (list, dict)):
                        result[field] = default_value
                    else:
                        result[field] = str(default_value)
                else:
                    result[field] = ""
            
            return result
            
        except json.JSONDecodeError as e:
            raise ValueError(
                f"응답을 JSON으로 파싱할 수 없습니다: {e}\n"
                f"원본 응답: {response_text[:200]}..."
            )
    
    def extract_robot_action(
        self,
        response_text: str
    ) -> Dict[str, str]:
        """
        로봇 제어 명령 추출 (robot_action, context 형식)
        
        Args:
            response_text: VLM이 반환한 원본 텍스트
            
        Returns:
            {"robot_action": str, "context": str}
        """
        return self.parse_json_response(
            response_text,
            strict=True
        )
    
    def process(
        self,
        response_text: str,
        strict: bool = True
    ) -> Dict[str, str]:
        """
        응답 텍스트를 처리하여 구조화된 데이터 반환
        
        Args:
            response_text: VLM이 반환한 원본 텍스트
            strict: True이면 필수 필드 검증
            
        Returns:
            파싱된 딕셔너리
        """
        return self.parse_json_response(response_text, strict=strict)


# 편의 함수
def parse_vlm_response(
    response_text: str,
    required_fields: Optional[list] = None
) -> Dict[str, str]:
    """
    VLM 응답을 파싱하는 편의 함수
    
    Args:
        response_text: VLM이 반환한 원본 텍스트
        required_fields: 필수 필드 리스트
        
    Returns:
        파싱된 딕셔너리
    """
    processor = VLMResponsePostProcessor(required_fields=required_fields)
    return processor.process(response_text)

