"""
VLM Response Post-processing Module

Parses and validates raw responses returned by VLM Wrapper, converting them to structured data.
Handles JSON parsing and validation for robot control command extraction.
"""

import json
from typing import Dict, Optional, List, Union, Any


class VLMResponsePostProcessor:
    """
    VLM Response Post-processing Class
    
    Parses raw text returned by VLM and converts it to structured data.
    """
    
    def __init__(
        self,
        required_fields: Optional[list] = None,
        default_fields: Optional[Dict] = None
    ):
        """
        Initialize post-processor
        
        Args:
            required_fields: List of required fields (e.g., ["robot_action", "context"])
            default_fields: Default value dictionary (used when fields are missing)
        """
        self.required_fields = required_fields or ["robot_action", "context"]
        self.default_fields = default_fields or {}
    
    def parse_json_response(
        self,
        response_text: str,
        strict: bool = True
    ) -> Dict[str, str]:
        """
        Parse and extract JSON from response text
        
        Args:
            response_text: Raw text returned by VLM
            strict: If True, validate required fields; if False, handle flexibly
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            ValueError: JSON parsing failure or missing required fields
        """
        # Clean text
        response_text = response_text.strip()
        
        # Extract if JSON code block exists
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
        
        # Try JSON parsing
        try:
            parsed = json.loads(response_text)
            
            if not isinstance(parsed, dict):
                raise ValueError(f"Response is not in dictionary format: {type(parsed)}")
            
            # Check required fields
            if strict:
                for field in self.required_fields:
                    if field not in parsed:
                        raise ValueError(f"Response missing required field '{field}'.")
            
            # Apply defaults (preserve original type)
            result = {}
            
            # Process required_fields
            for field in self.required_fields:
                if field in parsed:
                    # Preserve original type for lists or dictionaries, convert rest to string
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
            
            # Include fields not in required_fields (e.g., reasoning)
            for field, value in parsed.items():
                if field not in result:
                    if isinstance(value, (list, dict)):
                        result[field] = value
                    else:
                        result[field] = str(value)
            
            return result
            
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse response as JSON: {e}\n"
                f"Original response: {response_text[:200]}..."
            )
    
    def extract_robot_action(
        self,
        response_text: str
    ) -> Dict[str, str]:
        """
        Extract robot control command (robot_action, context format)
        
        Args:
            response_text: Raw text returned by VLM
            
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
        """Process VLM response text and return structured data.
        
        Parses the raw text response from a VLM, extracts JSON if present,
        validates required fields, and returns a structured dictionary.
        This is the main method for converting VLM text responses into
        usable data structures.
        
        Args:
            response_text: Raw text response from the VLM. May contain:
                - Plain JSON: {"action": "move up", "reasoning": "..."}
                - JSON in code blocks: ```json {...} ```
                - Mixed text with JSON embedded
            strict: If True, raises ValueError if required fields are missing.
                If False, missing fields are filled with empty strings or
                default values. Defaults to True.
        
        Returns:
            Dictionary containing parsed fields. All values are strings except
            for lists and dictionaries which are preserved as-is. Fields include:
                - All required_fields (as specified in __init__)
                - Any additional fields found in the JSON response
                - Missing required fields (if strict=False): empty strings
                - Missing optional fields: empty strings or default values
        
        Raises:
            ValueError: If JSON parsing fails or if strict=True and required
                fields are missing. The error message includes the original
                response text (truncated) for debugging.
        
        Examples:
            >>> processor = VLMResponsePostProcessor(
            ...     required_fields=["action", "reasoning"]
            ... )
            >>> 
            >>> # Process JSON response
            >>> response = '{"action": "move up", "reasoning": "Go north"}'
            >>> result = processor.process(response)
            >>> print(result['action'])  # "move up"
            >>> 
            >>> # Process response with code block
            >>> response = '''Here's the action:
            ... ```json
            ... {"action": "pickup", "reasoning": "Pick up the object"}
            ... ```
            ... '''
            >>> result = processor.process(response)
            >>> print(result['action'])  # "pickup"
            >>> 
            >>> # Non-strict mode (missing fields allowed)
            >>> result = processor.process(response, strict=False)
            >>> # Missing fields will be empty strings
        
        Note:
            The processor automatically extracts JSON from code blocks if present.
            List and dictionary values are preserved as-is, while other values
            are converted to strings for consistency.
        """
        return self.parse_json_response(response_text, strict=strict)
    
    def get_token_logprobs_info(
        self,
        logprobs_metadata: Dict,
        token_positions: List[int]
    ) -> List[List[Any]]:
        """
        특정 토큰 위치들에 대한 logprobs 정보를 추출합니다.
        
        Args:
            logprobs_metadata: logprobs 메타데이터 딕셔너리
                - 'tokens': List[str] - 토큰 리스트
                - 'token_logprobs': List[float] - 각 토큰의 log probability
                - 'top_logprobs': List[List[Dict]] - 각 토큰 위치의 top-k logprobs
                - 'entropies': List[float] - 각 토큰 위치의 Shannon entropy
            token_positions: 토큰 위치 인덱스 리스트
        
        Returns:
            List of [token(평문), [token1:logprob, token2:logprob, ...], 섀넌 엔트로피, 토큰 위치]
            각 요소는 [token_str, top_logprobs_list, entropy, position] 형태
        """
        tokens = logprobs_metadata.get('tokens', [])
        token_logprobs = logprobs_metadata.get('token_logprobs', [])
        top_logprobs = logprobs_metadata.get('top_logprobs', [])
        entropies = logprobs_metadata.get('entropies', [])
        
        result = []
        for pos in token_positions:
            if pos < 0 or pos >= len(tokens):
                continue
            
            token_str = tokens[pos]
            top_logprobs_list = top_logprobs[pos] if pos < len(top_logprobs) else []
            entropy = entropies[pos] if pos < len(entropies) else None
            
            # top_logprobs를 [token:logprob, ...] 형태로 변환
            top_logprobs_formatted = []
            if top_logprobs_list:
                for item in top_logprobs_list:
                    if isinstance(item, dict):
                        token_name = item.get('token', '')
                        logprob = item.get('log_probability', 0.0)
                        top_logprobs_formatted.append(f"{token_name}:{logprob:.4f}")
            
            result.append([token_str, top_logprobs_formatted, entropy, pos])
        
        return result
    
    def find_action_token_positions(
        self,
        logprobs_metadata: Dict,
        action_field: str = "action"
    ) -> List[int]:
        """
        JSON을 파싱하여 action 필드의 실제 값 토큰 위치를 찾습니다.
        logprobs_metadata만 받아서 동작합니다.
        
        Args:
            logprobs_metadata: logprobs 메타데이터
                - 'tokens': List[str] - 토큰 리스트
            action_field: action 필드 이름 (기본값: "action")
        
        Returns:
            action 값에 해당하는 토큰 위치 인덱스 리스트
            (action이 배열인 경우 여러 개, 단일 값인 경우 하나)
        """
        import re
        
        tokens = logprobs_metadata.get('tokens', [])
        if not tokens:
            return []
        
        # JSON 파싱
        try:
            # tokens를 문자열로 재구성
            text = ''.join(tokens)
            
            # JSON 코드 블록 제거
            cleaned_text = text.strip()
            if "```json" in cleaned_text:
                start_idx = cleaned_text.find("```json") + 7
                end_idx = cleaned_text.find("```", start_idx)
                if end_idx != -1:
                    cleaned_text = cleaned_text[start_idx:end_idx].strip()
            elif "```" in cleaned_text:
                start_idx = cleaned_text.find("```") + 3
                end_idx = cleaned_text.find("```", start_idx)
                if end_idx != -1:
                    cleaned_text = cleaned_text[start_idx:end_idx].strip()
            
            parsed = json.loads(cleaned_text)
            action_value = parsed.get(action_field)
            
            if action_value is None:
                return []
            
            # action 값이 리스트인지 단일 값인지 확인
            if isinstance(action_value, list):
                action_values = [str(v).strip() for v in action_value if v]
            else:
                action_values = [str(action_value).strip()]
            
            if not action_values:
                return []
            
            # action 필드 이름 찾기
            action_field_idx = None
            for i, token in enumerate(tokens):
                # action 필드 이름 찾기 (정확히 매칭)
                token_clean = re.sub(r'[^\w]', '', token.lower())
                if token_clean == action_field.lower():
                    action_field_idx = i
                    break
            
            if action_field_idx is None:
                return []
            
            # action 필드 다음에 오는 값 찾기
            # JSON 구조: "action": ["value1", "value2", ...] 또는 "action": "value"
            # 토큰 예시: ['{', '"', 'action', '"', ':', '[', '"', 'up', '"', ',', '"', 'down', '"', ']', '}']
            result_positions = []
            i = action_field_idx + 1
            
            # action 필드 다음에 오는 구조 건너뛰기: ":", 공백, "[" 또는 '"'
            while i < len(tokens):
                token_clean = re.sub(r'[^\w]', '', tokens[i].lower())
                if token_clean:  # 유효한 단어가 나오면 중단
                    break
                i += 1
            
            # 각 action 값 찾기
            for action_val in action_values:
                if not action_val:
                    continue
                
                # action 값의 단어 추출 (문장부호 제외)
                action_words = re.findall(r'\w+', action_val.lower())
                if not action_words:
                    continue
                
                # 토큰에서 action 값 찾기
                # 문장부호는 건너뛰고 유효한 단어만 매칭
                found_start = None
                found_end = None
                word_idx = 0
                search_i = i
                
                while search_i < len(tokens) and word_idx < len(action_words):
                    token_clean = re.sub(r'[^\w]', '', tokens[search_i].lower())
                    
                    if token_clean:  # 유효한 단어
                        if token_clean == action_words[word_idx]:
                            if found_start is None:
                                found_start = search_i
                            found_end = search_i
                            word_idx += 1
                        else:
                            # 매칭 실패, 처음부터 다시 시작
                            if found_start is not None:
                                found_start = None
                                found_end = None
                                word_idx = 0
                                # 현재 토큰부터 다시 시작
                                continue
                    search_i += 1
                
                # action 값의 모든 단어를 찾았는지 확인
                if found_start is not None and word_idx == len(action_words):
                    # action 값의 첫 번째 토큰 위치 추가
                    result_positions.append(found_start)
                    # 다음 action 값 검색 시작 위치 업데이트
                    i = found_end + 1
                else:
                    # 매칭 실패, 다음 위치부터 계속 검색
                    if found_start is not None:
                        i = found_start + 1
                    else:
                        i = search_i
            
            # 중복 제거 및 정렬
            result_positions = sorted(list(set(result_positions)))
            return result_positions
            
        except (json.JSONDecodeError, KeyError, AttributeError):
            # 파싱 실패 시 빈 리스트 반환
            return []
    
    def get_action_logprobs(
        self,
        logprobs_metadata: Dict,
        action_field: str = "action"
    ) -> Dict[str, Any]:
        """
        action 필드에 대한 logprobs 정보를 추출합니다.
        logprobs_metadata만 받아서 동작합니다.
        
        Args:
            logprobs_metadata: logprobs 메타데이터
                - 'tokens': List[str] - 토큰 리스트
                - 'token_logprobs': List[float] - 각 토큰의 log probability
                - 'top_logprobs': List[List[Dict]] - 각 토큰 위치의 top-k logprobs
                - 'entropies': List[float] - 각 토큰 위치의 Shannon entropy
            action_field: action 필드 이름 (기본값: "action")
        
        Returns:
            Dictionary containing:
                - 'action_positions': List of token positions for action values
                - 'action_logprobs': List of logprobs info for each action
                    Each item: [token_str, top_logprobs_list, entropy, position]
                - 'action_entropies': List of entropies for each action
        """
        tokens = logprobs_metadata.get('tokens', [])
        if not tokens:
            return {
                'action_positions': [],
                'action_logprobs': [],
                'action_entropies': []
            }
        
        # tokens를 문자열로 재구성하여 JSON 파싱
        import re
        try:
            # tokens를 문자열로 재구성
            text = ''.join(tokens)
            
            # JSON 코드 블록 제거
            cleaned_text = text.strip()
            if "```json" in cleaned_text:
                start_idx = cleaned_text.find("```json") + 7
                end_idx = cleaned_text.find("```", start_idx)
                if end_idx != -1:
                    cleaned_text = cleaned_text[start_idx:end_idx].strip()
            elif "```" in cleaned_text:
                start_idx = cleaned_text.find("```") + 3
                end_idx = cleaned_text.find("```", start_idx)
                if end_idx != -1:
                    cleaned_text = cleaned_text[start_idx:end_idx].strip()
            
            parsed = json.loads(cleaned_text)
            action_value = parsed.get(action_field)
            
            if action_value is None:
                return {
                    'action_positions': [],
                    'action_logprobs': [],
                    'action_entropies': []
                }
            
            # action 값이 리스트인지 단일 값인지 확인
            if isinstance(action_value, list):
                action_values = [str(v).strip() for v in action_value if v]
            else:
                action_values = [str(action_value).strip()]
            
            if not action_values:
                return {
                    'action_positions': [],
                    'action_logprobs': [],
                    'action_entropies': []
                }
            
            # action 필드 이름 찾기
            action_field_idx = None
            for i, token in enumerate(tokens):
                token_clean = re.sub(r'[^\w]', '', token.lower())
                if token_clean == action_field.lower():
                    action_field_idx = i
                    break
            
            if action_field_idx is None:
                return {
                    'action_positions': [],
                    'action_logprobs': [],
                    'action_entropies': []
                }
            
            # action 필드 다음에 오는 값 찾기
            result_positions = []
            i = action_field_idx + 1
            
            # action 필드 다음에 오는 구조 건너뛰기: ":", 공백, "[" 또는 '"'
            while i < len(tokens):
                token_clean = re.sub(r'[^\w]', '', tokens[i].lower())
                if token_clean:  # 유효한 단어가 나오면 중단
                    break
                i += 1
            
            # 각 action 값 찾기
            for action_val in action_values:
                if not action_val:
                    continue
                
                # action 값의 단어 추출 (문장부호 제외)
                action_words = re.findall(r'\w+', action_val.lower())
                if not action_words:
                    continue
                
                # 토큰에서 action 값 찾기
                found_start = None
                found_end = None
                word_idx = 0
                search_i = i
                
                while search_i < len(tokens) and word_idx < len(action_words):
                    token_clean = re.sub(r'[^\w]', '', tokens[search_i].lower())
                    
                    if token_clean:  # 유효한 단어
                        if token_clean == action_words[word_idx]:
                            if found_start is None:
                                found_start = search_i
                            found_end = search_i
                            word_idx += 1
                        else:
                            # 매칭 실패, 처음부터 다시 시작
                            if found_start is not None:
                                found_start = None
                                found_end = None
                                word_idx = 0
                                continue
                    search_i += 1
                
                # action 값의 모든 단어를 찾았는지 확인
                if found_start is not None and word_idx == len(action_words):
                    result_positions.append(found_start)
                    i = found_end + 1
                else:
                    if found_start is not None:
                        i = found_start + 1
                    else:
                        i = search_i
            
            # 중복 제거 및 정렬
            action_positions = sorted(list(set(result_positions)))
            
            # 각 action 위치에 대한 logprobs 정보 추출
            action_logprobs_info = self.get_token_logprobs_info(
                logprobs_metadata, action_positions
            )
            
            # 엔트로피 추출
            entropies = logprobs_metadata.get('entropies', [])
            action_entropies = [entropies[pos] if pos < len(entropies) else None 
                               for pos in action_positions]
            
            return {
                'action_positions': action_positions,
                'action_logprobs': action_logprobs_info,
                'action_entropies': action_entropies
            }
            
        except (json.JSONDecodeError, KeyError, AttributeError):
            return {
                'action_positions': [],
                'action_logprobs': [],
                'action_entropies': []
            }
    
    def print_action_logprobs_info(self, action_logprobs_info: dict):
        """
        Print action logprobs information in a formatted way
        
        Args:
            action_logprobs_info: Dictionary containing action logprobs info from get_action_logprobs()
                Expected format: {
                    'action_positions': List[int],
                    'action_logprobs': List[List[Any]],  # [token_str, top_logs, entropy, pos]
                    'action_entropies': List[float]
                }
        """
        if not action_logprobs_info:
            return
        
        # Import here to avoid circular dependency
        import utils.prompt_manager.terminal_formatting_utils as tfu
        
        action_positions = action_logprobs_info.get('action_positions', [])
        action_logprobs_list = action_logprobs_info.get('action_logprobs', [])
        action_entropies = action_logprobs_info.get('action_entropies', [])
        
        tfu.cprint("\n[5] Action logprobs info:", tfu.LIGHT_CYAN, bold=True)
        tfu.cprint(f"  Positions: {action_positions}", tfu.LIGHT_BLACK)
        tfu.cprint(f"  Count: {len(action_logprobs_list)}", tfu.LIGHT_BLACK)
        
        for idx, entry in enumerate(action_logprobs_list):
            if len(entry) >= 4:
                token_str, top_logs, entropy, pos = entry[0], entry[1], entry[2], entry[3]
                tfu.cprint(f"  - Action {idx+1} token: '{token_str}' (pos {pos})", tfu.LIGHT_BLACK)
                if entropy is not None:
                    tfu.cprint(f"    entropy: {entropy:.4f}", tfu.LIGHT_BLACK)
                if top_logs:
                    tfu.cprint(f"    top logprobs: {top_logs}", tfu.LIGHT_BLACK)
        
        if action_entropies:
            entropies_str = [round(e, 4) if e is not None else None for e in action_entropies]
            tfu.cprint(f"  Entropies list: {entropies_str}", tfu.LIGHT_BLACK)


# Convenience function
def parse_vlm_response(
    response_text: str,
    required_fields: Optional[list] = None
) -> Dict[str, str]:
    """
    Convenience function for parsing VLM response
    
    Args:
        response_text: Raw text returned by VLM
        required_fields: List of required fields
        
    Returns:
        Parsed dictionary
    """
    processor = VLMResponsePostProcessor(required_fields=required_fields)
    return processor.process(response_text)

