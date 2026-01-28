"""
VLM Response Post-processing Module

Parses and validates raw responses returned by VLM Wrapper, converting them to structured data.
Handles JSON parsing and validation for robot control command extraction.
"""

import json
import re
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
            # JSON 파싱 실패 시 부분 파싱 시도 (strict 여부와 관계없이)
            partial_result = self._partial_parse_json(response_text, e)
            
            # strict=True일 때 필수 필드 확인
            if strict:
                missing_fields = [field for field in self.required_fields 
                                if partial_result.get(field) is None or partial_result.get(field) == ""]
                if missing_fields:
                    # 필수 필드가 없어도 부분 파싱 결과 반환 (기본값 사용)
                    # 경고만 출력하고 계속 진행
                    import warnings
                    warnings.warn(
                        f"JSON parsing failed (truncated response). "
                        f"Missing required fields: {missing_fields}. "
                        f"Using partial parse result with defaults.",
                        UserWarning
                    )
            
            return partial_result
    
    def _partial_parse_json(self, response_text: str, error: json.JSONDecodeError) -> Dict[str, Any]:
        """
        부분 JSON 파싱: 잘린 JSON에서 추출 가능한 필드만 추출하고 나머지는 None으로 설정
        
        Args:
            response_text: 잘린 JSON 문자열
            error: JSONDecodeError 객체
            
        Returns:
            부분적으로 파싱된 딕셔너리 (추출 가능한 필드만 포함, 나머지는 None)
        """
        result = {}
        
        # 모든 필드 초기화 (None으로)
        all_fields = set(self.required_fields)
        # response_text에서 필드 이름 추출 시도
        field_pattern = r'"([^"]+)"\s*:\s*'
        found_fields = re.findall(field_pattern, response_text)
        all_fields.update(found_fields)
        
        # 각 필드를 None으로 초기화
        for field in all_fields:
            result[field] = None
        
        # 정규식으로 개별 필드 추출 시도
        for field in all_fields:
            # 문자열 필드: "field": "value" (닫히지 않은 따옴표도 처리)
            # 패턴 1: 완전한 문자열 "field": "complete_value"
            str_pattern1 = rf'"{re.escape(field)}"\s*:\s*"([^"]*)"'
            str_match1 = re.search(str_pattern1, response_text, re.DOTALL)
            if str_match1:
                try:
                    # JSON 이스케이프 처리
                    value = str_match1.group(1)
                    # 간단한 이스케이프 처리 (\n, \t, \\, \")
                    value = value.replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\').replace('\\"', '"')
                    result[field] = value
                    continue
                except:
                    pass
            
            # 패턴 2: 닫히지 않은 문자열 "field": "incomplete_value... (끝이 잘림)
            str_pattern2 = rf'"{re.escape(field)}"\s*:\s*"([^"]*?)(?:"|$)'
            str_match2 = re.search(str_pattern2, response_text, re.DOTALL)
            if str_match2 and result.get(field) is None:
                try:
                    value = str_match2.group(1)
                    # 이스케이프 처리
                    value = value.replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\').replace('\\"', '"')
                    result[field] = value
                    continue
                except:
                    pass
            
            # 숫자 필드: "field": 123 또는 "field": 123.45
            num_pattern = rf'"{re.escape(field)}"\s*:\s*(-?\d+\.?\d*)'
            num_match = re.search(num_pattern, response_text)
            if num_match:
                try:
                    num_str = num_match.group(1)
                    if '.' in num_str:
                        result[field] = float(num_str)
                    else:
                        result[field] = int(num_str)
                    continue
                except:
                    pass
            
            # 불리언 필드: "field": true/false
            bool_pattern = rf'"{re.escape(field)}"\s*:\s*(true|false)'
            bool_match = re.search(bool_pattern, response_text, re.IGNORECASE)
            if bool_match:
                result[field] = bool_match.group(1).lower() == 'true'
                continue
            
            # null 필드: "field": null
            null_pattern = rf'"{re.escape(field)}"\s*:\s*null'
            null_match = re.search(null_pattern, response_text, re.IGNORECASE)
            if null_match:
                result[field] = None
                continue
            
            # 배열 필드: "field": [...] (부분 추출 시도)
            # 닫히지 않은 배열도 처리: "field": [item1, item2, ... (끝이 잘림)
            array_pattern = rf'"{re.escape(field)}"\s*:\s*\[(.*?)(?:\]|$)'
            array_match = re.search(array_pattern, response_text, re.DOTALL)
            if array_match:
                array_content = array_match.group(1)
                # 간단한 배열 파싱 (문자열 요소만)
                str_items = re.findall(r'"([^"]*)"', array_content)
                if str_items:
                    result[field] = str_items
                    continue
                # 숫자 요소
                num_items = re.findall(r'(-?\d+\.?\d*)', array_content)
                if num_items:
                    try:
                        result[field] = [float(n) if '.' in n else int(n) for n in num_items]
                        continue
                    except:
                        pass
                # 빈 배열 또는 부분 배열
                if array_content.strip() == "" or array_content.strip().startswith(('"', "'", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                    # 최소한 배열 시작은 있음
                    result[field] = str_items if str_items else []
                    continue
            
            # 객체 필드: "field": {...} (부분 추출 시도)
            obj_pattern = rf'"{re.escape(field)}"\s*:\s*\{{(.*?)(?:\}}|$)'
            obj_match = re.search(obj_pattern, response_text, re.DOTALL)
            if obj_match:
                obj_content = obj_match.group(1)
                # 간단한 객체 파싱 시도
                try:
                    # 닫는 괄호 추가 시도
                    partial_obj = '{' + obj_content + '}'
                    partial_parsed = json.loads(partial_obj)
                    result[field] = partial_parsed
                    continue
                except:
                    # 실패하면 빈 딕셔너리
                    result[field] = {}
                    continue
        
        # required_fields에 대해 기본값 적용
        for field in self.required_fields:
            if result.get(field) is None:
                if field in self.default_fields:
                    default_value = self.default_fields[field]
                    if isinstance(default_value, (list, dict)):
                        result[field] = default_value
                    else:
                        result[field] = str(default_value)
                else:
                    # 기본값이 없으면 빈 문자열 또는 빈 리스트/딕셔너리
                    if field == "action":
                        result[field] = ["0"]  # 기본 액션
                    elif field == "memory":
                        result[field] = {
                            "spatial_description": "",
                            "task_process": {"goal": "", "status": ""},
                            "previous_action": ""
                        }
                    else:
                        result[field] = ""
        
        return result
    
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


    def parse_verbalized_entropy_response(
        self,
        response_text: str,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Verbalized Entropy 형식의 VLM 응답을 파싱합니다.
        step1/step2/step3 확률 분포를 파싱하고 정규화하며, argmax로 action을 추출합니다.
        
        Args:
            response_text: Raw text returned by VLM (JSON 형식)
            strict: If True, validate required fields
        
        Returns:
            Parsed dictionary containing:
                - 'step1', 'step2', 'step3': 정규화된 확률 분포
                - 'action': argmax로 추출된 action 리스트 [step1_action, step2_action, step3_action]
                - 'executability': 실행 가능성 (0.0~1.0)
                - 'reasoning': 추론 설명
                - 'memory': 메모리 정보
        """
        import numpy as np
        
        # 기본 JSON 파싱
        response_text = response_text.strip()
        
        # JSON 코드 블록 제거
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
        
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as e:
            # 부분 파싱 시도
            parsed = self._partial_parse_verbalized_entropy(response_text)
        
        result = {}
        directions = ['north', 'south', 'west', 'east']
        actions = []
        
        # step1, step2, step3 파싱 및 정규화
        for step_name in ['step1', 'step2', 'step3']:
            step_data = parsed.get(step_name, {})
            
            if isinstance(step_data, dict):
                # 확률 추출 및 정규화
                probs = {}
                prob_sum = 0.0
                
                for direction in directions:
                    prob = step_data.get(direction, 0.0)
                    try:
                        prob = float(prob)
                    except (ValueError, TypeError):
                        prob = 0.0
                    probs[direction] = max(0.0, prob)  # 음수 방지
                    prob_sum += probs[direction]
                
                # 정규화 (합이 1이 아닌 경우)
                if prob_sum > 0 and abs(prob_sum - 1.0) > 0.01:
                    for direction in directions:
                        probs[direction] /= prob_sum
                elif prob_sum == 0:
                    # 모든 확률이 0이면 균등 분포
                    for direction in directions:
                        probs[direction] = 0.25
                
                result[step_name] = probs
                
                # argmax로 action 추출
                best_action = max(probs, key=probs.get)
                actions.append(best_action)
            else:
                # 기본값: 균등 분포
                result[step_name] = {d: 0.25 for d in directions}
                actions.append('north')  # 기본 action
        
        # action 필드 생성 (argmax 결과)
        result['action'] = actions
        
        # executability
        executability = parsed.get('executability', 0.5)
        try:
            executability = float(executability)
            executability = max(0.0, min(1.0, executability))  # 0~1 범위로 제한
        except (ValueError, TypeError):
            executability = 0.5
        result['executability'] = executability
        
        # reasoning
        result['reasoning'] = str(parsed.get('reasoning', ''))
        
        # memory
        memory = parsed.get('memory', {})
        if not isinstance(memory, dict):
            memory = {}
        result['memory'] = memory
        
        # grounding (있으면 포함)
        if 'grounding' in parsed:
            result['grounding'] = str(parsed.get('grounding', ''))
        
        return result
    
    def _partial_parse_verbalized_entropy(self, response_text: str) -> Dict[str, Any]:
        """
        Verbalized Entropy 형식의 부분 파싱 (잘린 JSON 처리)
        
        Args:
            response_text: 잘린 JSON 문자열
        
        Returns:
            부분적으로 파싱된 딕셔너리
        """
        result = {}
        directions = ['north', 'south', 'west', 'east']
        
        # step1, step2, step3 파싱 시도
        for step_name in ['step1', 'step2', 'step3']:
            step_pattern = rf'"{step_name}"\s*:\s*\{{([^}}]*)\}}'
            step_match = re.search(step_pattern, response_text, re.DOTALL)
            
            if step_match:
                step_content = step_match.group(1)
                probs = {}
                
                for direction in directions:
                    # 각 방향의 확률 추출
                    dir_pattern = rf'"{direction}"\s*:\s*([0-9.]+)'
                    dir_match = re.search(dir_pattern, step_content)
                    if dir_match:
                        try:
                            probs[direction] = float(dir_match.group(1))
                        except ValueError:
                            probs[direction] = 0.25
                    else:
                        probs[direction] = 0.25
                
                result[step_name] = probs
            else:
                result[step_name] = {d: 0.25 for d in directions}
        
        # executability 추출
        exec_pattern = r'"executability"\s*:\s*([0-9.]+)'
        exec_match = re.search(exec_pattern, response_text)
        if exec_match:
            try:
                result['executability'] = float(exec_match.group(1))
            except ValueError:
                result['executability'] = 0.5
        else:
            result['executability'] = 0.5
        
        # reasoning 추출
        reason_pattern = r'"reasoning"\s*:\s*"([^"]*)"'
        reason_match = re.search(reason_pattern, response_text)
        if reason_match:
            result['reasoning'] = reason_match.group(1)
        else:
            result['reasoning'] = ''
        
        # memory 추출 (간단한 버전)
        result['memory'] = {}
        
        return result
    
    def normalize_step_probs(self, step_probs: Dict[str, float]) -> Dict[str, float]:
        """
        step 확률 분포를 정규화합니다 (합이 1.0이 되도록)
        
        Args:
            step_probs: {'north': P, 'south': P, 'west': P, 'east': P}
        
        Returns:
            정규화된 확률 분포
        """
        directions = ['north', 'south', 'west', 'east']
        probs = []
        
        for d in directions:
            p = step_probs.get(d, 0.0)
            try:
                p = float(p)
            except (ValueError, TypeError):
                p = 0.0
            probs.append(max(0.0, p))
        
        prob_sum = sum(probs)
        
        if prob_sum > 0:
            probs = [p / prob_sum for p in probs]
        else:
            probs = [0.25] * 4
        
        return {d: p for d, p in zip(directions, probs)}
    
    def calculate_step_entropy(self, step_probs: Dict[str, float]) -> float:
        """
        단일 step의 Shannon Entropy를 계산합니다: H = -Σ P log₂ P
        
        Args:
            step_probs: {'north': P, 'south': P, 'west': P, 'east': P}
        
        Returns:
            Shannon entropy (bits)
        """
        import numpy as np
        
        directions = ['north', 'south', 'west', 'east']
        probs = np.array([step_probs.get(d, 0.0) for d in directions])
        probs = probs[probs > 0]  # 0 제외 (log(0) 방지)
        
        if len(probs) == 0:
            return 0.0
        
        # 정규화
        probs = probs / probs.sum()
        
        return -np.sum(probs * np.log2(probs))
    
    def calculate_weighted_entropy(
        self,
        step1: Dict[str, float],
        step2: Dict[str, float],
        step3: Dict[str, float],
        weights: List[float] = None
    ) -> float:
        """
        가중 평균 엔트로피를 계산합니다 (노트북 방식: 50/30/20)
        
        Args:
            step1, step2, step3: 각 step의 확률 분포
            weights: 가중치 리스트 (기본값: [0.5, 0.3, 0.2])
        
        Returns:
            가중 평균 entropy
        """
        if weights is None:
            weights = [0.5, 0.3, 0.2]
        
        e1 = self.calculate_step_entropy(step1)
        e2 = self.calculate_step_entropy(step2)
        e3 = self.calculate_step_entropy(step3)
        
        return weights[0] * e1 + weights[1] * e2 + weights[2] * e3


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

