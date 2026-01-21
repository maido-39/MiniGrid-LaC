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
    
    def process_without_logprobs(
        self,
        response_text: str,
        logprobs_metadata: Optional[Dict] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """Process VLM response and return clean JSON without logprobs.
        
        This method processes the response text and returns a clean JSON structure,
        removing any logprobs-related metadata. This is useful when you want to
        use the response as a normal JSON without logprobs information.
        
        Args:
            response_text: Raw text response from the VLM
            logprobs_metadata: Optional logprobs metadata (ignored, kept for API consistency)
            strict: If True, raises ValueError if required fields are missing.
        
        Returns:
            Dictionary containing parsed fields without logprobs information.
        
        Examples:
            >>> processor = VLMResponsePostProcessor(
            ...     required_fields=["action", "reasoning"]
            ... )
            >>> response = '{"action": "move up", "reasoning": "Go north"}'
            >>> result = processor.process_without_logprobs(response)
            >>> # Returns clean JSON without logprobs
        """
        # Simply process as normal JSON (logprobs are not in the text anyway)
        return self.process(response_text, strict=strict)
    
    def process_with_action_logprobs(
        self,
        response_text: str,
        logprobs_metadata: Dict,
        action_field: str = "action",
        strict: bool = True
    ) -> Dict[str, Any]:
        """Process VLM response and wrap logprobs for tokens after action field.
        
        This method processes the response text, extracts the action field value,
        and wraps logprobs information for the tokens that make up the action value
        and all subsequent tokens in the response.
        
        Args:
            response_text: Raw text response from the VLM
            logprobs_metadata: Dictionary containing logprobs information:
                - 'tokens': List of tokens in the response
                - 'token_logprobs': List of log probabilities for each token
                - 'top_logprobs': List of top-k logprobs for each token position
                - 'entropies': List of Shannon entropies for each token position
            action_field: Name of the action field in the JSON (default: "action")
            strict: If True, raises ValueError if required fields are missing.
        
        Returns:
            Dictionary containing parsed fields with logprobs wrapped:
                - All original fields from the JSON
                - 'action_logprobs': Dictionary containing:
                    - 'action_tokens': List of tokens that make up the action value
                    - 'action_token_logprobs': List of logprobs for action tokens
                    - 'action_top_logprobs': List of top-k logprobs for action tokens
                    - 'action_entropies': List of entropies for action tokens
                    - 'action_start_idx': Starting token index of action in response
                - 'remaining_logprobs': Dictionary containing:
                    - 'tokens': List of tokens after action
                    - 'token_logprobs': List of logprobs for remaining tokens
                    - 'top_logprobs': List of top-k logprobs for remaining tokens
                    - 'entropies': List of entropies for remaining tokens
                    - 'start_idx': Starting token index of remaining tokens
        
        Examples:
            >>> processor = VLMResponsePostProcessor(
            ...     required_fields=["action", "reasoning"]
            ... )
            >>> response = '{"action": "move up", "reasoning": "Go north"}'
            >>> logprobs = {
            ...     'tokens': ['{', '"', 'action', '"', ':', '"', 'move', 'up', ...],
            ...     'entropies': [0.5, 0.3, ...]
            ... }
            >>> result = processor.process_with_action_logprobs(
            ...     response, logprobs, action_field="action"
            ... )
            >>> # result['action_logprobs'] contains logprobs for "move up" tokens
            >>> # result['remaining_logprobs'] contains logprobs for remaining tokens
        """
        # First, parse the JSON response
        parsed = self.process(response_text, strict=strict)
        
        # Extract tokens and logprobs from metadata
        tokens = logprobs_metadata.get('tokens', [])
        token_logprobs = logprobs_metadata.get('token_logprobs', [])
        top_logprobs = logprobs_metadata.get('top_logprobs', [])
        entropies = logprobs_metadata.get('entropies', [])
        
        if not tokens:
            # No logprobs available, return parsed result as-is
            return parsed
        
        # Find the action field value in the response text
        action_value = parsed.get(action_field, "")
        if not action_value:
            # No action field, return parsed result with all logprobs as remaining
            parsed['remaining_logprobs'] = {
                'tokens': tokens,
                'token_logprobs': token_logprobs,
                'top_logprobs': top_logprobs,
                'entropies': entropies,
                'start_idx': 0
            }
            return parsed
        
        # Convert action value to string for token matching
        action_str = str(action_value)
        
        # Find where the action value appears in the response text
        # We need to find the token indices that correspond to the action value
        response_lower = response_text.lower()
        action_lower = action_str.lower()
        
        # Try to find action value in response text
        action_start_in_text = response_lower.find(action_lower)
        
        if action_start_in_text == -1:
            # Action value not found in text, try to find it in tokens
            # This is a fallback: search for tokens that match action value
            action_tokens_str = ' '.join(tokens).lower()
            action_start_in_tokens = action_tokens_str.find(action_lower)
            
            if action_start_in_tokens != -1:
                # Find the token index
                # Count tokens before the match
                text_before = action_tokens_str[:action_start_in_tokens]
                token_count_before = len(text_before.split())
                action_start_idx = token_count_before
            else:
                # Cannot find action in tokens, use heuristic:
                # Look for action field name, then find value after it
                action_field_lower = action_field.lower()
                for i, token in enumerate(tokens):
                    if action_field_lower in token.lower():
                        # Found action field, value should be a few tokens after
                        # Skip field name, colon, quotes
                        action_start_idx = min(i + 3, len(tokens))
                        break
                else:
                    action_start_idx = 0
        else:
            # Found action in text, need to map to token indices
            # This is approximate: count characters before action
            chars_before = response_text[:action_start_in_text]
            # Estimate token index (rough approximation)
            # Count spaces and punctuation as token boundaries
            approx_tokens_before = len(chars_before.split())
            action_start_idx = min(approx_tokens_before, len(tokens))
        
        # Find action end: action value length in tokens
        # Approximate: action value length / average token length
        action_token_count = max(1, len(action_str.split()))
        action_end_idx = min(action_start_idx + action_token_count, len(tokens))
        
        # Extract action logprobs
        action_tokens = tokens[action_start_idx:action_end_idx]
        action_token_logprobs = token_logprobs[action_start_idx:action_end_idx] if token_logprobs else []
        action_top_logprobs = top_logprobs[action_start_idx:action_end_idx] if top_logprobs else []
        action_entropies = entropies[action_start_idx:action_end_idx] if entropies else []
        
        # Extract remaining logprobs (after action)
        remaining_tokens = tokens[action_end_idx:]
        remaining_token_logprobs = token_logprobs[action_end_idx:] if token_logprobs else []
        remaining_top_logprobs = top_logprobs[action_end_idx:] if top_logprobs else []
        remaining_entropies = entropies[action_end_idx:] if entropies else []
        
        # Wrap logprobs in result
        parsed['action_logprobs'] = {
            'action_tokens': action_tokens,
            'action_token_logprobs': action_token_logprobs,
            'action_top_logprobs': action_top_logprobs,
            'action_entropies': action_entropies,
            'action_start_idx': action_start_idx,
            'action_end_idx': action_end_idx
        }
        
        parsed['remaining_logprobs'] = {
            'tokens': remaining_tokens,
            'token_logprobs': remaining_token_logprobs,
            'top_logprobs': remaining_top_logprobs,
            'entropies': remaining_entropies,
            'start_idx': action_end_idx
        }
        
        return parsed


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

