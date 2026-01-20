"""
VLM Response Post-processing Module

Parses and validates raw responses returned by VLM Wrapper, converting them to structured data.
Handles JSON parsing and validation for robot control command extraction.
"""

import json
import re
from typing import Dict, Optional


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
        
        # More robust JSON extraction using regex
        # This will find a JSON object enclosed in ```json ... ``` or ``` ... ```
        json_match = re.search(r"```(?:json)?\s*({.+?})\s*```", response_text, re.DOTALL)
        
        if json_match:
            response_text = json_match.group(1)
        # Fallback for cases where only the JSON object is returned without backticks
        elif not response_text.startswith('{'):
             start_idx = response_text.find('{')
             if start_idx != -1:
                 response_text = response_text[start_idx:]

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

