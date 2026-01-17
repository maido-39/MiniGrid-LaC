######################################################
#                                                    #
#                         VLM                        #
#                   POST-PROCESSOR                   #
#                                                    #
######################################################


"""
VLM Response Post-Processing Module

Parses and validates the raw response returned by the VLM Wrapper, converting it into structured data.
Handles JSON parsing and validation for extracting robot control commands.
"""




######################################################
#                                                    #
#                      LIBRARIES                     #
#                                                    #
######################################################


import json
from typing import Dict, Optional




######################################################
#                                                    #
#                       CLASS                        #
#                                                    #
######################################################


class VLMResponsePostProcessor:
    """
    VLM Response Postprocessing Class
    
    Parses the raw text returned by VLM and converts it into structured data.
    """
    
    def __init__(self,
                 required_fields: Optional[list] = None,
                 default_fields: Optional[Dict] = None
                ):
        """
        Post-Processor Initialization
        
        Parameters
        ----------
            required_fields: List of required fields (e.g., [“robot_action”, “context”])
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
        Parsing and extracting JSON from response text
        
        Parameters
        ----------
            response_text: Raw text returned by VLM
            strict: True for mandatory field validation, False for flexible handling
            
        Returns
        -------
            Parsed JSON dictionary
            
        Raises
        ------
            ValueError: JSON parsing failure or missing required field
        """
        
        # Text Organization
        response_text = response_text.strip()
        
        # Extract JSON code blocks if present
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
        
        # Attempting JSON parsing
        try:
            parsed = json.loads(response_text)
            
            if not isinstance(parsed, dict):
                raise ValueError(f"The response is not in dictionary format: {type(parsed)}")
            
            # 필수 필드 확인
            if strict:
                for field in self.required_fields:
                    if field not in parsed:
                        raise ValueError(f"Required fields for response '{field}'There is none.")
            
            # Apply default values (retain original type)
            result = {}
            
            # required_fields Processing
            for field in self.required_fields:
                if field in parsed:
                    # Lists and dictionaries retain their original types; everything else is converted to strings.
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
            
            # Includes fields not listed in required_fields (e.g., reasoning)
            for field, value in parsed.items():
                if field not in result:
                    if isinstance(value, (list, dict)):
                        result[field] = value
                    else:
                        result[field] = str(value)
            
            return result
            
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Unable to parse the response as JSON: {e}\n"
                f"Original Response: {response_text[:200]}..."
            )
    
    def extract_robot_action(
        self,
        response_text: str
    ) -> Dict[str, str]:
        """
        Robot Control Command Extraction (robot_action, context format)
        
        Parameters
        ----------
            response_text: Original text returned by VLM
            
        Returns
        -------
            {“robot_action”: str, “context”: str}
        """
        
        return self.parse_json_response(response_text,
                                        strict=True
                                       )
    
    def process(self,
                response_text: str,
                strict: bool = True
               ) -> Dict[str, str]:
        """
        Process response text to return structured data
        
        Args:
            response_text: Raw text returned by VLM
            strict: True for mandatory field validation
            
        Returns:
            Parsed dictionary
        """
        
        return self.parse_json_response(response_text, strict=strict)




######################################################
#                                                    #
#                    CONVENIENCE                     #
#                     FUNCTIONS                      #
#                                                    #
######################################################


def parse_vlm_response(response_text: str,
                       required_fields: Optional[list] = None
                      ) -> Dict[str, str]:
    """
    Convenience function for parsing VLM responses
    
    Parameters
    ----------
        response_text: Raw text returned by VLM
        required_fields: List of required fields
        
    Returns
    -------
        Parsed dictionary
    """
    
    processor = VLMResponsePostProcessor(required_fields=required_fields)
    
    return processor.process(response_text)



