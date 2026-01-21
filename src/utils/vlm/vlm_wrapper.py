"""
VLM (Vision Language Model) Python Wrapper

This module provides a Wrapper class for easy use of various VLM models (OpenAI GPT-4o, Gemini, etc.).
Handles only VLM calls and basic input/output processing.

Key features:
- Image processing and encoding
- System Prompt, User Prompt input
- API calls and parameter specification
- Returns raw response messages
- Automatic handler selection based on model name

This module internally uses the new handler system (vlm.handlers).
The existing API is maintained for compatibility.
"""

import time
from typing import Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image

# Import new handler system
from .handlers import OpenAIHandler
from .vlm_manager import VLMManager

# Import GeminiHandler (always check, will raise ImportError if not available)
from .handlers.gemini_handler import GeminiHandler

# Export VLMManager for backward compatibility
__all__ = ["VLMWrapper", "VLMManager", "ChatGPT4oVLMWrapper"]  # ChatGPT4oVLMWrapper is alias for backward compatibility


class VLMWrapper:
    """
    Vision Language Model Wrapper
    
    Receives image and text prompts, calls VLM API, and returns raw response.
    Post-processing (parsing, validation, etc.) should be handled in a separate module.
    
    This class automatically selects the appropriate handler based on model name:
    - Models starting with "gemini": Uses GeminiHandler
    - Other models (gpt-4o, gpt-4, etc.): Uses OpenAIHandler
    
    Supports multiple VLM providers:
    - OpenAI: gpt-4o, gpt-4o-mini, gpt-4, gpt-4-turbo, etc.
    - Google Gemini: gemini-2.5-flash, gemini-1.5-pro, gemini-1.5-flash, etc.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        thinking_budget: Optional[int] = None,
        vertexai: bool = False,
        credentials: Optional[Union[str, object]] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        logprobs: Optional[int] = None
    ):
        """
        Initialize wrapper
        
        Args:
            api_key: API key. If None, automatically load from environment variable:
                - For OpenAI models: OPENAI_API_KEY
                - For Gemini models: GEMINI_API_KEY or GOOGLE_API_KEY
            model: Model name to use (default: "gpt-4o")
                - OpenAI models: "gpt-4o", "gpt-4o-mini", "gpt-4", etc.
                - Gemini models: "gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash", etc.
                - Vertex AI models: "gemini-2.5-flash-vertex", "gemini-2.5-flash-logprobs"
            temperature: Generation temperature (default: 0.0)
            max_tokens: Maximum number of tokens (default: 1000)
            thinking_budget: Thinking budget for Gemini 2.5 Flash model (default: None)
                - None: Use default thinking (enabled by default for gemini-2.5-flash)
                - 0: Disable thinking (faster, lower cost)
                - Positive integer: Set thinking budget in tokens
                - Note: Only supported for gemini-2.5-flash model
            vertexai: If True, use Vertex AI instead of Gemini API (default: False)
                - Only for Gemini models
                - Requires credentials, project_id, and location
            credentials: Service account credentials for Vertex AI
                - Can be a path to JSON key file (str) or credentials object
            project_id: Google Cloud project ID for Vertex AI
            location: Google Cloud location for Vertex AI (default: "us-central1")
            logprobs: Number of top logprobs to return (default: None, disabled)
                - Only supported with Vertex AI
                - Recommended: 5
        """
        # Select handler based on model name
        model_lower = model.lower() if model else ""
        
        if model_lower.startswith("gemini"):
            # Use GeminiHandler for Gemini models
            self._handler = GeminiHandler(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking_budget=thinking_budget,
                vertexai=vertexai,
                credentials=credentials,
                project_id=project_id,
                location=location,
                logprobs=logprobs
            )
        else:
            # Use OpenAIHandler for OpenAI models (default)
            self._handler = OpenAIHandler(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        self._handler.initialize()
        
        # Maintain attributes for compatibility
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.is_gpt5_model = model.startswith("gpt-5") if model else False
    
    def _encode_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> str:
        """
        Encode image to base64 (maintained for compatibility)
        
        Args:
            image: Image path (str/Path), numpy array, or PIL Image
            
        Returns:
            base64 encoded image string
        """
        return self._handler.encode_image(image)
    
    def generate(
        self,
        image: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        system_prompt: str = "",
        user_prompt: str = "",
        debug: bool = False
    ) -> str:
        """Generate a response from the VLM using image and text prompts.
        
        Sends the provided image and prompts to the VLM API
        and returns the raw text response. This method handles image encoding
        and API communication automatically.
        
        Args:
            image: Input image for vision analysis. Can be:
                - str: Path to image file (e.g., "path/to/image.png")
                - Path: Path object to image file
                - numpy.ndarray: RGB image array of shape (H, W, 3) with dtype uint8
                - PIL.Image: PIL Image object
                - None: No image (text-only request)
                Defaults to None.
            system_prompt: System-level prompt that defines the assistant's
                behavior and context. This sets the overall role and guidelines
                for the VLM. Defaults to "".
            user_prompt: User-level prompt containing the specific request
                or question. This is the main input that the VLM will respond to.
                Defaults to "".
            debug: If True, print detailed debug information about the response.
                Defaults to False.
        
        Returns:
            str: Raw text response from the VLM. This is the unprocessed response
                and may need parsing depending on the use case.
        
        Raises:
            RuntimeError: If the API call fails (network error, API error, etc.).
            TypeError: If the image type is not supported.
        
        Examples:
            >>> wrapper = VLMWrapper()
            >>> 
            >>> # Text-only request
            >>> response = wrapper.generate(
            ...     system_prompt="You are a helpful assistant.",
            ...     user_prompt="What is the capital of France?"
            ... )
            >>> 
            >>> # Image + text request
            >>> import numpy as np
            >>> image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            >>> response = wrapper.generate(
            ...     image=image,
            ...     system_prompt="You are a robot controller.",
            ...     user_prompt="What action should I take?"
            ... )
            >>> 
            >>> # Using file path
            >>> response = wrapper.generate(
            ...     image="path/to/image.png",
            ...     system_prompt="Analyze this image.",
            ...     user_prompt="Describe what you see."
            ... )
            
            >>> # With debug output
            >>> response = wrapper.generate(
            ...     image=image,
            ...     system_prompt="You are a robot controller.",
            ...     user_prompt="What action should I take?",
            ...     debug=True
            ... )
        
        Note:
            The response is returned as raw text. For structured responses
            (e.g., JSON), use VLMResponsePostProcessor to parse the response.
        """
        # Measure inference time
        start_time = time.time()
        
        # Call handler with metadata if debug is enabled
        if debug:
            result = self._handler.generate(image, system_prompt, user_prompt, return_metadata=True)
            if isinstance(result, tuple):
                response, metadata = result
            else:
                response = result
                metadata = {}
        else:
            response = self._handler.generate(image, system_prompt, user_prompt, return_metadata=False)
            metadata = {}
        
        inference_time = time.time() - start_time
        
        # Debug output
        if debug:
            print("\n" + "="*80)
            print("[DEBUG] RAW VLM RESPONSE:")
            print("="*80)
            print(response)
            print("\n" + "="*80)
            print("[DEBUG] RESPONSE ANALYSIS:")
            print("="*80)
            print(f"  Length: {len(response)} characters")
            print(f"  First 100 chars: {response[:100]}")
            print(f"  Last 100 chars: {response[-100:]}")
            print(f"  Contains '```json': {'```json' in response}")
            print(f"  Number of '```': {response.count('```')}")
            print(f"  Number of '{{': {response.count('{')}")
            print(f"  Number of '}}': {response.count('}')}")
            if "```json" in response:
                start = response.find("```json")
                end = response.find("```", start + 7)
                if end != -1:
                    json_part = response[start+7:end].strip()
                    print(f"  Extracted JSON length: {len(json_part)}")
                    print(f"  Extracted JSON (full):")
                    print(json_part)
                else:
                    print(f"  ⚠️  Found ```json but NO closing ```")
                    json_part = response[start+7:].strip()
                    print(f"  Extracted JSON (no closing, full):")
                    print(json_part)
            print("\n" + "="*80)
            print("[DEBUG] API METADATA:")
            print("="*80)
            print(f"  Inference Time: {inference_time:.3f} seconds")
            if metadata:
                if metadata.get('input_tokens') is not None:
                    print(f"  Input Tokens: {metadata['input_tokens']}")
                if metadata.get('output_tokens') is not None:
                    print(f"  Output Tokens: {metadata['output_tokens']}")
                if metadata.get('total_tokens') is not None:
                    print(f"  Total Tokens: {metadata['total_tokens']}")
                if metadata.get('thinking_tokens') is not None:
                    print(f"  Thinking Tokens: {metadata['thinking_tokens']}")
                if metadata.get('thinking_content'):
                    print(f"  Thinking Content (full):")
                    print(metadata['thinking_content'])
                elif metadata.get('thinking_content') is None and metadata.get('thinking_tokens') is None:
                    print(f"  Thinking: Not available (may require thinking_config)")
            else:
                print(f"  Token information: Not available")
            print("="*80 + "\n")
        
        return response
    
    def __call__(
        self,
        image: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        system_prompt: str = "",
        user_prompt: str = "",
        debug: bool = False
    ) -> str:
        """
        Use as callable object (for convenience)
        
        Args:
            image: Input image (if None, send text only without image)
            system_prompt: System prompt
            user_prompt: User prompt
            debug: If True, print detailed debug information about the response.
            
        Returns:
            Raw response text (str)
        """
        return self.generate(image, system_prompt, user_prompt, debug=debug)

    def generate_with_logprobs(
        self,
        image: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        system_prompt: str = "",
        user_prompt: str = "",
        debug: bool = False
    ) -> tuple:
        """Generate a response with logprobs from the VLM.
        
        This method is specifically for Vertex AI models that support logprobs.
        It returns both the response text and logprobs metadata.
        
        Args:
            image: Input image for vision analysis. Can be:
                - str: Path to image file
                - Path: Path object to image file
                - numpy.ndarray: RGB image array of shape (H, W, 3) with dtype uint8
                - PIL.Image: PIL Image object
                - None: No image (text-only request)
            system_prompt: System-level prompt that defines the assistant's behavior
            user_prompt: User-level prompt containing the specific request
            debug: If True, print detailed debug information about the response.
        
        Returns:
            tuple: (response_text: str, logprobs_metadata: dict)
                - response_text: Raw text response from the VLM
                - logprobs_metadata: Dictionary containing:
                    - 'logprobs_result': The logprobs result object from the API
                    - 'tokens': List of tokens in the response
                    - 'token_logprobs': List of log probabilities for each token
                    - 'top_logprobs': List of top-k logprobs for each token position
                    - 'entropies': List of Shannon entropies for each token position
        
        Raises:
            RuntimeError: If the API call fails or if logprobs are not available.
            ValueError: If the handler doesn't support logprobs (not Vertex AI).
        
        Examples:
            >>> wrapper = VLMWrapper(
            ...     model="gemini-2.5-flash-vertex",
            ...     logprobs=5,
            ...     credentials="/path/to/key.json",
            ...     project_id="my-project"
            ... )
            >>> response, logprobs = wrapper.generate_with_logprobs(
            ...     system_prompt="You are a helpful assistant.",
            ...     user_prompt="What is the capital of France?"
            ... )
            >>> print(f"Response: {response}")
            >>> print(f"Tokens: {logprobs['tokens']}")
            >>> print(f"Entropies: {logprobs['entropies']}")
        """
        # Check if handler supports logprobs
        if not hasattr(self._handler, 'vertexai') or not self._handler.vertexai:
            raise ValueError(
                "logprobs are only available with Vertex AI. "
                "Use model='gemini-2.5-flash-vertex' or set vertexai=True, "
                "and provide credentials, project_id, and logprobs parameter."
            )
        
        # Measure inference time
        start_time = time.time()
        
        # Call handler with metadata (logprobs will be included)
        result = self._handler.generate(
            image, system_prompt, user_prompt,
            return_metadata=True
        )
        
        if isinstance(result, tuple):
            response, metadata = result
        else:
            response = result
            metadata = {}
        
        inference_time = time.time() - start_time
        
        # Extract and process logprobs
        logprobs_metadata = {}
        
        if 'logprobs_result' in metadata:
            logprobs_result = metadata['logprobs_result']
            logprobs_metadata['logprobs_result'] = logprobs_result
            
            # Extract tokens and logprobs
            import numpy as np
            
            tokens = []
            token_logprobs = []
            top_logprobs = []
            entropies = []
            
            if hasattr(logprobs_result, 'chosen_candidates'):
                for i, chosen in enumerate(logprobs_result.chosen_candidates):
                    tokens.append(chosen.token)
                    token_logprobs.append(chosen.log_probability)
                    
                    # Get top candidates for this position
                    if (hasattr(logprobs_result, 'top_candidates') and 
                        i < len(logprobs_result.top_candidates)):
                        top_candidates = logprobs_result.top_candidates[i]
                        top_k = []
                        if hasattr(top_candidates, 'candidates'):
                            for cand in top_candidates.candidates:
                                top_k.append({
                                    'token': cand.token,
                                    'log_probability': cand.log_probability
                                })
                        top_logprobs.append(top_k)
                        
                        # Calculate Shannon entropy
                        if top_k:
                            probs = [np.exp(c['log_probability']) for c in top_k]
                            probs_sum = np.sum(probs)
                            if probs_sum > 0:
                                probs = [p / probs_sum for p in probs]
                                entropy = -np.sum([p*np.log2(p) if p > 0 else 0 for p in probs])
                                entropies.append(entropy)
                            else:
                                entropies.append(0.0)
                        else:
                            entropies.append(0.0)
                    else:
                        top_logprobs.append([])
                        entropies.append(0.0)
            
            logprobs_metadata['tokens'] = tokens
            logprobs_metadata['token_logprobs'] = token_logprobs
            logprobs_metadata['top_logprobs'] = top_logprobs
            logprobs_metadata['entropies'] = entropies
        
        # Add other metadata
        logprobs_metadata['inference_time'] = inference_time
        logprobs_metadata.update({k: v for k, v in metadata.items() if k != 'logprobs_result'})
        
        # Debug output
        if debug:
            print("\n" + "="*80)
            print("[DEBUG] RAW VLM RESPONSE (with logprobs):")
            print("="*80)
            print(response)
            print("\n" + "="*80)
            print("[DEBUG] LOGPROBS METADATA:")
            print("="*80)
            print(f"  Inference Time: {inference_time:.3f} seconds")
            
            if 'tokens' in logprobs_metadata:
                print(f"  Number of tokens: {len(logprobs_metadata['tokens'])}")
                SHOW_RAW_TOKENSLOGPROBS = False
                if SHOW_RAW_TOKENSLOGPROBS and 'entropies' in logprobs_metadata:
                    print(f"  Tokens: {logprobs_metadata['tokens']}")
                    print(f"  Entropies: {logprobs_metadata['entropies']}")
                    if logprobs_metadata['entropies']:
                        print(f"  Average entropy: {np.mean(logprobs_metadata['entropies']):.4f}")
                
                # Display top-k logprobs (can be toggled via SHOW_TOP_K_LOGPROBS flag)
                # Set to False to disable verbose top-k logprobs output
                SHOW_TOP_K_LOGPROBS = False
                if SHOW_TOP_K_LOGPROBS and 'top_logprobs' in logprobs_metadata and logprobs_metadata['top_logprobs']:
                    print(f"\n  Top-k Logprobs for each token:")
                    for i, (token, top_k) in enumerate(zip(
                        logprobs_metadata['tokens'],
                        logprobs_metadata['top_logprobs']
                    )):
                        if top_k:
                            print(f"\n    Token {i} ('{token}'):")
                            for j, candidate in enumerate(top_k):
                                logprob = candidate.get('log_probability', 0)
                                prob = np.exp(logprob)
                                cand_token = candidate.get('token', '')
                                print(f"      {j+1}. '{cand_token}': prob={prob:.6f} (logprob={logprob:.4f})")
                        else:
                            print(f"    Token {i} ('{token}'): No top candidates")
            if metadata:
                if metadata.get('input_tokens') is not None:
                    print(f"  Input Tokens: {metadata['input_tokens']}")
                if metadata.get('output_tokens') is not None:
                    print(f"  Output Tokens: {metadata['output_tokens']}")
                if metadata.get('total_tokens') is not None:
                    print(f"  Total Tokens: {metadata['total_tokens']}")
            print("="*80 + "\n")
        
        return response, logprobs_metadata


# Backward compatibility alias
ChatGPT4oVLMWrapper = VLMWrapper

# Usage example
if __name__ == "__main__":
    # Initialize wrapper
    wrapper = VLMWrapper()
    
    # Example usage
    response = wrapper.generate(
        # image="path/to/image.png",
        system_prompt="You are a helpful assistant.",
        user_prompt="describe the miku's characteristics"
    )
    print(response)

