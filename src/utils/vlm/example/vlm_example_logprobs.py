"""
VLM Logprobs Usage Example

Demonstrates usage of Vertex AI logprobs feature with Gemini models.
This example shows how to:
1. Initialize VLMWrapper with Vertex AI credentials
2. Generate responses with logprobs
3. Process logprobs data using postprocessor
4. Extract action-specific logprobs
"""

import sys
import os
from pathlib import Path

# Add src directory to path for imports when running as script
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np

# Import VLM components
try:
    from utils.vlm.vlm_wrapper import VLMWrapper
    from utils.vlm.vlm_postprocessor import VLMResponsePostProcessor
except ImportError:
    # Fallback for relative import if running as module
    try:
        from ..vlm_wrapper import VLMWrapper
        from ..vlm_postprocessor import VLMResponsePostProcessor
    except ImportError:
        print("Error: Could not import VLM modules")
        sys.exit(1)


def example_1_basic_logprobs():
    """Example 1: Basic logprobs usage with Vertex AI"""
    print("="*80)
    print("Example 1: Basic logprobs usage with Vertex AI")
    print("="*80)
    
    # Vertex AI credentials setup
    # Option 1: Use environment variables
    # export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
    # export GOOGLE_CLOUD_PROJECT="your-project-id"
    # export GOOGLE_CLOUD_LOCATION="us-central1"
    
    # Option 2: Pass directly
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    if not credentials_path or not project_id:
        print("[SKIP] Vertex AI credentials not configured.")
        print("Set environment variables:")
        print("  - GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json")
        print("  - GOOGLE_CLOUD_PROJECT=your-project-id")
        print("  - GOOGLE_CLOUD_LOCATION=us-central1 (optional)")
        return
    
    try:
        # Initialize wrapper with Vertex AI
        wrapper = VLMWrapper(
            model="gemini-2.5-flash-vertex",  # or "gemini-2.5-flash-logprobs"
            logprobs=5,  # Get top-5 logprobs for each token
            credentials=credentials_path,
            project_id=project_id,
            location=location,
            temperature=0.0,
            max_tokens=1000
        )
        
        # Generate response with logprobs
        print("\n[1] Generating response with logprobs...")
        response, logprobs_metadata = wrapper.generate_with_logprobs(
            system_prompt="You are a helpful assistant.",
            user_prompt="What is the capital of France? Answer in one word.",
            debug=True
        )
        
        print(f"\n[2] Response: {response}")
        print(f"\n[3] Number of tokens: {len(logprobs_metadata.get('tokens', []))}")
        
        if 'tokens' in logprobs_metadata:
            print(f"\n[4] Tokens: {logprobs_metadata['tokens'][:20]}...")  # First 20 tokens
            if 'entropies' in logprobs_metadata:
                avg_entropy = np.mean(logprobs_metadata['entropies'])
                print(f"[5] Average entropy: {avg_entropy:.4f} bits")
            
            # Display top-k logprobs for each token
            if 'top_logprobs' in logprobs_metadata and logprobs_metadata['top_logprobs']:
                # Get k value from first non-empty top_logprobs
                k_value = 0
                for top_k in logprobs_metadata['top_logprobs']:
                    if top_k:
                        k_value = len(top_k)
                        break
                
                print(f"\n[6] Top-{k_value} Logprobs for each token:")
                for i, (token, top_k) in enumerate(zip(
                    logprobs_metadata['tokens'],
                    logprobs_metadata['top_logprobs']
                )):
                    if top_k:
                        print(f"\n  Token {i} ('{token}'):")
                        # Calculate probabilities from logprobs
                        for j, candidate in enumerate(top_k):
                            logprob = candidate.get('log_probability', 0)
                            prob = np.exp(logprob)
                            cand_token = candidate.get('token', '')
                            print(f"    {j+1}. '{cand_token}': prob={prob:.6f} (logprob={logprob:.4f})")
                    else:
                        print(f"\n  Token {i} ('{token}'): No top candidates available")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


def example_2_action_logprobs():
    """Example 2: Extract logprobs for action field in JSON response"""
    print("\n" + "="*80)
    print("Example 2: Extract logprobs for action field in JSON response")
    print("="*80)
    
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    if not credentials_path or not project_id:
        print("[SKIP] Vertex AI credentials not configured.")
        return
    
    try:
        # Initialize wrapper
        wrapper = VLMWrapper(
            model="gemini-2.5-flash-vertex",
            logprobs=5,
            credentials=credentials_path,
            project_id=project_id,
            location=location,
            temperature=0.0,
            max_tokens=2000
        )
        
        # System prompt for robot control
        system_prompt = """You are a robot controller. 
Respond with JSON format containing:
- action: The action to take (e.g., "move up", "pickup", "drop")
- reasoning: Brief explanation of why this action was chosen
"""
        
        user_prompt = """Based on the current situation, what action should the robot take?
Respond in JSON format:
{
  "action": "move up",
  "reasoning": "The goal is to the north"
}
"""
        
        print("\n[1] Generating response with logprobs...")
        response, logprobs_metadata = wrapper.generate_with_logprobs(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            debug=False
        )
        
        print(f"\n[2] Response:\n{response}")
        
        # Process with postprocessor
        print("\n[3] Processing with postprocessor...")
        processor = VLMResponsePostProcessor(
            required_fields=["action", "reasoning"]
        )
        
        # Option A: Get clean JSON without logprobs
        print("\n[4] Option A: Clean JSON (without logprobs)")
        parsed_clean = processor.process_without_logprobs(
            response,
            logprobs_metadata
        )
        print(f"  Action: {parsed_clean.get('action')}")
        print(f"  Reasoning: {parsed_clean.get('reasoning')}")
        
        # Option B: Get JSON with action logprobs wrapped
        print("\n[5] Option B: JSON with action logprobs wrapped")
        parsed_with_logprobs = processor.process_with_action_logprobs(
            response,
            logprobs_metadata,
            action_field="action"
        )
        print(f"  Action: {parsed_with_logprobs.get('action')}")
        print(f"  Reasoning: {parsed_with_logprobs.get('reasoning')}")
        
        # Display action logprobs
        if 'action_logprobs' in parsed_with_logprobs:
            action_logprobs = parsed_with_logprobs['action_logprobs']
            print(f"\n[6] Action Logprobs:")
            print(f"  Action tokens: {action_logprobs.get('action_tokens', [])}")
            print(f"  Number of action tokens: {len(action_logprobs.get('action_tokens', []))}")
            if action_logprobs.get('action_entropies'):
                avg_entropy = np.mean(action_logprobs['action_entropies'])
                print(f"  Average entropy for action: {avg_entropy:.4f} bits")
        
        # Display remaining logprobs
        if 'remaining_logprobs' in parsed_with_logprobs:
            remaining = parsed_with_logprobs['remaining_logprobs']
            print(f"\n[7] Remaining Logprobs:")
            print(f"  Number of remaining tokens: {len(remaining.get('tokens', []))}")
            if remaining.get('entropies'):
                avg_entropy = np.mean(remaining['entropies'])
                print(f"  Average entropy for remaining: {avg_entropy:.4f} bits")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


def example_3_entropy_analysis():
    """Example 3: Analyze token entropy for uncertainty estimation"""
    print("\n" + "="*80)
    print("Example 3: Analyze token entropy for uncertainty estimation")
    print("="*80)
    
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    if not credentials_path or not project_id:
        print("[SKIP] Vertex AI credentials not configured.")
        return
    
    try:
        wrapper = VLMWrapper(
            model="gemini-2.5-flash-vertex",
            logprobs=5,
            credentials=credentials_path,
            project_id=project_id,
            location=location,
            temperature=0.0,
            max_tokens=1000
        )
        
        # Generate response
        response, logprobs_metadata = wrapper.generate_with_logprobs(
            system_prompt="You are a helpful assistant.",
            user_prompt="What is 2+2? Answer with just the number.",
            debug=False
        )
        
        print(f"\n[1] Response: {response}")
        
        # Analyze entropy
        if 'tokens' in logprobs_metadata and 'entropies' in logprobs_metadata:
            tokens = logprobs_metadata['tokens']
            entropies = logprobs_metadata['entropies']
            
            print(f"\n[2] Token-wise Entropy Analysis:")
            print(f"  Total tokens: {len(tokens)}")
            print(f"  Average entropy: {np.mean(entropies):.4f} bits")
            print(f"  Max entropy: {np.max(entropies):.4f} bits (token: '{tokens[np.argmax(entropies)]}')")
            print(f"  Min entropy: {np.min(entropies):.4f} bits (token: '{tokens[np.argmin(entropies)]}')")
            
            # High uncertainty tokens (entropy > threshold)
            threshold = np.mean(entropies) + np.std(entropies)
            high_uncertainty = [(t, e) for t, e in zip(tokens, entropies) if e > threshold]
            if high_uncertainty:
                print(f"\n[3] High Uncertainty Tokens (entropy > {threshold:.4f}):")
                for token, entropy in high_uncertainty[:10]:  # Show first 10
                    print(f"  '{token}': {entropy:.4f} bits")
        
        # Show top logprobs for first few tokens
        if 'top_logprobs' in logprobs_metadata and logprobs_metadata['top_logprobs']:
            print(f"\n[4] Top-5 Logprobs for First 3 Tokens:")
            for i, top_k in enumerate(logprobs_metadata['top_logprobs'][:3]):
                if top_k:
                    print(f"  Token position {i}:")
                    for j, candidate in enumerate(top_k[:5]):
                        prob = np.exp(candidate.get('log_probability', 0))
                        print(f"    {j+1}. '{candidate.get('token', '')}': {prob:.4f} (logprob: {candidate.get('log_probability', 0):.4f})")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


def example_4_vlm_processor():
    """Example 4: Using VLMProcessor with logprobs"""
    print("\n" + "="*80)
    print("Example 4: Using VLMProcessor with logprobs")
    print("="*80)
    
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    if not credentials_path or not project_id:
        print("[SKIP] Vertex AI credentials not configured.")
        return
    
    try:
        from utils.vlm.vlm_processor import VLMProcessor
        
        # Initialize processor with logprobs
        processor = VLMProcessor(
            model="gemini-2.5-flash-vertex",
            logprobs=5,
            credentials=credentials_path,
            project_id=project_id,
            location=location,
            temperature=0.0,
            max_tokens=2000,
            debug=False
        )
        
        # Create dummy image (or use real image)
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        system_prompt = """You are a robot controller. 
Respond with JSON format containing:
- action: The action to take (e.g., ["0"] for move up, ["1"] for move down)
- reasoning: Brief explanation
- grounding: Grounding information
- memory: Memory structure
"""
        
        user_prompt = """What action should the robot take? Respond in JSON format."""
        
        print("\n[1] Requesting with logprobs...")
        response, logprobs_metadata = processor.requester_with_logprobs(
            image=dummy_image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            debug=False
        )
        
        print(f"\n[2] Response:\n{response}")
        
        print("\n[3] Parsing with action logprobs...")
        parsed = processor.parser_action_with_logprobs(
            response,
            logprobs_metadata,
            action_field="action",
            remove_logprobs=False
        )
        
        print(f"\n[4] Parsed result:")
        print(f"  Action: {parsed.get('action')}")
        print(f"  Reasoning: {parsed.get('reasoning')}")
        
        if 'action_logprobs' in parsed:
            action_logprobs = parsed['action_logprobs']
            print(f"\n[5] Action Logprobs:")
            print(f"  Action tokens: {action_logprobs.get('action_tokens', [])}")
            if action_logprobs.get('action_entropies'):
                avg_entropy = np.mean(action_logprobs['action_entropies'])
                print(f"  Average entropy: {avg_entropy:.4f} bits")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("VLM Logprobs Usage Examples")
    print("="*80)
    print("\nNote: This example requires Vertex AI credentials.")
    print("Set environment variables:")
    print("  - GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json")
    print("  - GOOGLE_CLOUD_PROJECT=your-project-id")
    print("  - GOOGLE_CLOUD_LOCATION=us-central1 (optional)")
    print("\n" + "="*80)
    
    # Run examples
    example_1_basic_logprobs()
    example_2_action_logprobs()
    example_3_entropy_analysis()
    example_4_vlm_processor()
    
    print("\n" + "="*80)
    print("Example execution completed")
    print("="*80)
