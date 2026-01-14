"""
VLM Handler System Usage Example

Demonstrates usage of the new handler-based VLM system.
"""

import numpy as np
from PIL import Image

# Method 1: Use existing ChatGPT4oVLMWrapper (maintain compatibility)
from vlm_wrapper import ChatGPT4oVLMWrapper

# Method 2: Use new VLMManager
from vlm import VLMManager

# Method 3: Use handler directly
from ..handlers import OpenAIHandler


def example_1_legacy_wrapper():
    """Example 1: Use existing ChatGPT4oVLMWrapper (maintain compatibility)"""
    print("=" * 60)
    print("Example 1: Use existing ChatGPT4oVLMWrapper")
    print("=" * 60)
    
    # Can use existing method as-is
    wrapper = ChatGPT4oVLMWrapper(
        model="gpt-4o",
        temperature=0.0,
        max_tokens=1000
    )
    
    # Usage is the same
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    print("\n[Input]")
    print(f"  Image: NumPy array (shape: {test_image.shape})")
    print(f"  System Prompt: 'You are a helpful assistant.'")
    print(f"  User Prompt: 'Describe what you see in this image.'")
    print(f"\nCalling VLM...")
    
    try:
        response = wrapper.generate(
            image=test_image,
            system_prompt="You are a helpful assistant.",
            user_prompt="Describe what you see in this image."
        )
        
        print(f"\n[Output]")
        print(f"  Raw response:")
        print(f"  {response[:200]}...")
    except Exception as e:
        print(f"  Error: {e}")
    
    print()


def example_2_vlm_manager():
    """Example 2: Use new VLMManager"""
    print("=" * 60)
    print("Example 2: Use new VLMManager")
    print("=" * 60)
    
    # Create VLMManager
    manager = VLMManager()
    
    # Create and register handler
    manager.create_handler(
        handler_type="openai",
        name="my_openai",
        set_as_default=True,
        model="gpt-4o",
        temperature=0.0,
        max_tokens=1000
    )
    
    print("\nRegistered handlers:", manager.list_handlers())
    
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    print("\n[Input]")
    print(f"  Image: NumPy array (shape: {test_image.shape})")
    print(f"  System Prompt: 'You are a helpful assistant.'")
    print(f"  User Prompt: 'What do you see?'")
    print(f"\nCalling VLM...")
    
    try:
        # Call through manager
        response = manager.generate(
            image=test_image,
            system_prompt="You are a helpful assistant.",
            user_prompt="What do you see?"
        )
        
        print(f"\n[Output]")
        print(f"  Raw response:")
        print(f"  {response[:200]}...")
    except Exception as e:
        print(f"  Error: {e}")
    
    print()


def example_3_multiple_handlers():
    """Example 3: Register and use multiple handlers"""
    print("=" * 60)
    print("Example 3: Register and use multiple handlers")
    print("=" * 60)
    
    manager = VLMManager()
    
    # Register multiple handlers
    manager.create_handler(
        handler_type="openai",
        name="gpt4o",
        model="gpt-4o",
        temperature=0.0,
        max_tokens=1000
    )
    
    manager.create_handler(
        handler_type="openai",
        name="gpt4o_mini",
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=500
    )
    
    print("\nRegistered handlers:", manager.list_handlers())
    
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Use specific handler
    print("\n[Using gpt4o handler]")
    try:
        response1 = manager.generate(
            image=test_image,
            system_prompt="You are a helpful assistant.",
            user_prompt="Describe the image.",
            handler_name="gpt4o"
        )
        print(f"  Response length: {len(response1)} characters")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n[Using gpt4o_mini handler]")
    try:
        response2 = manager.generate(
            image=test_image,
            system_prompt="You are a helpful assistant.",
            user_prompt="Describe the image.",
            handler_name="gpt4o_mini"
        )
        print(f"  Response length: {len(response2)} characters")
    except Exception as e:
        print(f"  Error: {e}")
    
    print()


def example_4_direct_handler():
    """Example 4: Use handler directly"""
    print("=" * 60)
    print("Example 4: Use handler directly")
    print("=" * 60)
    
    # Create handler directly
    handler = OpenAIHandler(
        model="gpt-4o",
        temperature=0.0,
        max_tokens=1000
    )
    
    # Initialize
    handler.initialize()
    
    print(f"\nModel name: {handler.get_model_name()}")
    print(f"Supported image formats: {handler.get_supported_image_formats()}")
    
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    print("\n[Input]")
    print(f"  Image: NumPy array (shape: {test_image.shape})")
    print(f"  System Prompt: 'You are a helpful assistant.'")
    print(f"  User Prompt: 'Analyze this image.'")
    print(f"\nCalling VLM...")
    
    try:
        # Call handler directly
        response = handler.generate(
            image=test_image,
            system_prompt="You are a helpful assistant.",
            user_prompt="Analyze this image."
        )
        
        print(f"\n[Output]")
        print(f"  Raw response:")
        print(f"  {response[:200]}...")
    except Exception as e:
        print(f"  Error: {e}")
    
    print()


def example_5_callable():
    """Example 5: Use as callable object"""
    print("=" * 60)
    print("Example 5: Use as callable object")
    print("=" * 60)
    
    # Use manager as callable object
    manager = VLMManager()
    manager.create_handler(
        handler_type="openai",
        name="default",
        set_as_default=True,
        model="gpt-4o",
        temperature=0.0,
        max_tokens=1000
    )
    
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    print("\n[Input]")
    print(f"  Image: NumPy array (shape: {test_image.shape})")
    print(f"  System Prompt: 'You are a helpful assistant.'")
    print(f"  User Prompt: 'What is this?'")
    print(f"\nCalling VLM...")
    
    try:
        # Call like a function
        response = manager(
            image=test_image,
            system_prompt="You are a helpful assistant.",
            user_prompt="What is this?"
        )
        
        print(f"\n[Output]")
        print(f"  Raw response:")
        print(f"  {response[:200]}...")
    except Exception as e:
        print(f"  Error: {e}")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VLM Handler System Usage Examples")
    print("=" * 60 + "\n")
    
    # Run examples (each example can be run independently)
    try:
        example_1_legacy_wrapper()
    except Exception as e:
        print(f"❌ Error running example 1: {e}\n")
        import traceback
        traceback.print_exc()
    
    try:
        example_2_vlm_manager()
    except Exception as e:
        print(f"❌ Error running example 2: {e}\n")
        import traceback
        traceback.print_exc()
    
    try:
        example_3_multiple_handlers()
    except Exception as e:
        print(f"❌ Error running example 3: {e}\n")
        import traceback
        traceback.print_exc()
    
    try:
        example_4_direct_handler()
    except Exception as e:
        print(f"❌ Error running example 4: {e}\n")
        import traceback
        traceback.print_exc()
    
    try:
        example_5_callable()
    except Exception as e:
        print(f"❌ Error running example 5: {e}\n")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
    print("Example execution completed")
    print("=" * 60)

