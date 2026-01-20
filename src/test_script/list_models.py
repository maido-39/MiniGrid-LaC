"""
This script lists all available Gemini models for your API key that support the 'generateContent' method.
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def list_gemini_models():
    """
    Lists available Gemini models.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("Error: API key not found.")
        print("Please set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable,")
        print("or add it to your .env file.")
        return

    try:
        genai.configure(api_key=api_key)
        
        print("Available Gemini models that support 'generateContent':")
        models_found = False
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
                models_found = True
        
        if not models_found:
            print("No models found that support 'generateContent'.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    list_gemini_models()
