"""
Verification script for Gemini Backend and Prompt Steering.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.gemini_backend import GeminiBackend
from steering.prompt_steering import get_steering_prompt

def test_prompt_construction():
    print("\n--- Testing Prompt Construction Logic ---")
    
    # Case 1: High Extraversion
    alphas = {"Extraversion": 2.0}
    prompt = get_steering_prompt(alphas)
    print(f"Alphas: {alphas}")
    print(f"Generated Prompt:\n{prompt}")
    assert "highly extraverted" in prompt
    
    # Case 2: Low Agreeableness
    alphas = {"Agreeableness": -1.5}
    prompt = get_steering_prompt(alphas)
    print(f"Alphas: {alphas}")
    print(f"Generated Prompt:\n{prompt}")
    assert "low in agreeableness" in prompt
    
    print("Prompt construction logic verified!\n")

def main():
    test_prompt_construction()

    print("Initializing GeminiBackend...")
    try:
        backend = GeminiBackend(
            model_name="gemini-1.5-flash",
            temperature=0.7,
            alpha_strength=1.0
        )
    except Exception as e:
        print(f"Failed to initialize backend: {e}")
        return

    prompt = "Introduce yourself to the town council."
    
    print(f"\nBase Prompt: {prompt}\n")

    # Test Case 1: High Extraversion
    alphas_high_e = {"Extraversion": 2.0}
    print(f"--- Testing High Extraversion {alphas_high_e} ---")
    res_high = backend.generate(prompt, max_new_tokens=100, alphas=alphas_high_e)
    print(f"Response:\n{res_high.text}\n")

    # Test Case 2: Low Extraversion (Introversion)
    alphas_low_e = {"Extraversion": -2.0}
    print(f"--- Testing Low Extraversion {alphas_low_e} ---")
    res_low = backend.generate(prompt, max_new_tokens=100, alphas=alphas_low_e)
    print(f"Response:\n{res_low.text}\n")

    # Test Case 3: High Agreeableness + High Conscientiousness
    alphas_mix = {"Agreeableness": 1.5, "Conscientiousness": 1.5}
    print(f"--- Testing Mixed Traits {alphas_mix} ---")
    res_mix = backend.generate(prompt, max_new_tokens=100, alphas=alphas_mix)
    print(f"Response:\n{res_mix.text}\n")
    
    print("Verification Complete.")

if __name__ == "__main__":
    main()
