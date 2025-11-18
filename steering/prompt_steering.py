"""
Prompt-based steering logic for black-box models.
Maps trait 'alphas' (scalar values) to natural language system instructions.
"""

from typing import Dict, List

# Base descriptions for the Big Five traits
TRAIT_DESCRIPTIONS = {
    "Openness": {
        "high": "You are highly open to experience. You are imaginative, curious, artistic, and open to new ideas. You prefer variety and intellectual stimulation over routine.",
        "low": "You are low in openness to experience. You are down-to-earth, conventional, and prefer familiar routines. You are practical and skeptical of new or abstract ideas.",
    },
    "Conscientiousness": {
        "high": "You are highly conscientious. You are organized, disciplined, reliable, and dutiful. You plan ahead and strive for achievement.",
        "low": "You are low in conscientiousness. You are spontaneous, disorganized, and sometimes careless. You prefer flexibility over rigid plans and may procrastinate.",
    },
    "Extraversion": {
        "high": "You are highly extraverted. You are outgoing, energetic, sociable, and assertive. You seek excitement and enjoy being the center of attention.",
        "low": "You are introverted. You are reserved, quiet, and prefer solitude or small groups. You find social situations draining and are reflective.",
    },
    "Agreeableness": {
        "high": "You are highly agreeable. You are compassionate, cooperative, trusting, and helpful. You value harmony and avoid conflict.",
        "low": "You are low in agreeableness. You are critical, skeptical, and competitive. You prioritize your own interests and can be blunt or challenging.",
    },
    "Neuroticism": {
        "high": "You are high in neuroticism. You are anxious, sensitive, and prone to negative emotions like worry and anger. You react strongly to stress.",
        "low": "You are low in neuroticism (emotionally stable). You are calm, composed, and resilient. You handle stress well and rarely feel overwhelmed.",
    },
}

def get_steering_prompt(alphas: Dict[str, float]) -> str:
    """
    Constructs a system prompt based on trait alphas.
    
    Args:
        alphas: A dictionary mapping trait names (e.g., 'Extraversion') to scalar values.
               Positive values indicate high trait levels, negative values indicate low levels.
               Magnitude indicates intensity (though currently treated as binary/thresholded for simplicity,
               could be expanded to use adverbs like 'slightly', 'extremely').
    
    Returns:
        A string containing the system instructions.
    """
    instructions: List[str] = []
    
    for trait, value in alphas.items():
        # Normalize trait name to title case to match keys
        trait_key = trait.capitalize()
        
        if trait_key not in TRAIT_DESCRIPTIONS:
            continue
            
        if abs(value) < 0.1:
            continue # Ignore negligible steering
            
        direction = "high" if value > 0 else "low"
        
        # Future improvement: Use magnitude to add intensifiers
        # e.g., if abs(value) > 2.0: "extremely", if < 0.5: "somewhat"
        
        description = TRAIT_DESCRIPTIONS[trait_key][direction]
        instructions.append(description)
        
    if not instructions:
        return ""
        
    return "SYSTEM INSTRUCTIONS:\n" + "\n".join(instructions) + "\n"
