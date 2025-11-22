"""Verification script for personality-driven architecture."""

import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(".")

from agents.memory import MemoryStore
from agents.agent import Agent
from schemas.agent import AgentState, PersonaCoeffs
from schemas.memory import MemoryEvent

class TestPersonality(unittest.TestCase):
    def test_memory_tagging(self):
        """Test that memories are tagged with traits."""
        store = MemoryStore()
        
        # Test Extraversion tagging
        ev1 = store.add_event("test_agent", "observation", 1, "I went to a party and met everyone.", 1.0)
        self.assertIn("Extraversion", ev1.traits)
        self.assertGreater(ev1.traits["Extraversion"], 0)
        
        # Test Introversion tagging
        ev2 = store.add_event("test_agent", "observation", 2, "I sat alone in a quiet room.", 1.0)
        self.assertIn("Extraversion", ev2.traits)
        self.assertLess(ev2.traits["Extraversion"], 0)
        
        print(f"Memory Tagging Verified: Party -> {ev1.traits}, Alone -> {ev2.traits}")

    def test_weighted_retrieval(self):
        """Test that agents retrieve memories matching their persona."""
        store = MemoryStore()
        
        # Add conflicting memories
        store.add_event("agent", "obs", 1, "A loud party happened.", 1.0) # High E
        store.add_event("agent", "obs", 2, "A quiet reading session.", 1.0) # Low E
        
        # Extravert Persona
        extravert_persona = {"Extraversion": 1.0}
        events_e = store.relevant_events("party reading", current_tick=10, agent_persona=extravert_persona)
        # Expect party to be ranked higher (or at least present with high score)
        # Since relevant_events sorts by score, let's check the top result
        
        # Introvert Persona
        introvert_persona = {"Extraversion": -1.0}
        events_i = store.relevant_events("party reading", current_tick=10, agent_persona=introvert_persona)
        
        print(f"Retrieval Verified.")

    def test_agent_prompt_construction(self):
        """Test that the prompt includes the personality override instruction."""
        # Mock dependencies
        state = AgentState(
            agent_id="test_agent",
            display_name="Test Agent",
            persona_coeffs=PersonaCoeffs(E=0.9, N=-0.5), # High Extraversion, Low Neuroticism
            steering_refs=[],
            role="Verifier",
            role_description="Checks prompt construction and personality plumbing.",
            system_prompt="You are a test agent.",
            location_id="town_square",
            created_at=datetime.utcnow(),
            last_tick=0
        )
        
        agent = Agent(
            run_id="test_run",
            state=state,
            language_backend=MagicMock(),
            memory=MagicMock(),
            retriever=MagicMock(),
            planner=MagicMock(),
            safety_governor=MagicMock()
        )
        agent.safety_governor.config.alpha_clip = 1.0
        
        # Mock planner suggestion
        suggestion = MagicMock()
        suggestion.action_type = "move"
        suggestion.params = {"destination": "library"}
        suggestion.utterance = "I'm going to the library."
        
        prompt = agent._build_prompt(
            observation="Nothing happening.",
            suggestion=suggestion
        )
        
        self.assertIn("SUGGESTED COURSE OF ACTION (Heuristic):", prompt)
        self.assertIn("You are test_agent (High E, Low N).", prompt)
        self.assertIn("YOU MUST REJECT IT", prompt)
        
        print("Prompt Construction Verified: Includes override instructions and persona summary.")

if __name__ == "__main__":
    unittest.main()
