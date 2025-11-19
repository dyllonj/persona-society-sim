import unittest
from unittest.mock import MagicMock
from orchestrator.meta_manager import MetaOrchestrator, AlignmentContext
from agents.planner import Planner, PlanSuggestion

class TestManagerOrchestration(unittest.TestCase):
    def test_meta_orchestrator_silence_detection(self):
        orchestrator = MetaOrchestrator()
        
        # Setup: 2 agents in room "room1", silent for 4 ticks
        world_state = {"agent1": "room1", "agent2": "room1"}
        agents = {"agent1": MagicMock(), "agent2": MagicMock()}
        
        # Tick 0: Activity happens (initialization)
        orchestrator.update_activity(0, ["room1"])
        
        # Tick 4: 4 ticks later, should be silent > 3 ticks
        directives = orchestrator.alignment_directives(
            tick=4, 
            agents=agents, 
            world_state=world_state
        )
        
        # Expect force_collaboration
        self.assertIn("force_collaboration", directives["agent1"].planning_hints)
        self.assertIn("force_collaboration", directives["agent2"].planning_hints)
        
        # Tick 2: Only 2 ticks later, should NOT force
        orchestrator.update_activity(2, ["room1"]) # Activity at tick 2
        directives = orchestrator.alignment_directives(
            tick=4, 
            agents=agents, 
            world_state=world_state
        )
        self.assertNotIn("force_collaboration", directives["agent1"].planning_hints)

    def test_planner_force_collaboration(self):
        planner = Planner()
        
        # Normal plan
        suggestion = planner.plan(
            goals=["explore"], 
            memory_summary="", 
            planning_hints=[]
        )
        self.assertNotEqual(suggestion.params.get("topic"), "forced_collab")
        
        # Forced plan
        suggestion = planner.plan(
            goals=["explore"], 
            memory_summary="", 
            planning_hints=["force_collaboration"]
        )
        self.assertEqual(suggestion.action_type, "talk")
        self.assertEqual(suggestion.params.get("topic"), "forced_collab")

if __name__ == "__main__":
    unittest.main()
