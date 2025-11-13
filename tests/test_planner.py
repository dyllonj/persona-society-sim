from types import SimpleNamespace

from agents.planner import Planner


def _make_objective(obj_type: str, description: str):
    return SimpleNamespace(
        objective_id="obj-1",
        agent_id="agent-42",
        type=obj_type,
        description=description,
        requirements={obj_type: 1},
        progress={},
    )


def test_planner_moves_toward_objective_location_when_needed():
    planner = Planner()
    objective = _make_objective("gather", "gather needed supplies")

    plan = planner.plan([], "", current_location="town_square", active_objective=objective)

    assert plan.action_type == "move"
    assert plan.params["destination"] == "market"
    assert "market" in plan.utterance


def test_planner_uses_objective_specific_action_when_at_location():
    planner = Planner()
    objective = _make_objective("collaborate", "collaborate with neighbors")

    plan = planner.plan([], "", current_location="community_center", active_objective=objective)

    assert plan.action_type == "talk"
    assert "collaborate" in plan.params["utterance"].lower()
    assert "collaborate" in plan.utterance.lower()
