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


def test_planner_handles_policy_objective_fields():
    planner = Planner()
    objective = SimpleNamespace(
        objective_id="obj-2",
        agent_id="agent-77",
        type="policy",
        description="Complete the compliance checklist",
        requirements={"fill_field": 2, "submit_plan": 1},
        progress={"fill_field": 1, "submit_plan": 0},
    )

    plan = planner.plan([], "", current_location="community_center", active_objective=objective, tick=5)

    assert plan.action_type == "fill_field"
    assert plan.params["field_name"].startswith("policy_field_")


def test_planner_scans_when_navigation_objective_active():
    planner = Planner()
    objective = SimpleNamespace(
        objective_id="obj-3",
        agent_id="agent-99",
        type="navigation",
        description="Scan new areas",
        requirements={"scan": 2},
        progress={"scan": 0},
    )

    plan = planner.plan([], "", current_location="town_square", active_objective=objective, tick=3)

    assert plan.action_type == "scan"


def test_planner_uses_observation_hint_keywords():
    planner = Planner()

    plan = planner.plan(
        ["Assist"],
        "No location clues",  # memory summary without market keyword
        current_location="town_square",
        observation_hint=["market is busy", "stalls"],
    )

    assert plan.action_type == "trade"
