"""Pure renderers for final decisions; no answer computation or privileged state."""

def intersection_prompt(observation):
    # Render only available evidence. Never compute or insert the solution.
    lists = "\n".join(f"{c['id']} ({c['subject']}): {', '.join(c['allowed_plans'])}"
                      for c in observation["known_evidence"])
    return (
        "Choose a meeting plan that is allowed by EVERY constraint below. "
        "The answer must occur in ALL THREE lists, not merely in one or two. "
        "Find their set intersection and check your selected plan against each list. "
        "Exactly one plan is common to all three lists. All evidence is available here; no other agent is needed.\n"
        + lists
        + '\nReturn ONLY JSON with this structure: {"action":"submit_report","params":{"plan_id":"plan ID","evidence_ids":["all three evidence IDs"]}}. '
        "Use the actual IDs above."
    )


def final_decision_prompt(observation):
    if observation["phase"] != "submit":
        raise ValueError("Final-decision renderer requires submit phase")
    evidence = observation["known_evidence"]
    if len(evidence) == 3:
        # Preserve the exact tested all-clues prompt, including evidence order.
        return intersection_prompt(observation)
    lists = "\n".join(f"{c['id']} ({c['subject']}): {', '.join(c['allowed_plans'])}" for c in evidence)
    return (
        f"FINAL TURN: you have {len(evidence)} of the three constraints. Missing constraints are unknown. "
        "Choose a meeting plan consistent with EVERY available list below. "
        "More than one candidate may remain; select one, without inventing missing evidence.\n"
        + lists
        + '\nReturn ONLY JSON: {"action":"submit_report","params":{"plan_id":"plan ID","evidence_ids":["known evidence IDs"]}}. '
        "Use only actual evidence IDs above."
    )
