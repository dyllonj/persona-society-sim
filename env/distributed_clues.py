"""Versioned private-information meeting task on the original town map.

This opt-in environment uses World for locations and encounters, but does not use
the legacy research corpus, planner, report grader, or automatic broadcasts.
Only observe() output reaches a policy. audit() is privileged evaluator data.
"""
from __future__ import annotations

import copy
import itertools
import random

from env.world import World


CONDITIONS = ("communication", "no_communication", "all_clues_single")
# "together" is the original layout. "scattered" starts each resident in a
# different room, so sharing first requires finding a co-located peer.
LAYOUTS = ("together", "scattered")
# "dropout" makes one seed-chosen resident unavailable after round 0: it takes
# no further turns and never submits. Peers are not told.
DISRUPTIONS = ("none", "dropout")


class DistributedClues:
    version = "town-distributed-clues-v1"

    def __init__(self, seed=7, condition="communication", rounds=3, layout="together", disruption="none"):
        if condition not in CONDITIONS or type(rounds) is not int or not 1 <= rounds <= 12:
            raise ValueError("Invalid condition or rounds (1..12 required)")
        if layout not in LAYOUTS or disruption not in DISRUPTIONS:
            raise ValueError("Invalid layout or disruption")
        if disruption == "dropout" and (condition == "all_clues_single" or rounds < 2):
            raise ValueError("Dropout requires multiple residents and at least two rounds")
        self.seed, self.condition, self.rounds = seed, condition, rounds
        self.layout, self.disruption = layout, disruption
        self.task_version = self.version
        if layout == "scattered":
            self.task_version += "+scattered-start-v1"
        if disruption == "dropout":
            self.task_version += "+dropout-after-round0-v1"
        self.world = World()
        self.agents = ["resident_a", "resident_b", "resident_c"]
        rng = random.Random(seed)
        # Three independent binary constraints: one clue leaves 4 candidates,
        # two leave 2, and all three leave exactly 1. Shuffle the mapping so IDs
        # and rendered locations/times do not encode the private constraints.
        codes = list(itertools.product((0, 1), repeat=3))
        rng.shuffle(codes)
        chosen = tuple(rng.randrange(2) for _ in range(3))
        self.plans = [
            {"id": f"plan_{i + 1}", "venue": venue, "time": hour}
            for i, (venue, hour) in enumerate(itertools.product(
                ("library", "community_center"), ("09:00", "11:00", "14:00", "16:00")))
        ]
        self._answer = self.plans[codes.index(chosen)]["id"]
        owners = self.agents.copy()
        rng.shuffle(owners)
        self._clues = {}
        self._owners = {}
        evidence_codes = rng.sample(range(65536), 3)
        for dimension, subject in enumerate(("venue availability", "attendee availability", "equipment availability")):
            evidence_id = f"e{evidence_codes[dimension]:04x}"
            self._clues[evidence_id] = {
                "id": evidence_id, "subject": subject,
                "allowed_plans": [p["id"] for p, code in zip(self.plans, codes)
                                  if code[dimension] == chosen[dimension]],
            }
            self._owners[evidence_id] = owners[dimension]
        self._initial = {a: {e for e, owner in self._owners.items() if owner == a} for a in self.agents}
        if condition == "all_clues_single":
            self.agents = ["resident_a"]
            self._initial = {"resident_a": set(self._clues)}
        self._known = copy.deepcopy(self._initial)
        self._inbox = {a: [] for a in self.agents}
        self._last_result = {a: None for a in self.agents}
        self.submissions = {}
        self.events = []
        rooms = ["town_square"] * len(self.agents)
        if layout == "scattered":
            # Separate stream: clues, answer and turn order match the together layout.
            rooms = sorted(self.world.locations)
            random.Random(seed ^ 0x5CA7).shuffle(rooms)
        for agent, room in zip(self.agents, rooms):
            self.world.add_agent(agent, room)
        order = self.agents.copy()
        random.Random(seed ^ 0x5A17).shuffle(order)
        self.dropped = None
        if disruption == "dropout":
            self.dropped = random.Random(seed ^ 0xD20F).choice(sorted(self.agents))
        self.active = [a for a in self.agents if a != self.dropped]
        self.schedule = [(r, a, "deliberate") for r in range(rounds) for a in order
                         if r == 0 or a != self.dropped]
        self.schedule += [(rounds, a, "submit") for a in order if a != self.dropped]
        self.cursor = 0

    @property
    def done(self):
        return self.cursor == len(self.schedule)

    @property
    def current_agent(self):
        return None if self.done else self.schedule[self.cursor][1]

    def observe(self, agent):
        if agent not in self.agents or self.done or agent != self.current_agent:
            raise ValueError("Observation requested out of turn")
        round_index, _, phase = self.schedule[self.cursor]
        # Do not expose peer locations, actions, submissions or grades: those
        # would become side channels when communication is disabled.
        return copy.deepcopy({
            "task": "Choose the ONE meeting plan satisfying all three private constraints. Share evidence to learn missing constraints. Submit the plan and all three evidence IDs.",
            "agent": agent, "residents": self.agents, "round": round_index,
            "phase": phase, "deliberation_rounds": self.rounds,
            "communication_enabled": self.condition != "no_communication",
            "location": self.world.agent_location(agent),
            "neighbors": sorted(self.world.locations[self.world.agent_location(agent)].neighbors),
            "plans": self.plans,
            "known_evidence": [self._clues[e] for e in sorted(self._known[agent])],
            "recent_messages": self._inbox[agent][-3:],
            "last_result": self._last_result[agent],
            **({"start_layout": "scattered"} if self.layout == "scattered" else {}),
        })

    def step(self, agent, action):
        if self.done or agent != self.current_agent:
            raise ValueError("Action requested out of turn")
        round_index, _, phase = self.schedule[self.cursor]
        before = sorted(self._known[agent])
        result = self._execute(agent, action, phase)
        event = {"turn": self.cursor, "round": round_index, "agent": agent,
                 "phase": phase, "action": copy.deepcopy(action),
                 "known_before": before, "result": copy.deepcopy(result)}
        self.events.append(event)
        # Scores and submissions are never revealed during the episode.
        self._last_result[agent] = {k: v for k, v in result.items()
                                    if k not in ("answer_correct", "supported_success", "evidence_complete", "transfers")}
        self.cursor += 1
        self.world.tick = self.cursor
        return copy.deepcopy(result)

    def _execute(self, agent, action, phase):
        def invalid(detail):
            return {"valid": False, "error": detail}
        if not isinstance(action, dict) or set(action) != {"action", "params"}:
            return invalid("Expected exactly action and params")
        name, params = action["action"], action["params"]
        if not isinstance(name, str) or not isinstance(params, dict):
            return invalid("action must be a string and params an object")
        if phase == "submit":
            if name != "submit_report" or set(params) != {"plan_id", "evidence_ids"}:
                return invalid("Final turn requires submit_report with plan_id and evidence_ids")
            evidence = params["evidence_ids"]
            if not isinstance(params["plan_id"], str) or params["plan_id"] not in {p["id"] for p in self.plans}:
                return invalid("Unknown plan")
            if not self._valid_evidence_list(evidence):
                return invalid("evidence_ids must be a unique list of at most three strings")
            if not set(evidence) <= self._known[agent]:
                return invalid("Cannot cite evidence you have not received")
            correct = params["plan_id"] == self._answer
            complete = set(evidence) == set(self._clues)
            result = {"valid": True, "submitted": True, "answer_correct": correct,
                      "evidence_complete": complete, "supported_success": correct and complete}
            self.submissions[agent] = {**copy.deepcopy(params), **result}
            return result
        if name == "wait" and not params:
            return {"valid": True}
        if name == "move" and set(params) == {"destination"}:
            destination = params["destination"]
            current = self.world.agent_location(agent)
            if not isinstance(destination, str) or destination not in self.world.locations[current].neighbors:
                return invalid("Destination must be an adjacent town location")
            self.world.move_agent(agent, destination)
            return {"valid": True, "location": destination}
        if name == "share" and set(params) == {"evidence_ids", "message"}:
            evidence, message = params["evidence_ids"], params["message"]
            if not self._valid_evidence_list(evidence):
                return invalid("evidence_ids must be a unique list of at most three strings")
            if not isinstance(message, str) or len(message) > 280:
                return invalid("message must contain at most 280 characters")
            if not set(evidence) <= self._known[agent]:
                return invalid("Cannot share evidence you have not received")
            recipients, transfers = [], []
            if self.condition != "no_communication":
                room = self.world.agent_location(agent)
                recipients = sorted(self.world.locations[room].occupants - {agent})
                for recipient in recipients:
                    new = sorted(set(evidence) - self._known[recipient])
                    self._known[recipient].update(evidence)
                    self._inbox[recipient].append({"from": agent, "message": message, "evidence_ids": evidence.copy()})
                    transfers.append({"recipient": recipient, "new_evidence_ids": new})
            return {"valid": True, "delivered_to": recipients, "transfers": transfers}
        return invalid("Use share, move or wait during deliberation")

    @staticmethod
    def _valid_evidence_list(value):
        return (isinstance(value, list) and len(value) <= 3
                and all(isinstance(e, str) and 1 <= len(e) <= 64 for e in value)
                and len(set(value)) == len(value))

    def summary(self):
        submitted = list(self.submissions.values())
        return {
            "task_version": self.task_version, "seed": self.seed, "condition": self.condition,
            # Under dropout, scores cover the residents still able to submit.
            "complete": self.done, "agents": len(self.active), "turn_budget": len(self.schedule),
            "turns_executed": self.cursor, "invalid_actions": sum(not e["result"]["valid"] for e in self.events),
            "valid_submissions": len(submitted),
            "correct_answers": sum(s["answer_correct"] for s in submitted),
            "supported_submissions": sum(s["supported_success"] for s in submitted),
            "team_success": any(s["supported_success"] for s in submitted),
            "supported_fraction": sum(s["supported_success"] for s in submitted) / len(self.active),
            "evidence_coverage": {a: len(self._known[a]) / 3 for a in self.active},
            "new_evidence_deliveries": sum(len(t["new_evidence_ids"]) for e in self.events for t in e["result"].get("transfers", [])),
            **({"dropped_agent": self.dropped} if self.dropped else {}),
            "interpretation": "Evidence receipt is observable; it does not prove internal reasoning or human-like collaboration. No-communication cannot achieve full evidence support by construction; compare answer accuracy too.",
        }

    def audit(self):
        return copy.deepcopy({"task_version": self.task_version, "answer": self._answer,
                              "clues": self._clues, "owners": self._owners,
                              **({"dropped_agent": self.dropped} if self.dropped else {}),
                              "initial_evidence": {a: sorted(e) for a, e in self._initial.items()}})
