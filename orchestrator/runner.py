"""Main simulation loop orchestrating agents, world, and logging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from agents.agent import Agent
from env import actions
from env.world import World
from orchestrator.scheduler import Scheduler
from schemas.logs import ActionLog


@dataclass
class TickResult:
    tick: int
    action_logs: List[ActionLog]


class SimulationRunner:
    def __init__(self, world: World, scheduler: Scheduler, agents: Iterable[Agent]):
        self.world = world
        self.scheduler = scheduler
        self.agents: Dict[str, Agent] = {agent.state.agent_id: agent for agent in agents}
        self.logs: List[ActionLog] = []

    def run(self, steps: int, max_events_per_tick: int = 16) -> List[TickResult]:
        history: List[TickResult] = []
        for _ in range(steps):
            tick_logs: List[ActionLog] = []
            encounters = self.scheduler.sample(list(self.agents.keys()), max_events_per_tick)
            for encounter in encounters:
                agent = self.agents[encounter.agent_id]
                act_result = agent.act(encounter.context, self.world.tick)
                env_result = actions.execute(
                    self.world,
                    agent.state.agent_id,
                    act_result["action_type"],
                    act_result["params"],
                )
                log = ActionLog(
                    action_id=f"{agent.state.agent_id}-{self.world.tick}",
                    run_id="debug",
                    tick=self.world.tick,
                    agent_id=agent.state.agent_id,
                    action_type=env_result.action_type,
                    params=act_result["params"],
                    outcome="success" if env_result.success else "fail",
                    info=env_result.info,
                )
                tick_logs.append(log)
            self.world.step()
            history.append(TickResult(tick=self.world.tick, action_logs=tick_logs))
        return history
