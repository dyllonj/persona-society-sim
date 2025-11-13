"""Main simulation loop orchestrating agents, world, and logging."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from uuid import uuid4

from agents.agent import ActionDecision, Agent
from env import actions
from env.world import World
from orchestrator.console_logger import ConsoleLogger
from orchestrator.scheduler import Scheduler
from schemas.logs import ActionLog, MsgLog
from storage.log_sink import LogSink


@dataclass
class TickResult:
    tick: int
    action_logs: List[ActionLog]


class SimulationRunner:
    def __init__(
        self,
        run_id: str,
        world: World,
        scheduler: Scheduler,
        agents: Iterable[Agent],
        log_sink: LogSink,
        temperature: float,
        top_p: float,
        console_logger: Optional[ConsoleLogger] = None,
    ):
        self.run_id = run_id
        self.world = world
        self.scheduler = scheduler
        self.agents: Dict[str, Agent] = {agent.state.agent_id: agent for agent in agents}
        self.log_sink = log_sink
        self.temperature = temperature
        self.top_p = top_p
        self.console_logger = console_logger or ConsoleLogger(enabled=False)

    def run(self, steps: int, max_events_per_tick: int = 16) -> List[TickResult]:
        history: List[TickResult] = []
        sim_start_time = time.time()

        for step_idx in range(steps):
            tick_start_time = time.time()
            tick_logs: List[ActionLog] = []
            encounters = self.scheduler.sample(list(self.agents.keys()), max_events_per_tick)

            # Log tick start
            self.console_logger.log_tick_start(self.world.tick, len(encounters))

            for encounter in encounters:
                agent = self.agents[encounter.agent_id]
                decision = agent.act(encounter.context, self.world.tick)
                env_result = actions.execute(
                    self.world,
                    agent.state.agent_id,
                    decision.action_type,
                    decision.params,
                )
                action_log = ActionLog(
                    action_id=str(uuid4()),
                    run_id=self.run_id,
                    tick=self.world.tick,
                    agent_id=agent.state.agent_id,
                    action_type=env_result.action_type,
                    params=decision.params,
                    outcome="success" if env_result.success else "fail",
                    info={**env_result.info, "utterance": decision.utterance},
                )
                tick_logs.append(action_log)
                self.log_sink.log_action(action_log)

                # Log action to console
                self.console_logger.log_action(action_log)

                msg_log = MsgLog(
                    msg_id=str(uuid4()),
                    run_id=self.run_id,
                    tick=self.world.tick,
                    channel="room",
                    from_agent=agent.state.agent_id,
                    to_agent=None,
                    room_id=self.world.agent_location(agent.state.agent_id),
                    content=decision.utterance,
                    tokens_in=decision.tokens_in,
                    tokens_out=decision.tokens_out,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    steering_snapshot=decision.steering_snapshot,
                    layers_used=decision.layers_used,
                )
                self.log_sink.log_message(msg_log)

                # Log message to console
                self.console_logger.log_message(msg_log)

                if decision.safety_event:
                    self.log_sink.log_safety(decision.safety_event)

            self.world.step()
            self.log_sink.flush(self.world.tick)

            # Log tick end with duration
            tick_duration_ms = (time.time() - tick_start_time) * 1000
            self.console_logger.log_tick_end(self.world.tick, tick_duration_ms)

            history.append(TickResult(tick=self.world.tick, action_logs=tick_logs))

        # Log simulation summary
        total_time = time.time() - sim_start_time
        self.console_logger.log_summary(self.run_id, steps, len(self.agents), total_time)

        return history
