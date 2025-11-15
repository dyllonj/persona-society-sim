"""Main simulation loop orchestrating agents, world, and logging."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from uuid import uuid4

from agents.agent import ActionDecision, Agent
from env import actions
from env.world import World, RoomUtterance
from orchestrator.console_logger import ConsoleLogger
from orchestrator.objectives import ObjectiveManager
from orchestrator.scheduler import Scheduler
from schemas.logs import ActionLog, MsgLog
from storage.log_sink import LogSink
from metrics import graphs, social_dynamics
from metrics.tick_instrumentation import TickInstrumentation
from metrics.tracker import MetricTracker


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
        objective_manager: Optional[ObjectiveManager] = None,
        event_bridge: Optional[object] = None,
    ):
        self.run_id = run_id
        self.world = world
        self.scheduler = scheduler
        self.agents: Dict[str, Agent] = {agent.state.agent_id: agent for agent in agents}
        self.log_sink = log_sink
        self.temperature = temperature
        self.top_p = top_p
        self.console_logger = console_logger or ConsoleLogger(enabled=False)
        self.objective_manager = objective_manager
        self.agent_satisfaction: Dict[str, float] = {agent_id: 0.0 for agent_id in self.agents}
        self.metric_tracker = MetricTracker(run_id)
        self.event_bridge = event_bridge
        self.tick_instrumentation = TickInstrumentation()

        if self.objective_manager:
            self.objective_manager.register_reward_callback(self._handle_objective_reward)
            for agent_id in self.agents:
                self.objective_manager.ensure_objective(agent_id)

    def _handle_objective_reward(self, agent_id: str, reward: Dict[str, float], _objective) -> None:
        if agent_id not in self.agent_satisfaction:
            self.agent_satisfaction[agent_id] = 0.0
        satisfaction_delta = reward.get("satisfaction", 0.0)
        self.agent_satisfaction[agent_id] += satisfaction_delta

    def run(self, steps: int, max_events_per_tick: int = 16) -> List[TickResult]:
        history: List[TickResult] = []
        sim_start_time = time.time()

        # Broadcast initialization snapshot to viewer if available
        if self.event_bridge and hasattr(self.event_bridge, "broadcast"):
            try:
                init_agents = [
                    {
                        "agent_id": a.state.agent_id,
                        "display_name": a.state.display_name,
                        "persona_coeffs": a.state.persona_coeffs.model_dump(),
                        "location_id": a.state.location_id,
                    }
                    for a in self.agents.values()
                ]
                world_payload = self.world.serialize(include_agents=True, agent_ids=self.agents.keys())
                self.event_bridge.broadcast({"type": "init", "world": world_payload, "agents": init_agents})
            except Exception:
                pass

        for step_idx in range(steps):
            tick_start_time = time.time()
            tick_logs: List[ActionLog] = []
            encounters = self.scheduler.sample(list(self.agents.keys()), max_events_per_tick)
            # Simple collaboration metric: share of actions happening with peers present
            collab_actions = 0
            total_actions = 0

            self.tick_instrumentation.on_tick_start(self.world.tick)

            # Log tick start
            self.console_logger.log_tick_start(self.world.tick, len(encounters))

            for encounter in encounters:
                encounter_transcript: List[RoomUtterance] = list(encounter.transcript)
                for agent_id in encounter.participants:
                    agent = self.agents[agent_id]
                    current_location = self.world.agent_location(agent.state.agent_id)
                    active_objective = (
                        self.objective_manager.current_objective(agent.state.agent_id)
                        if self.objective_manager
                        else None
                    )

                    base_context = self.world.sample_context(agent.state.agent_id)
                    decision = agent.act(
                        base_context,
                        self.world.tick,
                        current_location=current_location,
                        active_objective=active_objective,
                        recent_dialogue=tuple(encounter_transcript),
                        rule_context=self.world.institutional_guidance(),
                    )
                    src_location = current_location
                    env_result = actions.execute(
                        self.world,
                        agent.state.agent_id,
                        decision.action_type,
                        decision.params,
                    )
                    # Attach source location for moves to improve logging clarity
                    if decision.action_type == "move" and "destination" in decision.params:
                        decision.params = {**decision.params, "from": src_location}

                    action_log = ActionLog(
                        action_id=str(uuid4()),
                        run_id=self.run_id,
                        tick=self.world.tick,
                        agent_id=agent.state.agent_id,
                        action_type=env_result.action_type,
                        params=decision.params,
                        outcome="success" if env_result.success else "fail",
                        info={**env_result.info, "utterance": decision.utterance},
                        prompt_text=decision.prompt_text,
                        prompt_hash=decision.prompt_hash,
                        plan_metadata=decision.plan_metadata,
                        reflection_summary=decision.reflection_summary,
                        reflection_implications=decision.reflection_implications,
                    )
                    tick_logs.append(action_log)
                    self.log_sink.log_action(action_log)
                    try:
                        self.tick_instrumentation.record_action(
                            agent_id=agent.state.agent_id,
                            action_type=env_result.action_type,
                            success=env_result.success,
                            params=decision.params,
                            info=env_result.info,
                            steering_snapshot=decision.steering_snapshot,
                            persona_coeffs=agent.state.persona_coeffs.model_dump(),
                            encounter_room=encounter.room_id,
                            encounter_participants=encounter.participants,
                            satisfaction=self.agent_satisfaction.get(agent.state.agent_id, 0.0),
                        )
                    except Exception:
                        pass

                    # Broadcast action to viewer
                    if self.event_bridge and hasattr(self.event_bridge, "broadcast"):
                        try:
                            self.event_bridge.broadcast(
                                {
                                    "type": "action",
                                    "tick": self.world.tick,
                                    "agent_id": action_log.agent_id,
                                    "action_type": action_log.action_type,
                                    "params": action_log.params,
                                    "outcome": action_log.outcome,
                                }
                            )
                        except Exception:
                            pass

                    # Update collaboration metric using occupancy at source room for
                    # actions that represent room-level collaboration (speak/share/work)
                    try:
                        # Count actions that broadcast or imply collaboration in-room
                        executed_type = env_result.action_type
                        collab_like = {"talk", "trade", "work", "research", "cite", "submit_report"}
                        if executed_type in collab_like:
                            # Use source location occupancy (pre-move context)
                            room = src_location
                            occupants = (
                                len(self.world.locations.get(room).occupants)
                                if room in self.world.locations
                                else 0
                            )
                            total_actions += 1
                            if occupants and occupants > 1:
                                collab_actions += 1
                            # Metrics tracker for per-agent summaries
                            self.metric_tracker.on_action(action_log, occupants)
                        else:
                            # Still feed metrics with current occupancy for visibility
                            room_now = self.world.agent_location(agent.state.agent_id)
                            occ_now = (
                                len(self.world.locations.get(room_now).occupants)
                                if room_now in self.world.locations
                                else 0
                            )
                            self.metric_tracker.on_action(action_log, occ_now)
                    except Exception:
                        # Metric collection should never interfere with the run
                        pass

                    if self.objective_manager:
                        completed_objective = self.objective_manager.process_action_log(action_log)
                        if completed_objective is not None:
                            self.objective_manager.ensure_objective(agent.state.agent_id)

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

                    # Broadcast chat to viewer
                    if self.event_bridge and hasattr(self.event_bridge, "broadcast"):
                        try:
                            self.event_bridge.broadcast(
                                {
                                    "type": "chat",
                                    "tick": self.world.tick,
                                    "from_agent": msg_log.from_agent,
                                    "room_id": msg_log.room_id,
                                    "content": msg_log.content,
                                }
                            )
                        except Exception:
                            pass

                    if decision.safety_event:
                        self.log_sink.log_safety(decision.safety_event)

                    if (
                        decision.action_type == "talk"
                        and decision.utterance
                        and current_location == encounter.room_id
                    ):
                        encounter_transcript.append(
                            RoomUtterance(
                                speaker=agent.state.agent_id,
                                content=decision.utterance,
                                tick=self.world.tick,
                            )
                        )

            self.world.step()
            self._log_tick_snapshots()
            self.log_sink.flush(self.world.tick)

            # Log tick end with duration
            tick_duration_ms = (time.time() - tick_start_time) * 1000
            ratio = (collab_actions / total_actions) if total_actions else 0.0
            self.console_logger.log_tick_end(self.world.tick, tick_duration_ms, ratio)
            try:
                self.metric_tracker.on_tick_end(self.world.tick, ratio)
            except Exception:
                pass

            # Broadcast tick summary with positions
            if self.event_bridge and hasattr(self.event_bridge, "broadcast"):
                try:
                    world_state = self.world.serialize(include_agents=True, agent_ids=self.agents.keys())
                    agent_slice = world_state.get("agents", {})
                    positions = {agent_id: data.get("location_id", "unknown") for agent_id, data in agent_slice.items()}
                    inventories = {agent_id: data.get("inventory", {}) for agent_id, data in agent_slice.items()}
                    self.event_bridge.broadcast(
                        {
                            "type": "tick",
                            "tick": self.world.tick,
                            "positions": positions,
                            "stats": {"collab_ratio": ratio, "duration_ms": tick_duration_ms},
                            "inventories": inventories,
                            "world": world_state,
                        }
                    )
                except Exception:
                    pass

            history.append(TickResult(tick=self.world.tick, action_logs=tick_logs))

        # Log simulation summary
        total_time = time.time() - sim_start_time
        self.console_logger.log_summary(self.run_id, steps, len(self.agents), total_time)

        try:
            self.metric_tracker.flush()
        except Exception:
            pass
        return history

    def _log_tick_snapshots(self) -> None:
        try:
            tick_idx = max(0, self.world.tick - 1)
            for graph_input in self.tick_instrumentation.graph_inputs():
                snapshot = graphs.snapshot_from_edges(
                    self.run_id,
                    tick_idx,
                    graph_input.edges,
                    trait_key=graph_input.trait_key,
                    band_metadata=graph_input.band_metadata,
                )
                self.log_sink.log_graph_snapshot(snapshot)
            wealth_snapshot = self.world.economy.snapshot()
            macro_inputs = self.tick_instrumentation.macro_inputs(wealth_snapshot, self.agent_satisfaction)
            for macro in macro_inputs:
                metrics_snapshot = social_dynamics.build_metrics_snapshot(
                    self.run_id,
                    tick_idx,
                    macro.cooperation_events,
                    macro.wealth,
                    macro.opinions,
                    macro.conflicts,
                    macro.enforcement_cost,
                    trait_key=macro.trait_key,
                    band_metadata=macro.band_metadata,
                )
                self.log_sink.log_metrics_snapshot(metrics_snapshot)
        except Exception:
            pass
