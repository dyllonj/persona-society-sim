"""Main simulation loop orchestrating agents, world, and logging."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from uuid import uuid4

from agents.agent import ActionDecision, Agent
from env import actions
from env.world import World, RoomUtterance
from orchestrator.console_logger import ConsoleLogger
from orchestrator.objectives import ObjectiveManager
from orchestrator.probes import ProbeAssignment, ProbeManager
from orchestrator.scheduler import Scheduler
from orchestrator.meta_manager import AlignmentContext, MetaOrchestrator
from schemas.logs import (
    ActionLog,
    MsgLog,
    CitationLog,
    ReportGradeLog,
    ResearchFactLog,
    ProbeLog,
    BehaviorProbeLog,
)
from storage.log_sink import LogSink
from metrics import graphs, social_dynamics
from metrics.tick_instrumentation import TickInstrumentation
from metrics.tracker import MetricTracker
from metrics.persona_bands import trait_band_key


@dataclass
class TickResult:
    tick: int
    action_logs: List[ActionLog]


@dataclass
class TraitMetadata:
    trait_key: Optional[str]
    trait_name: Optional[str]
    trait_band: Optional[str]
    alpha_value: Optional[float]
    alpha_bucket: Optional[str]


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
        probe_manager: Optional[ProbeManager] = None,
        meta_orchestrator: Optional[MetaOrchestrator] = None,
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
        persona_map = {agent_id: agent.state.persona_coeffs for agent_id, agent in self.agents.items()}
        metrics_dir = log_sink.parquet_dir if log_sink.parquet_dir else Path("metrics")
        self.metric_tracker = MetricTracker(run_id, agent_personas=persona_map, out_dir=metrics_dir)
        self.event_bridge = event_bridge
        self.tick_instrumentation = TickInstrumentation()
        self.probe_manager = probe_manager
        self.meta_orchestrator = meta_orchestrator

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

        # Start viewer if it has a start method (e.g. TUI or WebSocket server)
        if self.event_bridge and hasattr(self.event_bridge, "start"):
            try:
                self.event_bridge.start()
            except Exception:
                pass

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

            alignment_contexts: Dict[str, AlignmentContext] = {}
            if self.meta_orchestrator:
                alignment_contexts = self.meta_orchestrator.alignment_directives(
                    self.world.tick, self.agents, self.objective_manager
                )
                if self.meta_orchestrator.last_broadcast:
                    self.console_logger.log_info(
                        f"Meta broadcast: {self.meta_orchestrator.last_broadcast}"
                    )
                    if self.event_bridge and hasattr(self.event_bridge, "broadcast"):
                        try:
                            self.event_bridge.broadcast(
                                {
                                    "type": "meta_broadcast",
                                    "tick": self.world.tick,
                                    "message": self.meta_orchestrator.last_broadcast,
                                    "global_goals": list(self.meta_orchestrator.global_goals),
                                }
                            )
                        except Exception:
                            pass

            for encounter in encounters:
                encounter_transcript: List[RoomUtterance] = list(encounter.transcript)
                for agent_id in encounter.participants:
                    agent = self.agents[agent_id]
                    current_location = self.world.agent_location(agent.state.agent_id)
                    peers_present = any(peer != agent.state.agent_id for peer in encounter.participants)
                    active_objective = (
                        self.objective_manager.current_objective(agent.state.agent_id)
                        if self.objective_manager
                        else None
                    )

                    base_context = self.world.sample_context(agent.state.agent_id)
                    probe_assignment: Optional[ProbeAssignment] = None
                    if self.probe_manager:
                        probe_assignment = self.probe_manager.pending_probe(
                            agent.state.agent_id, self.world.tick
                        )
                        if not probe_assignment:
                            probe_assignment = self.probe_manager.assign_probe(
                                agent.state.agent_id, self.world.tick
                            )
                        if probe_assignment:
                            base_context = probe_assignment.inject(base_context)
                            base_context = probe_assignment.inject(base_context)
                    
                    # Broadcast processing status
                    if self.event_bridge and hasattr(self.event_bridge, "broadcast"):
                        try:
                            self.event_bridge.broadcast(
                                {
                                    "type": "processing",
                                    "tick": self.world.tick,
                                    "agent_id": agent.state.agent_id,
                                }
                            )
                            # Force a small sleep to ensure TUI updates before blocking inference
                            time.sleep(0.1)
                        except Exception:
                            pass

                    decision = agent.act(
                        base_context,
                        self.world.tick,
                        current_location=current_location,
                        active_objective=active_objective,
                        recent_dialogue=tuple(encounter_transcript),
                        rule_context=self.world.institutional_guidance(),
                        peers_present=peers_present,
                        alignment_context=alignment_contexts.get(agent.state.agent_id),
                    )
                    if probe_assignment:
                        decision.probe_id = probe_assignment.probe_id
                        decision.probe_kind = probe_assignment.kind
                    src_location = current_location
                    env_result = actions.execute(
                        self.world,
                        agent.state.agent_id,
                        decision.action_type,
                        decision.params,
                    )
                    trait_meta = self._trait_metadata(agent, decision.steering_snapshot)
                    self._emit_structured_logs(agent, env_result, trait_meta)
                    # Attach source location for moves to improve logging clarity
                    if decision.action_type == "move" and "destination" in decision.params:
                        decision.params = {**decision.params, "from": src_location}

                    base_info = dict(env_result.info or {})
                    action_info = {
                        **base_info,
                        "utterance": decision.utterance,
                        "steering_snapshot": decision.steering_snapshot,
                        "persona_coeffs": agent.state.persona_coeffs.model_dump(),
                        "trait_key": trait_meta.trait_key,
                        "trait_band": trait_meta.trait_band,
                        "alpha_value": trait_meta.alpha_value,
                        "alpha_bucket": trait_meta.alpha_bucket,
                    }
                    action_log = ActionLog(
                        action_id=str(uuid4()),
                        run_id=self.run_id,
                        tick=self.world.tick,
                        agent_id=agent.state.agent_id,
                        action_type=env_result.action_type,
                        params=decision.params,
                        outcome="success" if env_result.success else "fail",
                        info=action_info,
                        prompt_text=decision.prompt_text,
                        prompt_hash=decision.prompt_hash,
                        plan_metadata=decision.plan_metadata,
                        reflection_summary=decision.reflection_summary,
                        reflection_implications=decision.reflection_implications,
                    )
                    tick_logs.append(action_log)
                    self.log_sink.log_action(action_log)
                    if probe_assignment:
                        self._log_probe_response(agent, decision, probe_assignment)
                        self.probe_manager.complete_probe(
                            agent.state.agent_id, probe_assignment, self.world.tick
                        )
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
                            prompt_hash=decision.prompt_hash,
                            prompt_text=decision.prompt_text,
                            plan_metadata=decision.plan_metadata,
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
                        collab_like = {"talk", "work", "research", "cite", "submit_report"}
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
                    try:
                        self.metric_tracker.on_message(msg_log)
                    except Exception:
                        pass

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
            tick_idx = self._log_tick_snapshots()
            self._warn_prompt_duplication(tick_idx)
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

        # Stop viewer if it has a stop method
        if self.event_bridge and hasattr(self.event_bridge, "stop"):
            try:
                self.event_bridge.stop()
            except Exception:
                pass

        return history

    def _trait_metadata(self, agent: Agent, steering_snapshot: Dict[str, float]) -> TraitMetadata:
        persona = agent.state.persona_coeffs.model_dump()
        snapshot = steering_snapshot or {}
        trait_key = trait_band_key(persona, snapshot)
        trait_name: Optional[str] = None
        trait_band: Optional[str] = None
        if trait_key and ":" in trait_key:
            trait_name, trait_band = trait_key.split(":", 1)
        elif trait_key:
            trait_name = trait_key
        alpha_value: Optional[float] = None
        if trait_name:
            try:
                alpha_value = float(snapshot.get(trait_name, 0.0))
            except (TypeError, ValueError):
                alpha_value = None
        alpha_bucket = self._alpha_bucket(alpha_value)
        return TraitMetadata(
            trait_key=trait_key,
            trait_name=trait_name,
            trait_band=trait_band,
            alpha_value=alpha_value,
            alpha_bucket=alpha_bucket,
        )

    def _alpha_bucket(self, alpha_value: Optional[float]) -> Optional[str]:
        if alpha_value is None:
            return None
        magnitude = abs(alpha_value)
        for label, lower, upper in MetricTracker.ALPHA_BUCKETS:
            lower_ok = lower is None or magnitude >= lower
            upper_ok = upper is None or magnitude < upper
            if lower_ok and upper_ok:
                return label
        return None

    def _emit_structured_logs(
        self,
        agent: Agent,
        env_result: actions.ActionResult,
        trait_meta: TraitMetadata,
    ) -> None:
        info = env_result.info or {}
        if env_result.action_type == "research":
            facts = self._coerce_fact_payload(info.get("facts_found"))
            doc_id = str(info.get("doc_id") or "")
            for fact in facts:
                fact_id = str(fact.get("fact_id") or "")
                if not fact_id:
                    continue
                fact_log = ResearchFactLog(
                    log_id=str(uuid4()),
                    run_id=self.run_id,
                    tick=self.world.tick,
                    agent_id=agent.state.agent_id,
                    doc_id=doc_id,
                    fact_id=fact_id,
                    fact_answer=str(fact.get("answer") or ""),
                    target_answer=fact.get("target_answer"),
                    correct=bool(fact.get("correct", False)),
                    trait_key=trait_meta.trait_key,
                    trait_band=trait_meta.trait_band,
                    alpha_value=trait_meta.alpha_value,
                    alpha_bucket=trait_meta.alpha_bucket,
                )
                self.log_sink.log_research_fact(fact_log)
                try:
                    self.metric_tracker.on_research_fact(fact_log)
                except Exception:
                    pass
        elif env_result.action_type == "cite":
            doc_id = str(info.get("doc_id") or "")
            if doc_id:
                citation_log = CitationLog(
                    log_id=str(uuid4()),
                    run_id=self.run_id,
                    tick=self.world.tick,
                    agent_id=agent.state.agent_id,
                    doc_id=doc_id,
                    trait_key=trait_meta.trait_key,
                    trait_band=trait_meta.trait_band,
                    alpha_value=trait_meta.alpha_value,
                    alpha_bucket=trait_meta.alpha_bucket,
                )
                self.log_sink.log_citation(citation_log)
                try:
                    self.metric_tracker.on_citation(citation_log)
                except Exception:
                    pass
        elif env_result.action_type == "submit_report":
            payload = info
            if isinstance(info, dict) and "grading" in info:
                payload = info["grading"]
            grade_payload = self._coerce_grade_payload(payload)
            if grade_payload:
                report_log = ReportGradeLog(
                    log_id=str(uuid4()),
                    run_id=self.run_id,
                    tick=self.world.tick,
                    agent_id=agent.state.agent_id,
                    targets_total=int(grade_payload.get("targets_total", 0)),
                    facts_correct=int(grade_payload.get("facts_correct", 0)),
                    citations_valid=int(grade_payload.get("citations_valid", 0)),
                    reward_points=float(grade_payload.get("reward_points", 0.0)),
                    trait_key=trait_meta.trait_key,
                    trait_band=trait_meta.trait_band,
                    alpha_value=trait_meta.alpha_value,
                    alpha_bucket=trait_meta.alpha_bucket,
                )
                self.log_sink.log_report_grade(report_log)
                try:
                    self.metric_tracker.on_report_grade(report_log)
                except Exception:
                    pass

    def _coerce_fact_payload(self, payload) -> List[Dict[str, object]]:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return []
        if isinstance(payload, list):
            return payload
        return []

    def _coerce_grade_payload(self, payload) -> Optional[Dict[str, object]]:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return None
        if isinstance(payload, dict):
            return payload
        return None

    def _log_tick_snapshots(self) -> int:
        tick_idx = max(0, self.world.tick - 1)
        try:
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
                    prompt_duplication_rate=macro.prompt_duplication_rate,
                    plan_reuse_rate=macro.plan_reuse_rate,
                )
                self.log_sink.log_metrics_snapshot(metrics_snapshot)
        except Exception:
            pass
        return tick_idx

    def _warn_prompt_duplication(self, tick_idx: int) -> None:
        if tick_idx != 0:
            return
        share, sample = self.tick_instrumentation.top_prompt_duplication()
        if share <= 0.4:
            return
        percent = share * 100
        snippet: str
        if sample:
            snippet = sample.strip()
            max_len = 200
            if len(snippet) > max_len:
                snippet = snippet[: max_len - 3] + "..."
        else:
            snippet = "<no prompt sample>"
        message = (
            f"Tick 0 prompt duplication rate is {percent:.0f}% for a single prompt hash. "
            f"Sample prompt: {snippet}"
        )
        self.console_logger.log_warning(message)

    def _log_probe_response(
        self,
        agent: Agent,
        decision: ActionDecision,
        assignment: ProbeAssignment,
    ) -> None:
        if not decision.utterance:
            return
        if assignment.kind == "likert":
            score, hint = ProbeManager.score_likert_response(decision.utterance)
            probe_log = ProbeLog(
                log_id=str(uuid4()),
                run_id=self.run_id,
                tick=self.world.tick,
                agent_id=agent.state.agent_id,
                probe_id=assignment.probe_id,
                trait=assignment.trait,
                question=assignment.question or "",
                prompt_text=assignment.prompt,
                response_text=decision.utterance,
                score=score,
                parser_hint=hint,
            )
            self.log_sink.log_probe(probe_log)
        else:
            outcome, hint = ProbeManager.score_behavior_response(assignment, decision.utterance)
            behavior_log = BehaviorProbeLog(
                log_id=str(uuid4()),
                run_id=self.run_id,
                tick=self.world.tick,
                agent_id=agent.state.agent_id,
                probe_id=assignment.probe_id,
                scenario=assignment.scenario or "",
                prompt_text=assignment.prompt,
                response_text=decision.utterance,
                outcome=outcome,
                parser_hint=hint,
            )
            self.log_sink.log_behavior_probe(behavior_log)
