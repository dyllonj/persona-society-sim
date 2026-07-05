"""Decision pipeline primitives for queue-native runner refactors."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from agents.agent import ActionDecision, Agent
from env.world import RoomUtterance
from schemas.agent import Rule


@dataclass(frozen=True)
class DecisionRequest:
    agent: Agent
    observation: str
    tick: int
    current_location: Optional[str]
    active_objective: Any
    recent_dialogue: Sequence[RoomUtterance]
    rule_context: Sequence[Rule]
    peers_present: bool
    alignment_context: Any = None


@dataclass(frozen=True)
class DecisionResult:
    agent_id: str
    tick: int
    decision: ActionDecision


class DecisionPipeline:
    """Protocol-like base for agent decision execution."""

    def decide(self, request: DecisionRequest) -> DecisionResult:
        raise NotImplementedError

    def decide_many(self, requests: Iterable[DecisionRequest]) -> List[DecisionResult]:
        return [self.decide(request) for request in requests]

    def close(self) -> None:
        return None


class SerialDecisionPipeline(DecisionPipeline):
    def decide(self, request: DecisionRequest) -> DecisionResult:
        decision = request.agent.act(
            request.observation,
            request.tick,
            current_location=request.current_location,
            active_objective=request.active_objective,
            recent_dialogue=tuple(request.recent_dialogue),
            rule_context=list(request.rule_context),
            peers_present=request.peers_present,
            alignment_context=request.alignment_context,
        )
        return DecisionResult(
            agent_id=request.agent.state.agent_id,
            tick=request.tick,
            decision=decision,
        )


class QueueDecisionPipeline(DecisionPipeline):
    """Worker-pool implementation with synchronous decision barriers.

    Current ``SimulationRunner`` call sites still wait for each decision before
    mutating world state, preserving behavior. The executor provides the queue
    seam needed for later batched or per-encounter scheduling.
    """

    def __init__(self, workers: int = 1) -> None:
        self.workers = max(1, int(workers))
        self._executor = ThreadPoolExecutor(
            max_workers=self.workers,
            thread_name_prefix="agent-decision",
        )

    def submit(self, request: DecisionRequest) -> Future[DecisionResult]:
        return self._executor.submit(SerialDecisionPipeline().decide, request)

    def decide(self, request: DecisionRequest) -> DecisionResult:
        return self.submit(request).result()

    def decide_many(self, requests: Iterable[DecisionRequest]) -> List[DecisionResult]:
        request_list = list(requests)
        futures = [self.submit(request) for request in request_list]
        results = [future.result() for future in futures]
        order = {
            id(request): idx
            for idx, request in enumerate(request_list)
        }
        result_order = {
            (request.agent.state.agent_id, request.tick): order[id(request)]
            for request in request_list
        }
        return sorted(
            results,
            key=lambda result: result_order.get((result.agent_id, result.tick), 0),
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


__all__ = [
    "DecisionPipeline",
    "DecisionRequest",
    "DecisionResult",
    "QueueDecisionPipeline",
    "SerialDecisionPipeline",
]
