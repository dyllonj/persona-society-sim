"""Command-line entry point for running persona society simulations."""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import numpy as np
import torch
import yaml

from agents.agent import Agent
from agents.language_backend import HFBackend, LanguageBackend, MockBackend
from agents.memory import MemoryStore
from agents.planner import Planner
from agents.retrieval import MemoryRetriever
from env.world import World
from orchestrator.runner import SimulationRunner
from orchestrator.scheduler import Scheduler
from safety.governor import SafetyConfig, SafetyGovernor
from schemas.agent import AgentState, PersonaCoeffs
from storage.log_sink import LogSink

TRAIT_KEYS = ["E", "A", "C", "O", "N"]
GOAL_LIBRARY = [
    "organize meetup",
    "research town policy",
    "write collaborative brief",
    "support market trade",
    "improve community wellbeing",
    "explore new ideas",
]


def load_config(path: Path) -> Dict:
    return yaml.safe_load(path.read_text())


def sample_persona(base: Dict[str, float], rng: random.Random) -> PersonaCoeffs:
    def jitter(value: float) -> float:
        return max(-1.5, min(1.5, value + rng.uniform(-0.4, 0.4)))

    return PersonaCoeffs(**{trait: jitter(base.get(trait, 0.0)) for trait in TRAIT_KEYS})


def load_trait_vectors(traits: List[str], layers: List[int], vector_dir: Path) -> Dict[str, Dict[int, np.ndarray]]:
    store: Dict[str, Dict[int, np.ndarray]] = {}
    for trait in traits:
        per_layer: Dict[int, np.ndarray] = {}
        for layer in layers:
            path = vector_dir / f"{trait}.layer{layer}.npy"
            if path.exists():
                per_layer[layer] = np.load(path)
        if per_layer:
            store[trait] = per_layer
    return store


def build_language_backend(
    config: Dict,
    layers: List[int],
    trait_vectors: Dict[str, Dict[int, np.ndarray]],
    mock: bool,
) -> LanguageBackend:
    inference = config.get("inference", {})
    temperature = inference.get("temperature", 0.7)
    top_p = inference.get("top_p", 0.9)
    if mock:
        return MockBackend(seed=config.get("seed", 0), temperature=temperature, top_p=top_p)
    torch_vectors = {
        trait: {layer: torch.tensor(vector) for layer, vector in per_layer.items()}
        for trait, per_layer in trait_vectors.items()
    }
    return HFBackend(
        model_name=config["model_name"],
        layers=layers,
        trait_vectors=torch_vectors,
        temperature=temperature,
        top_p=top_p,
    )


def build_agents(
    run_id: str,
    config: Dict,
    world: World,
    backend: LanguageBackend,
    safety_governor: SafetyGovernor,
) -> List[Agent]:
    population = config["population"]
    rng = random.Random(config.get("seed", 7))
    base_persona = config.get("steering", {})
    inference = config.get("inference", {})
    max_tokens = inference.get("max_new_tokens", 120)
    locations = list(world.locations.keys())
    agents: List[Agent] = []
    for idx in range(population):
        agent_id = f"agent-{idx:03d}"
        display_name = f"Agent {idx:03d}"
        persona = sample_persona(base_persona, rng)
        location = rng.choice(locations)
        goals = rng.sample(GOAL_LIBRARY, k=min(2, len(GOAL_LIBRARY)))
        state = AgentState(
            agent_id=agent_id,
            display_name=display_name,
            persona_coeffs=persona,
            steering_refs=[],
            system_prompt="Town resident",
            location_id=location,
            goals=goals,
            created_at=datetime.utcnow(),
            last_tick=0,
        )
        world.add_agent(agent_id, location)
        memory = MemoryStore()
        retriever = MemoryRetriever(memory)
        planner = Planner()
        agent = Agent(
            run_id=run_id,
            state=state,
            language_backend=backend,
            memory=memory,
            retriever=retriever,
            planner=planner,
            safety_governor=safety_governor,
            max_new_tokens=max_tokens,
        )
        agents.append(agent)
    return agents


def main() -> None:
    parser = argparse.ArgumentParser(description="Persona Society Sim runner")
    parser.add_argument("config", type=Path, help="Path to YAML config")
    parser.add_argument("--mock-model", action="store_true", help="Use mock backend instead of HF model")
    parser.add_argument("--max-events", type=int, default=16, help="Max encounters per tick")
    parser.add_argument("--vector-dir", type=Path, default=Path("data/vectors"), help="Directory with steering vectors")
    args = parser.parse_args()

    config = load_config(args.config)
    run_id = config.get("run_id") or f"run-{uuid4().hex[:6]}"
    layers = config.get("layers", [])
    steering_base = config.get("steering", {})
    trait_vectors = load_trait_vectors(list(steering_base.keys() or TRAIT_KEYS), layers, args.vector_dir)
    safety_cfg = config.get("safety", {})
    safety = SafetyGovernor(
        SafetyConfig(
            alpha_clip=safety_cfg.get("alpha_clip", 1.5),
            toxicity_threshold=safety_cfg.get("toxicity_threshold", 0.4),
            governor_backoff=safety_cfg.get("governor_backoff", 0.2),
        )
    )
    backend = build_language_backend(config, layers, trait_vectors, mock=args.mock_model)
    world = World()
    scheduler = Scheduler(world, seed=config.get("seed", 7))
    agents = build_agents(run_id, config, world, backend, safety)
    logging_cfg = config.get("logging", {})
    log_sink = LogSink(run_id, logging_cfg.get("db_url"), logging_cfg.get("parquet_dir"))
    inference = config.get("inference", {})
    runner = SimulationRunner(
        run_id=run_id,
        world=world,
        scheduler=scheduler,
        agents=agents,
        log_sink=log_sink,
        temperature=inference.get("temperature", 0.7),
        top_p=inference.get("top_p", 0.9),
    )
    runner.run(config.get("steps", 200), max_events_per_tick=args.max_events)
    print(f"Run {run_id} completed {config.get('steps', 200)} steps with {len(agents)} agents.")


if __name__ == "__main__":
    main()
