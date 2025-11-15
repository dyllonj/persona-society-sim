"""Command-line entry point for running persona society simulations."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
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
from orchestrator.console_logger import ConsoleLogger
from orchestrator.objectives import ObjectiveManager
from orchestrator.probes import ProbeManager
from orchestrator.runner import SimulationRunner
from orchestrator.scheduler import Scheduler
from safety.governor import SafetyConfig, SafetyGovernor
from schemas.agent import AgentState, PersonaCoeffs
from schemas.objectives import ObjectiveTemplate, DEFAULT_OBJECTIVE_TEMPLATES
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


def load_trait_vectors(traits: List[str], vector_dir: Path) -> Dict[str, Dict[int, np.ndarray]]:
    store: Dict[str, Dict[int, np.ndarray]] = {}
    for trait in traits:
        meta_path = vector_dir / f"{trait}.meta.json"
        if not meta_path.exists():
            continue
        metadata = json.loads(meta_path.read_text())
        layer_entries = {entry["layer_id"]: entry for entry in metadata.get("layers", [])}
        preferred = metadata.get("preferred_layers") or []
        ordered_layers: List[int] = []
        seen = set()
        for layer_id in preferred:
            if layer_id in layer_entries and layer_id not in seen:
                ordered_layers.append(layer_id)
                seen.add(layer_id)
        if not ordered_layers:
            ordered_layers = sorted(layer_entries.keys())
        per_layer: Dict[int, np.ndarray] = {}
        for layer_id in ordered_layers:
            entry = layer_entries.get(layer_id)
            if not entry:
                continue
            vector_path_value = entry.get("vector_path")
            if not vector_path_value:
                continue
            vec_path = Path(vector_path_value)
            if not vec_path.exists():
                candidate = (meta_path.parent / vec_path).resolve()
                if candidate.exists():
                    vec_path = candidate
            if not vec_path.exists():
                continue
            vector = np.load(vec_path, allow_pickle=False)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            per_layer[layer_id] = vector.astype(np.float32)
        if per_layer:
            store[trait] = per_layer
    return store


def build_language_backend(
    config: Dict,
    trait_vectors: Dict[str, Dict[int, np.ndarray]],
    mock: bool,
) -> LanguageBackend:
    inference = config.get("inference", {})
    optimization = config.get("optimization", {})
    temperature = inference.get("temperature", 0.7)
    top_p = inference.get("top_p", 0.9)
    use_quantization = optimization.get("use_quantization", False)

    if mock:
        return MockBackend(seed=config.get("seed", 0), temperature=temperature, top_p=top_p)
    torch_vectors = {
        trait: {layer: torch.tensor(vector) for layer, vector in per_layer.items()}
        for trait, per_layer in trait_vectors.items()
    }
    return HFBackend(
        model_name=config["model_name"],
        trait_vectors=torch_vectors,
        temperature=temperature,
        top_p=top_p,
        use_quantization=use_quantization,
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
    optimization = config.get("optimization", {})
    max_tokens = inference.get("max_new_tokens", 120)
    reflect_every_n = optimization.get("reflect_every_n_ticks", 1)
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
            reflect_every_n_ticks=reflect_every_n,
        )
        agents.append(agent)
    return agents


def build_objective_manager(config: Dict, env_choice: str, difficulty: int) -> Optional[ObjectiveManager]:
    objectives_cfg = config.get("objectives") or {}
    if not objectives_cfg.get("enabled", False):
        return None
    templates_cfg = objectives_cfg.get("templates") or {}
    templates: Dict[str, ObjectiveTemplate] = {}
    for name, payload in templates_cfg.items():
        requirements = payload.get("requirements", {})
        if not requirements:
            continue
        templates[name] = ObjectiveTemplate(
            name=name,
            type=payload.get("type", name),
            description=payload.get("description", name),
            requirements=requirements,
            reward=payload.get("reward", {}),
        )
    if not templates:
        diff = max(1, difficulty)
        if env_choice == "research":
            base = DEFAULT_OBJECTIVE_TEMPLATES["research_facts"]
            requirements = {"research": diff, "submit_report": 1}
            templates = {
                base.name: ObjectiveTemplate(
                    name=base.name,
                    type=base.type,
                    description=base.description,
                    requirements=requirements,
                    reward=base.reward,
                )
            }
        elif env_choice == "policy":
            base = DEFAULT_OBJECTIVE_TEMPLATES["policy_checklist"]
            requirements = {"fill_field": diff, "submit_plan": 1}
            templates = {
                base.name: ObjectiveTemplate(
                    name=base.name,
                    type=base.type,
                    description=base.description,
                    requirements=requirements,
                    reward=base.reward,
                )
            }
        else:  # nav
            base = DEFAULT_OBJECTIVE_TEMPLATES["navigation_discovery"]
            requirements = {"scan": diff}
            templates = {
                base.name: ObjectiveTemplate(
                    name=base.name,
                    type=base.type,
                    description=base.description,
                    requirements=requirements,
                    reward=base.reward,
                )
            }
    return ObjectiveManager(
        templates=templates,
        enabled=True,
        seed=objectives_cfg.get("seed", config.get("seed", 7)),
    )


def build_probe_manager(config: Dict) -> Optional[ProbeManager]:
    probes_cfg = config.get("probes")
    return ProbeManager.from_config(probes_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Persona Society Sim runner")
    parser.add_argument("config", type=Path, help="Path to YAML config")
    parser.add_argument("--mock-model", action="store_true", help="Use mock backend instead of HF model")
    parser.add_argument("--max-events", type=int, default=16, help="Max encounters per tick")
    parser.add_argument("--vector-dir", type=Path, default=Path("data/vectors"), help="Directory with steering vectors")
    parser.add_argument("--env", choices=["research", "policy", "nav"], default="research", help="Select experiment environment (research, policy, nav)")
    parser.add_argument("--difficulty", type=int, default=3, help="Difficulty parameter (facts, checklist fields, or tokens)")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live console output showing agent actions and dialogues",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument(
        "--full-messages",
        action="store_true",
        help="Show complete dialogue content in live console output",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Start WebSocket bridge and static web viewer (http://127.0.0.1:8000)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_id = config.get("run_id") or f"run-{uuid4().hex[:6]}"
    steering_base = config.get("steering", {})
    trait_vectors = load_trait_vectors(list(steering_base.keys() or TRAIT_KEYS), args.vector_dir)
    safety_cfg = config.get("safety", {})
    safety = SafetyGovernor(
        SafetyConfig(
            alpha_clip=safety_cfg.get("alpha_clip", 1.5),
            toxicity_threshold=safety_cfg.get("toxicity_threshold", 0.4),
            governor_backoff=safety_cfg.get("governor_backoff", 0.2),
        )
    )
    backend = build_language_backend(config, trait_vectors, mock=args.mock_model)
    world = World()
    world.configure_environment(args.env, args.difficulty)
    scheduler = Scheduler(world, seed=config.get("seed", 7))
    agents = build_agents(run_id, config, world, backend, safety)
    logging_cfg = config.get("logging", {})
    log_sink = LogSink(run_id, logging_cfg.get("db_url"), logging_cfg.get("parquet_dir"))
    inference = config.get("inference", {})
    objective_manager = build_objective_manager(config, args.env, args.difficulty)
    probe_manager = build_probe_manager(config)

    # Create console logger if live mode is enabled
    console_logger = ConsoleLogger(
        enabled=args.live,
        use_colors=not args.no_color,
        truncate=not args.full_messages,
    )
    if args.live:
        console_logger.log_info(f"Starting simulation: {run_id}")
        console_logger.log_info(f"Agents: {len(agents)}, Steps: {config.get('steps', 200)}, Events/tick: {args.max_events}")

    # Optionally start viewer bridge
    event_bridge = None
    http_server = None
    if args.viewer:
        try:
            from viewer.ws_bridge import ViewerServer
            from viewer.http_server import StaticServer

            event_bridge = ViewerServer()
            event_bridge.start()
            http_server = StaticServer()
            http_server.start()
            if args.live:
                console_logger.log_info("Viewer: WebSocket ws://127.0.0.1:8765/ws")
                console_logger.log_info("Viewer: Open http://127.0.0.1:8000 in your browser")
        except Exception as e:
            if args.live:
                console_logger.log_warning(f"Failed to start viewer: {e}")

    runner = SimulationRunner(
        run_id=run_id,
        world=world,
        scheduler=scheduler,
        agents=agents,
        log_sink=log_sink,
        temperature=inference.get("temperature", 0.7),
        top_p=inference.get("top_p", 0.9),
        console_logger=console_logger,
        objective_manager=objective_manager,
        probe_manager=probe_manager,
        event_bridge=event_bridge,
    )
    runner.run(config.get("steps", 200), max_events_per_tick=args.max_events)

    if not args.live:
        print(f"Run {run_id} completed {config.get('steps', 200)} steps with {len(agents)} agents.")


if __name__ == "__main__":
    main()
