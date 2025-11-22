"""Command-line entry point for running persona society simulations."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import torch
import sys
import traceback
import yaml

from agents.agent import Agent
from agents.gemini_backend import GeminiBackend
from agents.language_backend import HFBackend, LanguageBackend, MockBackend
from agents.memory import MemoryStore
from agents.planner import Planner
from agents.retrieval import MemoryRetriever
from env.world import World
from orchestrator.console_logger import ConsoleLogger
from orchestrator.meta_manager import MetaOrchestrator
from orchestrator.objectives import ObjectiveManager
from orchestrator.probes import ProbeManager
from orchestrator.runner import SimulationRunner
from orchestrator.scheduler import Scheduler
from safety.governor import SafetyConfig, SafetyGovernor
from schemas.agent import AgentState, PersonaCoeffs
from schemas.objectives import ObjectiveTemplate, DEFAULT_OBJECTIVE_TEMPLATES
from steering.vector_store import VectorStore
from storage.log_sink import LogSink

TRAIT_KEYS = PersonaCoeffs.TRAITS
GOAL_LIBRARY = [
    "organize meetup",
    "research town policy",
    "write collaborative brief",
    "share research findings",
    "document sources at the library",
    "improve community wellbeing",
    "explore new ideas",
]

ROLE_ROSTERS: Dict[str, List[Tuple[str, str]]] = {
    "research": [
        ("Principal Investigator", "Defines the research agenda and synthesizes findings."),
        ("Field Researcher", "Interviews participants and gathers first-hand observations."),
        ("Citation Librarian", "Tracks sources and validates references."),
        ("Report Assembler", "Organizes notes into coherent drafts."),
        ("QA Reviewer", "Checks accuracy, tone, and completeness before submission."),
    ],
    "policy": [
        ("Requirements Lead", "Collects stakeholder needs and clarifies constraints."),
        ("Field Owner", "Provides domain context and real-world examples."),
        ("Policy Drafter", "Writes policy language and resolves ambiguities."),
        ("Compliance Submitter", "Prepares forms and ensures alignment with standards."),
        ("QA Auditor", "Reviews policies for gaps, risks, and adherence."),
    ],
    "nav": [
        ("Route Planner", "Charts efficient paths between rooms."),
        ("Room Scout", "Surveys rooms for obstacles or opportunities."),
        ("Signal Relay", "Passes navigation updates between teammates."),
        ("Data Logger", "Records positions, events, and timing."),
        ("Recovery/Support", "Assists stuck agents and coordinates regrouping."),
    ],
}


def load_config(path: Path) -> Dict:
    return yaml.safe_load(path.read_text())


def _load_metadata_file(path_value: Optional[str], *, config_dir: Optional[Path]) -> Dict[str, Any]:
    if not path_value:
        return {}
    candidate = Path(path_value)
    if not candidate.is_absolute():
        if config_dir is not None:
            candidate = (config_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()
    if not candidate.exists():
        return {}
    try:
        data = yaml.safe_load(candidate.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def sample_persona(base: Dict[str, float], rng: random.Random, jitter: float = 0.2) -> PersonaCoeffs:
    jitter = max(0.0, float(jitter))

    def jitter_value(value: float) -> float:
        if jitter == 0.0:
            return value
        return value + rng.uniform(-jitter, jitter)

    raw: Dict[str, float] = {}
    for trait in TRAIT_KEYS:
        value = base.get(trait, 0.0)
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = 0.0
        raw[trait] = jitter_value(numeric_value)
    return PersonaCoeffs(**raw)


def _persona_sampling_jitter(steering_cfg: Dict[str, Any], *, config_dir: Optional[Path]) -> float:
    metadata_files = steering_cfg.get("metadata_files") or {}
    personas_meta = _load_metadata_file(metadata_files.get("personas"), config_dir=config_dir)
    if not isinstance(personas_meta, dict):
        return 0.2
    sampling_cfg = personas_meta.get("sampling") or {}
    try:
        return float(sampling_cfg.get("jitter", 0.2))
    except (TypeError, ValueError):
        return 0.2


def _env_roster(env_choice: str) -> List[Tuple[str, str]]:
    return ROLE_ROSTERS.get(env_choice, ROLE_ROSTERS["research"])


def _assign_roles(population: int, roster: List[Tuple[str, str]], rng: random.Random) -> List[Tuple[str, str]]:
    if not roster:
        return [("Generalist", "Supports the team across tasks.")] * population
    if population <= len(roster):
        return roster[:population]
    assignments = roster[:]
    rng.shuffle(assignments)
    while len(assignments) < population:
        assignments.append(rng.choice(roster))
    return assignments[:population]


def _role_system_prompt(base_prompt: str, role: str, role_description: Optional[str]) -> str:
    if role_description:
        return f"{base_prompt} Role: {role} — {role_description}"
    return f"{base_prompt} Role: {role}."


def _steering_coefficients(steering_cfg: Dict[str, Any]) -> Dict[str, float]:
    if not steering_cfg:
        return {}
    coeffs = steering_cfg.get("coefficients")
    if isinstance(coeffs, dict):
        return {trait: float(value) for trait, value in coeffs.items()}
    legacy: Dict[str, float] = {}
    for trait in TRAIT_KEYS:
        value = steering_cfg.get(trait)
        if isinstance(value, (int, float)):
            legacy[trait] = float(value)
    return legacy


def load_trait_vectors(
    traits: List[str],
    vector_dir: Path,
    vector_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[str, Dict[int, float]]]:
    metadata_cfg = vector_metadata or {}
    vector_root = Path(metadata_cfg.get("vector_root") or vector_dir)
    defaults = metadata_cfg.get("defaults") or {}
    default_layers = defaults.get("layers")
    trait_overrides = metadata_cfg.get("traits") or {}
    store = VectorStore(vector_root)
    trait_vectors: Dict[str, Dict[int, np.ndarray]] = {}
    norms: Dict[str, Dict[int, float]] = {}
    fallback_traits: List[str] = []
    for trait in traits:
        override = (
            trait_overrides.get(trait)
            or trait_overrides.get(trait.lower())
            or trait_overrides.get(trait.upper())
        )
        vector_store_id = override.get("vector_store_id") if override else None
        layers = override.get("layers") if override else None
        if layers is None:
            layers = default_layers
        try:
            bundle = store.load(vector_store_id or trait, layers=layers)
        except Exception:
            fallback_traits.append(trait)
            continue
        trait_vectors[trait] = bundle.vectors
        layer_norms: Dict[int, float] = {}
        for layer_id, layer_meta in bundle.layer_metadata.items():
            norm_val = layer_meta.get("norm")
            if norm_val is not None:
                layer_norms[layer_id] = float(norm_val)
        if layer_norms:
            norms[trait] = layer_norms
    if fallback_traits:
        legacy_vectors, legacy_norms = _load_vectors_from_meta_files(
            fallback_traits, vector_root
        )
        trait_vectors.update(legacy_vectors)
        for trait, per_layer in legacy_norms.items():
            norms.setdefault(trait, {}).update(per_layer)
    return trait_vectors, norms


def shuffle_trait_vectors(
    trait_vectors: Dict[str, Dict[int, np.ndarray]],
    vector_norms: Dict[str, Dict[int, float]],
    rng: random.Random,
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[str, Dict[int, float]], Dict[str, str]]:
    """Shuffle trait-to-vector assignments to create a placebo steering map.

    Returns the shuffled vectors, shuffled norms, and the mapping from trait ->
    source trait used for the reassignment.
    """

    traits = list(trait_vectors.keys())
    if not traits:
        return trait_vectors, vector_norms, {}
    shuffled = traits[:]
    rng.shuffle(shuffled)
    mapping = dict(zip(traits, shuffled))

    shuffled_vectors: Dict[str, Dict[int, np.ndarray]] = {}
    shuffled_norms: Dict[str, Dict[int, float]] = {}
    for trait, source in mapping.items():
        shuffled_vectors[trait] = dict(trait_vectors.get(source, {}))
        if source in vector_norms:
            shuffled_norms[trait] = dict(vector_norms[source])
    return shuffled_vectors, shuffled_norms, mapping


def _load_vectors_from_meta_files(
    traits: List[str], vector_dir: Path
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[str, Dict[int, float]]]:
    store: Dict[str, Dict[int, np.ndarray]] = {}
    norms: Dict[str, Dict[int, float]] = {}
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
            norms.setdefault(trait, {})[layer_id] = float(norm)
            if norm > 0:
                vector = vector / norm
            per_layer[layer_id] = vector.astype(np.float32)
        if per_layer:
            store[trait] = per_layer
    return store, norms


def build_language_backend(
    config: Dict,
    trait_vectors: Dict[str, Dict[int, np.ndarray]],
    vector_norms: Dict[str, Dict[int, float]],
    mock: bool,
    use_gemini: bool = False,
    *,
    suppress_alphas: bool = False,
) -> LanguageBackend:
    inference = config.get("inference", {})
    optimization = config.get("optimization", {})
    temperature = inference.get("temperature", 0.7)
    top_p = inference.get("top_p", 0.9)
    use_quantization = optimization.get("use_quantization", False)
    steering_cfg = config.get("steering", {})
    alpha_strength = steering_cfg.get("strength", 1.0)
    if suppress_alphas:
        alpha_strength = 0.0
    max_gpu_memory = _maybe_float(optimization.get("max_gpu_memory_gb"))
    max_cpu_memory = _maybe_float(optimization.get("max_cpu_memory_gb"))
    raw_offload = optimization.get("offload_folder")
    offload_folder = str(raw_offload) if raw_offload else None

    if mock:
        return MockBackend(
            seed=config.get("seed", 0),
            temperature=temperature,
            top_p=top_p,
            alpha_strength=alpha_strength,
            suppress_alphas=suppress_alphas,
        )
    
    if use_gemini:
        return GeminiBackend(
            model_name="gemini-1.5-flash", # Could be configurable via config
            temperature=temperature,
            top_p=top_p,
            alpha_strength=alpha_strength,
            suppress_alphas=suppress_alphas,
        )

    torch_vectors = {
        trait: {layer: torch.tensor(vector) for layer, vector in per_layer.items()}
        for trait, per_layer in trait_vectors.items()
    }
    return HFBackend(
        model_name=config["model_name"],
        trait_vectors=torch_vectors,
        vector_norms=vector_norms,
        temperature=temperature,
        top_p=top_p,
        use_quantization=use_quantization,
        alpha_strength=alpha_strength,
        max_gpu_memory_gb=max_gpu_memory,
        max_cpu_memory_gb=max_cpu_memory,
        offload_folder=offload_folder,
        suppress_alphas=suppress_alphas,
    )


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_agents(
    run_id: str,
    config: Dict,
    world: World,
    backend: LanguageBackend,
    safety_governor: SafetyGovernor,
    env_choice: str,
    *,
    config_dir: Optional[Path] = None,
    suppress_alphas: bool = False,
) -> List[Agent]:
    population = config["population"]
    rng = random.Random(config.get("seed", 7))
    steering_cfg = config.get("steering", {})
    base_persona = {}
    persona_jitter = 0.0
    if not suppress_alphas:
        base_persona = _steering_coefficients(steering_cfg)
        persona_jitter = _persona_sampling_jitter(steering_cfg, config_dir=config_dir)
    inference = config.get("inference", {})
    optimization = config.get("optimization", {})
    max_tokens = inference.get("max_new_tokens", 120)
    reflect_every_n = optimization.get("reflect_every_n_ticks", 1)
    locations = list(world.locations.keys())
    role_roster = _env_roster(env_choice)
    role_assignments = _assign_roles(population, role_roster, rng)
    prompt_prefix = {
        "research": "You are part of a collaborative research sprint.",
        "policy": "You are collaborating on a policy drafting mission.",
        "nav": "You are coordinating a navigation and exploration mission.",
    }.get(env_choice, "You are contributing to the simulation objectives.")
    agents: List[Agent] = []
    for idx in range(population):
        agent_id = f"agent-{idx:03d}"
        display_name = f"Agent {idx:03d}"
        persona = sample_persona(base_persona, rng, jitter=persona_jitter)
        location = rng.choice(locations)
        role, role_description = role_assignments[idx]
        goals = rng.sample(GOAL_LIBRARY, k=min(2, len(GOAL_LIBRARY)))
        role_goal = f"Fulfill {role} duties"
        if role_description:
            role_goal = f"{role_goal}: {role_description}"
        goals.insert(0, role_goal)
        system_prompt = _role_system_prompt(prompt_prefix, role, role_description)
        state = AgentState(
            agent_id=agent_id,
            display_name=display_name,
            persona_coeffs=persona,
            steering_refs=[],
            role=role,
            role_description=role_description,
            system_prompt=system_prompt,
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
            suppress_alphas=suppress_alphas,
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


def build_meta_orchestrator(
    config: Dict, env_choice: str, *, config_dir: Optional[Path]
) -> Optional[MetaOrchestrator]:
    meta_cfg = config.get("meta_orchestrator")
    if meta_cfg is None:
        return None
    if not isinstance(meta_cfg, dict):
        return None
    if meta_cfg.get("enabled", True) is False:
        return None

    base_playbook = MetaOrchestrator.default_role_playbook()
    file_overrides = _load_metadata_file(
        meta_cfg.get("playbook_file"), config_dir=config_dir
    )
    inline_overrides = meta_cfg.get("role_playbook")
    if not isinstance(inline_overrides, dict):
        inline_overrides = {}
    role_playbook = MetaOrchestrator.merge_playbooks(base_playbook, file_overrides)
    role_playbook = MetaOrchestrator.merge_playbooks(role_playbook, inline_overrides)

    return MetaOrchestrator(
        global_goals=meta_cfg.get("global_goals"),
        recurring_reminders=meta_cfg.get("recurring_reminders"),
        agent_directives=meta_cfg.get("agent_directives"),
        role_playbook=role_playbook,
        environment=env_choice,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Persona Society Sim runner")
    parser.add_argument("config", type=Path, help="Path to YAML config")
    parser.add_argument("--mock-model", action="store_true", help="Use mock backend instead of HF model")
    parser.add_argument("--gemini", action="store_true", help="Use Gemini backend")
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
        help="Start WebSocket bridge and static web viewer (http://127.0.0.1:19123)",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Enable ASCII TUI visualization (replaces web viewer)",
    )
    parser.add_argument(
        "--no-steering",
        action="store_true",
        help="Disable persona steering vectors and run agents with neutral traits",
    )
    parser.add_argument(
        "--steering-mode",
        choices=["targeted", "placebo", "disabled"],
        default="targeted",
        help=(
            "Select steering preset: targeted (default vectors), placebo (randomly shuffle "
            "trait/vector assignments), or disabled (no steering)."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_id = config.get("run_id") or f"run-{uuid4().hex[:6]}"
    steering_cfg = config.get("steering", {})
    steering_mode = args.steering_mode
    if args.no_steering:
        steering_mode = "disabled"
    steering_enabled = bool(steering_cfg.get("enabled", True)) and steering_mode != "disabled"
    alpha_strength = float(steering_cfg.get("strength", 1.0)) if steering_enabled else 0.0
    steering_base = _steering_coefficients(steering_cfg) if steering_enabled else {}
    metadata_files = steering_cfg.get("metadata_files") or {}
    trait_vectors: Dict[str, Dict[int, np.ndarray]] = {}
    vector_norms: Dict[str, Dict[int, float]] = {}
    shuffle_mapping: Dict[str, str] = {}
    vector_metadata: Dict[str, Any] = {}
    if steering_enabled:
        vector_metadata = _load_metadata_file(
            metadata_files.get("vectors"), config_dir=args.config.parent
        )
        trait_vectors, vector_norms = load_trait_vectors(
            list(steering_base.keys() or TRAIT_KEYS),
            args.vector_dir,
            vector_metadata=vector_metadata,
        )
        if steering_mode == "placebo":
            rng = random.Random(config.get("seed", 7))
            trait_vectors, vector_norms, shuffle_mapping = shuffle_trait_vectors(
                trait_vectors, vector_norms, rng
            )
    safety_cfg = config.get("safety", {})
    safety = SafetyGovernor(
        SafetyConfig(
            alpha_clip=safety_cfg.get("alpha_clip", 1.0),
            toxicity_threshold=safety_cfg.get("toxicity_threshold", 0.4),
            governor_backoff=safety_cfg.get("governor_backoff", 0.2),
            global_alpha_strength=alpha_strength,
        )
    )
    backend = build_language_backend(
        config,
        trait_vectors,
        vector_norms,
        mock=args.mock_model,
        use_gemini=args.gemini,
        suppress_alphas=not steering_enabled,
    )
    world = World()
    world.configure_environment(args.env, args.difficulty)
    scheduler = Scheduler(world, seed=config.get("seed", 7))
    agents = build_agents(
        run_id,
        config,
        world,
        backend,
        safety,
        args.env,
        config_dir=args.config.parent,
        suppress_alphas=not steering_enabled,
    )
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
    if args.live and shuffle_mapping:
        console_logger.log_info(
            "Placebo steering enabled: trait/vector mapping shuffled %s"
            % shuffle_mapping
        )
    if args.live:
        console_logger.log_info(f"Starting simulation: {run_id}")
        console_logger.log_info(f"Agents: {len(agents)}, Steps: {config.get('steps', 200)}, Events/tick: {args.max_events}")

    # Optionally start viewer bridge
    event_bridge = None
    http_server = None
    
    try:
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
                    console_logger.log_info("Viewer: Open http://127.0.0.1:19123 in your browser")
            except Exception as e:
                if args.live:
                    console_logger.log_warning(f"Failed to start viewer: {e}")

        if args.tui:
            from viewer.ascii_tui import AsciiViewer
            event_bridge = AsciiViewer()
            # Disable console logger to avoid interference with TUI
            console_logger = ConsoleLogger(enabled=False)
            # Also disable standard print output from runner summary if possible,
            # though runner uses console_logger mostly.

        meta_orchestrator = build_meta_orchestrator(
            config, args.env, config_dir=args.config.parent
        )

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
            meta_orchestrator=meta_orchestrator,
        )
        runner.run(config.get("steps", 200), max_events_per_tick=args.max_events)

        if not args.live:
            print(f"Run {run_id} completed {config.get('steps', 200)} steps with {len(agents)} agents.")

    except Exception:
        # Ensure TUI is stopped before printing traceback so it's visible
        if event_bridge and hasattr(event_bridge, "stop"):
            try:
                event_bridge.stop()
            except Exception:
                pass
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Double check cleanup
        if event_bridge and hasattr(event_bridge, "stop"):
            try:
                event_bridge.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
