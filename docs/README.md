# Documentation index

Organized using the [Diataxis](https://diataxis.fr/) framework: tutorials
teach, how-tos accomplish a task, reference describes precisely, explanation
clarifies why.

## Tutorial

- [tutorial-getting-started.md](tutorial-getting-started.md) — run your first simulation, no GPU or API key needed.

## How-to guides

- [howto-run-simulations.md](howto-run-simulations.md) — environments, backends, viewing modes, throughput flags.
- [howto-compute-steering-vectors.md](howto-compute-steering-vectors.md) — extract and evaluate CAA persona vectors.
- [howto-migrate-legacy-data.md](howto-migrate-legacy-data.md) — convert old prompt schemas; clean up pre-removal `trade` records.
- [experiments/autoresearch/RUNBOOK.md](../experiments/autoresearch/RUNBOOK.md) — operating the autoresearch candidate/eval/promotion pipeline.

## Reference

- [reference-cli.md](reference-cli.md) — every `orchestrator.cli` flag.
- [reference-config.md](reference-config.md) — every YAML config schema (run configs, steering metadata, playbooks, probes, personas).
- [reference-modules.md](reference-modules.md) — public surface of the agent loop, world model, and viewer modes.
- [reference-data-schema.md](reference-data-schema.md) — every SQL table / Parquet dataset the simulator writes.
- [experiments/autoresearch/README.md](../experiments/autoresearch/README.md) — autoresearch pipeline command reference.

## Explanation

- [design.md](design.md) — system architecture and the reasoning behind the major structural choices.
- [explanation-steering.md](explanation-steering.md) — why CAA instead of prompting, in depth.
- [eval.md](eval.md) — evaluation protocols for the project's research questions.
- [explanation-known-gaps.md](explanation-known-gaps.md) — every live bug, dead config field, and doc/code mismatch found while writing this documentation, in one place.
- [jacobian-lens-integration.md](jacobian-lens-integration.md) — implementation contract for reproducible inference manifests, post-hoc Jacobian Lens replay, research design, and GPU budget gates.

## Also see

- [../README.md](../README.md) — project overview and quickstart.
- [../AGENTS.md](../AGENTS.md) — agent cognitive loop and telemetry, from the agent-instrumentation angle.
- [../INSTALL.md](../INSTALL.md) — installation.
- [../OPTIMIZATION_GUIDE.md](../OPTIMIZATION_GUIDE.md) — performance tuning.
