# DD/SADD paper reproduction

This directory is a deliberately narrow research harness for qualitatively
reproducing the computational results in the DD/SADD manuscript: Figures 2--7
and the circuit-resolved source data behind Table II. Figure 1 is conceptual
artwork and is not generated here. This is not an IonShuttler simulation API;
nothing here is imported by `mqt.ionshuttler`.

From the repository root, install the locked paper-only environment and run the
small sanity check:

```console
uv sync --group paper
uv run python -m reproduce.paper.run quick --output reproduce/paper/output/quick
```

The declared paper matrix uses the same command with `paper` in place of
`quick`:

```console
uv run python -m reproduce.paper.run paper --output reproduce/paper/output/paper
```

`paper.toml` records the architecture, compiler/DD settings, simulation
parameters, grids, and seeds. Paper mode uses the manuscript's 20 circuits:
Ising, QFT, random, QPE, and GHZ at 4, 5, 6, and 8 qubits. The exact six-round
periodic Ising family is constructed locally; the other native-gate circuits are
loaded from the exact checked-in OpenQASM 2 inputs generated for the paper with
MQT Bench's Quantinuum target at optimization level 2. `quick` follows the same
compiler, five-method schedule construction, YAQS, aggregation, and plotting
path while selecting one case, one sample, and CI-sized grids. The full run
contains many independent tensor-network simulations and can take many hours or
longer, depending on the machine.

Normal runs also load the checked-in `circuits/compiled_schedules.json` bundle,
which freezes the exact NoDD, comparator, uniform-SADD, and profile-aware SADD
schedules used by the harness. Rebuilding schedules is an explicit maintainer
operation because parallel CP-SAT may choose a different equally solved Full
SADD materialization:

```console
uv run python -m reproduce.paper.run paper --regenerate-schedules \
  --samples 1 --output reproduce/paper/output/regenerated
```

The overwrite is atomic. Review the resulting schedule-resource CSV before
committing a regenerated bundle. Quick mode and ordinary paper mode never
overwrite frozen inputs.

Paper mode uses four worker processes by default. Override that coarse circuit
parallelism with `--workers N` (or use `--workers 1` for sequential debugging).
Each worker reuses one ideal final state for every noisy trajectory of its
circuit/sample task, avoiding repeated ideal simulations. The process pool
deliberately uses the Unix `fork` start method and is intended for the Linux/WSL
research environment in which this artifact was developed.

For an early full-matrix comparison, select a deterministic sample prefix:

```console
uv run python -m reproduce.paper.run paper --samples 8 --output reproduce/paper/output/paper
```

Execution is globally sample-major. The runner first constructs and stores the
exact schedules for every selected circuit, then evaluates sample 0 across the
entire circuit/scenario/method matrix before starting sample 1. Each completed
sample is committed as an atomic CSV shard and immediately used to refresh the
aggregate CSVs and figures. Repeating the command skips completed samples;
increasing `--samples` extends the same deterministic prefix, so the first eight
samples of a later 64-sample run are exactly the original eight.

The output directory contains inspectable CSV files for raw trajectories,
per-case and aggregate metrics, schedule/runtime summaries, Table II source
rows, objective/fidelity points, and temporal/spatial proxy values, plus six PDF
figures. Restart-safe state lives under `checkpoints/` in that output directory:
an integrity-checked compiled-schedule snapshot, a small manifest, one main CSV
per sample, and a tiny `table_ii_*.csv` supplement containing the three Table II
operating-point simulations. This supplement lets older completed main shards be
repaired without repeating their expensive trajectories. Scientific
configuration or Python-version changes require a new output directory instead
of silently mixing results; exact compiled schedules are restored from the
checkpoint. Do not run two writers against the same output directory
concurrently.

To redraw the PDFs from an existing output's CSV files without validating or
changing its checkpoints, run:

```console
uv run python -m reproduce.paper.run paper --plots-only \
  --output reproduce/paper/output/paper
```

## Interpretation and limits

The harness checks qualitative claims, not exact pixels or archival numerical
identity. In particular, inspect whether the plots show the expected method
ordering, a transition between dephasing-dominated and control/heating-dominated
regimes, a distinction between profile-aware and profile-agnostic SADD, and the
finite temporal/spatial range over which the quasistatic phase proxy remains
representative.

Circuits use IonShuttler's supported native rotations and require no benchmark
service at runtime. The random inputs retain their original seed-0 provenance.
The private adapter targets exactly `mqt-yaqs==0.6.0`; future YAQS versions are
not supported by this artifact. Its stochastic dephasing, pulse error, heating,
pure-state infidelity, and covariance proxy are the assumptions of this paper,
not a general physical model. The checked-in seed makes runs repeatable on a
fixed software stack, but floating-point and solver differences across platforms
can still produce small numerical changes.
