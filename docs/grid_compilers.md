# Grid Compilers

Grid compilers route ions through graph-based segments, junctions, and
processing zones. The new `GridCompiler` provides a small common-contract
compiler. The established exact and heuristic command-line tools remain
available during the migration.

Use the {doc}`Linear compiler <linear_compiler>` when the compilation task must
resolve individual sites and local transport within a trap segment. The
{doc}`hardware model overview <hardware_models>` compares the two abstraction
levels.

## Common-contract compiler

{py:class}`mqt.ionshuttler.grid.GridCompiler` accepts a shared circuit input and
returns a shared {py:class}`mqt.ionshuttler.core.result.CompilationResult` with
a replayable Grid schedule and Grid-specific diagnostics.

```python
from mqt.ionshuttler.grid import GridArchitecture, GridCompiler, Junction, ProcessingZone, Segment

memory = Segment("memory", capacity=2)
processor = Segment("processor", capacity=2, processing_zones=(ProcessingZone("pz"),))
architecture = GridArchitecture(
    segments=(memory, processor),
    junctions=(Junction("memory-processor", (memory.end, processor.start)),),
)
circuit = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
rxx(0.5) q[0],q[1];
"""

result = GridCompiler(architecture).compile(
    circuit,
    initial_placement={"memory": (0, 1)},
)
result.validate()
```

The default strategy uses deterministic breadth-first routing. It can move
ordered ion chains and can rotate ions simultaneously around a full cycle. It
schedules ready gates together when they use independent ions and processing
zones. Search is bounded by {py:class}`mqt.ionshuttler.grid.GridCompilerConfig`.
A routing failure returns a validated `FAILED` result with the completed
schedule prefix.

Alternatively, the greedy strategy uses a short dependency horizon to select one
nearby gate for each processing zone. It builds an ion priority from those gates
and later circuit gates. Several ions can advance toward the same processing
zone in one timestep. Each transport candidate requests the next segment of a
shortest route. If that segment is full, the router completes the requested
junction crossing with a topology cycle or clears the blocked path from its free
end. The scheduler then accepts compatible transport candidates in priority
order.

Independent gates and transport actions can share a timestep. Transport on
unrelated resources can also continue while a longer gate remains active. The
greedy strategy assigns each gate to the nearest reachable processing zone that
supports it. A two-ion gate keeps this assignment while its operands move. The
greedy scheduler retains operands that are already at the processing zone for a
selected gate. Other ions on the processing-zone segment can leave while a gate
runs.

```python
from mqt.ionshuttler.grid import GridCompilerConfig, GridCompilerStrategy

compiler = GridCompiler(
    architecture,
    GridCompilerConfig(
        strategy=GridCompilerStrategy.GREEDY,
        allowed_junction_crossings=frozenset({("memory", "processor")}),
    ),
)
result = compiler.compile(
    circuit,
    initial_placement={"memory": (0, 1)},
)
```

`allowed_junction_crossings` is an optional routing policy. It can describe
one-way circulation without making transport direction a property of the Grid
hardware model.

The breadth-first strategy favors complete search on small architectures. The
greedy strategy favors bounded local work and concurrent transport. It does not
yet provide the path, hybrid, partitioning, or path-cache policies available in
the established heuristic tools. The hardware model accepts simultaneous cycles
with mixed ion-chain sizes, but the greedy router generates cycles from one-ion
moves to keep its candidate set bounded. The compiler façade delegates to
separate breadth-first and greedy schedulers, so a new strategy does not add
another branch to the scheduling loop.

## Exact compilation

The exact compiler targets small architectures with one processing zone. It
searches for a minimum-cost shuttling solution and is most useful as a reference
for compact instances.

```console
mqt-ionshuttler-exact --help
mqt-ionshuttler-exact inputs/algorithms_exact/qft_06.json
```

Pass `--plot` to visualize the result. Example architecture and algorithm files
are available in
[`inputs/algorithms_exact`](https://github.com/munich-quantum-toolkit/ionshuttler/tree/main/inputs/algorithms_exact).

## Heuristic compilation

The heuristic compiler scales to larger circuits and supports one or several
processing zones. It trades an optimality guarantee for practical runtime.

```console
mqt-ionshuttler-heuristic --help
mqt-ionshuttler-heuristic inputs/algorithms_heuristic/qft_60_4pzs.json
```

Example inputs are available in
[`inputs/algorithms_heuristic`](https://github.com/munich-quantum-toolkit/ionshuttler/tree/main/inputs/algorithms_heuristic).

The optional dependency-aware mode can schedule ready gates according to the
current ion positions instead of following one fixed gate sequence. The shared
fine-grained tabu partitioner is available from
{py:mod}`mqt.ionshuttler.partitioning`.

## Interface status

The established tools keep their command-line and JSON interfaces. The common
Grid compiler is a separate API. It does not adapt into the old mutable run
loop. Existing grid workflows remain available while their policies move to the
new compiler.

## See also

- {doc}`hardware_models` — compare the Linear and grid abstractions
- {doc}`references` — publications describing the exact and heuristic methods
