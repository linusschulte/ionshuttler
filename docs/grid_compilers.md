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

The compiler uses deterministic breadth-first routing. It can move ordered ion
chains and can rotate ions simultaneously around a full cycle. It schedules
ready gates together when they use independent ions and processing zones. Search
is bounded by {py:class}`mqt.ionshuttler.grid.GridCompilerConfig`. A routing
failure returns a validated `FAILED` result with the completed schedule prefix.

This first compiler favors clear behavior and complete search on small
architectures. It does not yet contain the legacy path, cycle, partitioning,
home-zone, caching, and priority policies used for larger workloads. The
hardware model accepts simultaneous cycles with mixed ion-chain sizes, but the
minimal router generates only cycles made from one-ion moves to keep its
candidate set bounded.

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
Grid compiler is a separate API. It does not adapt into the legacy mutable run
loop. Existing grid workflows remain available while their policies move to the
new compiler.

## See also

- {doc}`hardware_models` — compare the Linear and grid abstractions
- {doc}`references` — publications describing the exact and heuristic methods
