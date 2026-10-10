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

## Visualize a result

{py:func}`mqt.ionshuttler.visualize` returns an interactive
{py:class}`~mqt.ionshuttler.visualization.GridView` for a Grid result. In a
notebook, the view shows itself as the cell output. Elsewhere, save it as an
HTML file and open it in a browser. The file contains all data and needs no
network access.

```python
from mqt.ionshuttler import visualize

view = visualize(result)
view.save("schedule.html")
```

The view draws the schedule on a canvas. Use **Play**, **Previous layer**,
**Next layer**, and the time slider to move through the schedule. Ions that
cross a junction travel from their segment through the junction into the
destination segment. Simultaneous actions move together. Running gates highlight
their ions and processing zone. The header shows the time and the actions of the
running layer.

### Result viewer

The result viewer is one browser page for all results. It runs in your Python
process and listens only on this computer. Python replays each result; the page
draws it.

```python
from mqt.ionshuttler.visualization import GridVisualizer, open_viewer

visualizer = GridVisualizer(theme="dark")
visualizer.open(result)  # show one result
visualizer.compare({"first": first, "second": second}).open()  # show a comparison
open_viewer()  # open the viewer without a result
```

Use **Open result…** in the page, or drop a file on it, to show a result saved
with `result.save("result.json")`. The **View** menu switches between all
results of the session. In a notebook, `open` returns at once and the viewer
runs as long as the kernel. In a plain script, `open` waits until you press
Ctrl+C, so that the viewer keeps running. Under WSL, the viewer opens in the
Windows browser. The returned address also opens the viewer by hand. The viewer
shows Grid results; Linear results are not supported yet.

### Settings

{py:class}`~mqt.ionshuttler.visualization.GridVisualizer` holds the display
settings. The browser controls start from these values.

```python
from mqt.ionshuttler.visualization import GridVisualizer

visualizer = GridVisualizer(
    theme="dark",
    show_ion_labels=True,
    show_processing_zone_labels=False,
    show_hardware_ids=True,
    timesteps_per_second=8,
)
view = visualizer.visualize(result)
```

`theme` is `"dark"` (the default), `"light"`, or `"auto"`, which follows the
browser. Figures and videos use the same theme; pass `theme="light"` for print.
`show_hardware_ids` writes segment and junction IDs. `timesteps_per_second` sets
the playback speed. `width` and `height` set the view and video size in pixels.

The rectangular and square generators use junction IDs of the form
`j:row:column`. The visualizer uses those IDs to recover the generated lattice.
For other architectures, including hexagonal grids, it first attempts a planar
layout. A non-planar graph uses a deterministic force-directed layout instead.
To control the drawing, pass explicit coordinates. The mapping must contain each
junction exactly once. Keys can be junction IDs or `Junction` values.

```python
visualizer = GridVisualizer(junction_coordinates={"memory-processor": (0.0, 0.0)})
```

Coordinates belong to the visualization, not `GridArchitecture`. Changing a
drawing therefore does not change compilation, replay, or serialized hardware.

To arrange a drawing by hand, click **Edit layout** in the view and drag
junctions. Segments and ions follow at once, also during playback.
**Copy coordinates** copies the current positions as Python source for
`junction_coordinates`. For a comparison, it copies one mapping per panel for
the `junction_coordinates` argument of `compare`. **Reset** restores the
original layout.

### Ion and processing-zone colors

Each ion is drawn as a disk with a colored ring. By default, every ion has its
own ring color. Each processing zone also has its own color. A running gate
draws a thin outer ring around its ions in that color and thickens its zone.
`ion_colors="single"` gives all ions one ring color. A mapping sets the colors
of chosen ions, for example to show the role of each ion. A color can change at
a schedule time, and a label adds a legend entry:

```python
from mqt.ionshuttler.visualization import GridVisualizer, IonColor

data = IonColor(border="tab:blue", label="data")
idle = IonColor(border="#94a3b8", label="ancilla")
active = IonColor(border="crimson", fill="#fee2e2", label="active ancilla")
visualizer = GridVisualizer(
    ion_colors={0: data, 1: data, 2: [(0, idle), (120, active)]},
    processing_zone_colors={"pz": "teal"},
)
```

Colors accept any Matplotlib color. Ions without an entry have a gray ring. The
**Ion colors** switch in the view changes between distinct, single, and custom
colors. In **Edit layout** mode, click an ion or a processing zone to change its
color. **Copy settings** then also copies the changed colors.

### Video export

Open **Export video…** in the view and drag the two handles to select the first
and last timestep. The export uses the current theme, labels, speed, and frame
rate. The video shows `timesteps_per_second` timesteps per second, so the frame
rate changes only smoothness. The browser draws each frame at a fixed schedule
time and does not wait for playback. It writes a WebM file and needs a browser
with WebCodecs video encoding, such as a current Chromium-based browser.

Scripts can export a video without a browser. This uses Matplotlib with the same
layout, colors, and labels:

```python
view = GridVisualizer(theme="dark", video_start_time=315, video_end_time=420).visualize(result)
view.export_video("interesting.gif", frames_per_second=30)
view.export_video("interesting.mp4", start_time=315, end_time=360)
```

A `.gif` file needs no further software. The `.mp4`, `.m4v`, `.mov`, `.mkv`, and
`.webm` formats need the FFmpeg program.

### Static figures and comparisons

{py:meth}`~mqt.ionshuttler.visualization.GridVisualizer.plot` draws one schedule
time as a Matplotlib figure, for example for a paper:

```python
visualizer.plot(result, 120).savefig("t120.png", dpi=200)
```

{py:meth}`~mqt.ionshuttler.visualization.GridVisualizer.compare` plays several
results side by side with one clock. The clock uses absolute schedule time. A
result that ends earlier keeps showing its final state.

```python
view = visualizer.compare({"breadth-first": first, "greedy": second})
```

Results on different architectures can use their own explicit coordinates:

```python
view = visualizer.compare(
    {"loop": loop_result, "square": square_result},
    junction_coordinates={"loop": loop_coordinates},
)
```

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
