import pathlib
import sys
from datetime import datetime
from typing import Any

from .outside.compilation import (
    build_node_gate_id_lookup,
    create_dag,
    create_initial_sequence,
    create_updated_sequence_destructive,
)
from .outside.cycles import create_starting_config, get_ions
from .outside.graph_creator import GraphCreator, PZCreator
from .outside.partition import get_partition
from .outside.processing_zone import ProcessingZone
from .outside.shuttle import main as run_shuttle_main


def main(config: dict[str, Any]) -> None:
    # --- Extract Parameters from Config ---
    arch = config.get("arch")
    num_pzs_config = config.get("num_pzs", 1)
    seed = config.get("seed", 0)
    algorithm_name = config.get("algorithm_name")
    num_ions = config.get("num_ions")
    use_dag = config.get("use_dag", True)
    use_paths = config.get("use_paths", False)
    config.get("max_timesteps", 100000)
    plot_flag = config.get("plot", False)
    save_flag = config.get("save", False)
    failing_junctions = config.get("failing_junctions", 0)

    # Define base path for QASM files if needed
    qasm_base_dir_string = config.get("qasm_base_dir")
    if qasm_base_dir_string is None:
        qasm_base_dir = pathlib.Path(__file__).absolute().parent.parent.parent.parent / "inputs" / "qasm_files"
    else:
        qasm_base_dir = pathlib.Path(qasm_base_dir_string)

    # --- Validate Config ---
    if arch is None:
        msg = "Config parameter 'arch' is required but not set"
        raise ValueError(msg)

    if algorithm_name is None:
        msg = "Config parameter 'algorithm_name' is required but not set"
        raise ValueError(msg)

    if num_ions is None:
        msg = "Config parameter 'num_ions' is required but not set"
        raise ValueError(msg)

    if not isinstance(arch, list) or len(arch) != 4:
        msg = "Config parameter 'arch' must be a list of 4 integers [m, n, v, h]"
        raise ValueError(msg)

    # --- Setup ---
    start_time = datetime.now()
    cycle_or_paths_str = "Paths" if use_paths else "Cycles"
    m, n, v, h = arch

    # --- PZ Definitions ---
    height = -4.5
    pz_definitions = {
        "pz1": ProcessingZone(
            "pz1",
            [
                (float((m - 1) * v), float((n - 1) * h)),
                (float((m - 1) * v), float(0)),
                (float((m - 1) * v - height), float((n - 1) * h / 2)),
            ],
        ),
        "pz2": ProcessingZone("pz2", [(0.0, 0.0), (0.0, float((n - 1) * h)), (float(height), float((n - 1) * h / 2))]),
        "pz3": ProcessingZone(
            "pz3", [(float((m - 1) * v), float(0)), (float(0), float(0)), (float((m - 1) * v / 2), float(height))]
        ),
        "pz4": ProcessingZone(
            "pz4",
            [
                (float(0), float((n - 1) * h)),
                (float((m - 1) * v), float((n - 1) * h)),
                (float((m - 1) * v / 2), float((n - 1) * h - height)),
            ],
        ),
    }
    available_pz_names = list(pz_definitions.keys())
    pzs_to_use = [pz_definitions[name] for name in available_pz_names[:num_pzs_config]]

    if not pzs_to_use:
        print(f"Error: num_pzs ({num_pzs_config}) is invalid or results in no PZs selected.")
        sys.exit(1)

    print(f"Using {len(pzs_to_use)} PZs: {[pz.name for pz in pzs_to_use]}")
    print(f"Architecture: {arch}, Seed: {seed}")
    print(f"Algorithm: {algorithm_name}, ions: {num_ions}")
    print(f"DAG-Compilation: {use_dag}, Conflict Resolution: {cycle_or_paths_str}")

    # --- Graph Creation ---
    basegraph_creator = GraphCreator(m, n, v, h, failing_junctions, pzs_to_use)
    mz_graph = basegraph_creator.get_graph()
    pzgraph_creator = PZCreator(m, n, v, h, failing_junctions, pzs_to_use)
    graph = pzgraph_creator.get_graph()
    graph.mz_graph = mz_graph  # Attach MZ graph for BFS lookups if needed by Cycles/Paths

    graph.seed = seed
    graph.max_num_parking = 2
    graph.pzs = pzs_to_use  # List of ProcessingZone objects

    graph.plot = plot_flag
    graph.save = save_flag
    graph.arch = str(arch)  # For plotting/logging

    len(mz_graph.edges())

    print(f"Number of ions: {num_ions}")

    qasm_file_path = qasm_base_dir / algorithm_name / f"{algorithm_name}_{num_ions}.qasm"

    if not qasm_file_path.is_file():
        print(f"Error: QASM file not found at {qasm_file_path}")
        sys.exit(1)

    # --- Initial State & Sequence ---
    create_starting_config(graph, num_ions, seed=seed)
    graph.state = get_ions(graph)  # Get initial state {ion: edge_idc}

    initial_circuit = create_initial_sequence(qasm_file_path)
    graph.sequence = initial_circuit.sequence.copy()
    graph.gate_info = initial_circuit.gate_info
    gate_partition_cfg = config.get("gate_partition")
    gate_partition_algorithm_cfg = config.get("gate_partition_algorithm")
    graph.gate_pz_assignment = {}
    graph.current_gate_by_pz = {}
    graph.locked_gates = {}
    graph.dag_gate_id_lookup = {}
    gate_partition_for_run: dict[str, list[int]] | None = None
    gate_assignment: dict[int, str] = {}
    seq_length = len(graph.sequence)
    print(f"Number of Gates: {seq_length}")

    # --- Partitioning ---
    partitioning = True  # Make configurable
    partitions: dict[str, list[int]] = {}
    if partitioning:
        part = get_partition(qasm_file_path, len(graph.pzs))
        # Ensure partition list length matches num_pzs
        if len(part) != len(graph.pzs):
            print(f"Warning: Partitioning returned {len(part)} parts, but expected {len(graph.pzs)}. Adjusting...")
            # Simple fix: assign remaining qubits to the last partition, or distribute evenly.
            # This might need a more sophisticated balancing strategy.
            if len(part) < len(graph.pzs):
                print("Error: Partitioning failed to produce enough parts.")
                # Handle error appropriately, maybe fall back to non-partitioned approach or exit.
                sys.exit(1)
            else:  # More parts than PZs, merge extra parts into the last ones
                merged = [qubit for sublist in part[len(graph.pzs) - 1 :] for qubit in sublist]
                part = [*part[: len(graph.pzs) - 1], merged]

        partitions = {pz.name: part[i] for i, pz in enumerate(graph.pzs)}
        print(f"Partitions: {partitions}")
    else:
        # Fallback: Assign ions to closest PZ (example logic)
        print("Disabling Partitioning has to be implemented.")
        # TODO
        # ... (implement closest PZ assignment logic) ...

    # Create reverse map and validate partition
    map_to_pz: dict[int, str] = {}
    all_partition_elements = []
    for pz_name, elements in partitions.items():
        all_partition_elements.extend(elements)
        for element in elements:
            if element in map_to_pz:
                print(
                    f"Warning: Qubit {element} assigned to multiple partitions ({map_to_pz[element]}, {pz_name}). Check partitioning logic."
                )
            map_to_pz[element] = pz_name
    graph.map_to_pz = map_to_pz

    # Validation
    unique_sequence_qubits = {
        ion for gate_id in graph.sequence for ion in graph.gate_qubits(gate_id)
    }
    missing_qubits = unique_sequence_qubits - set(all_partition_elements)
    if missing_qubits:
        print(f"Error: Qubits {missing_qubits} from sequence are not in any partition.")
        # This indicates a problem with partitioning or qubit indexing.
        sys.exit(1)
    # Check for overlaps if needed (already done within map_to_pz creation loop)

    # --- DAG-Compilation Setup (if enabled) ---
    dag = None
    if use_dag:
        try:
            for pz in graph.pzs:
                pz.getting_processed = []
            dag = create_dag(qasm_file_path)
            graph.locked_gates = {}
            graph.dag_gate_id_lookup = build_node_gate_id_lookup(dag, graph.gate_info)
            dag.copy()  # Keep a copy of the original DAG if needed later
            # Initial DAG-based sequence update
            sequence, dag, gate_info = create_updated_sequence_destructive(
                graph, qasm_file_path, dag, use_dag=True
            )
            graph.sequence = sequence
            graph.gate_info = gate_info

        except Exception as e:
            print(f"Error during DAG creation or initial sequence update: {e}")
            print("Falling back to non-compiled sequence.")
            use_dag = False  # Disable use_dag if setup fails
            dag = None
            graph.sequence = initial_circuit.sequence.copy()  # Revert to basic sequence
            graph.gate_info = initial_circuit.gate_info
            graph.dag_gate_id_lookup = {}
    else:
        print("DAG disabled, using static QASM sequence.")
        graph.dag_gate_id_lookup = {}

    if gate_partition_cfg:
        gate_partition_for_run = {}
        for pz_name, gate_ids in gate_partition_cfg.items():
            gate_ids_int = [int(gate) for gate in gate_ids]
            gate_partition_for_run[pz_name] = gate_ids_int
            for gate_id in gate_ids_int:
                if gate_id in gate_assignment and gate_assignment[gate_id] != pz_name:
                    msg = (
                        f"Gate id {gate_id} assigned to multiple processing zones "
                        f"({gate_assignment[gate_id]}, {pz_name})."
                    )
                    raise ValueError(msg)
                gate_assignment[gate_id] = pz_name
    elif gate_partition_algorithm_cfg:
        if isinstance(gate_partition_algorithm_cfg, dict):
            algo_name = gate_partition_algorithm_cfg.get("name", "fgp_roee")
            algo_params = gate_partition_algorithm_cfg.get("params", {})
        else:
            algo_name = str(gate_partition_algorithm_cfg)
            algo_params = {}
        algo_name_lower = algo_name.lower()
        if algo_name_lower == "fgp_roee":
            from .outside.fgp_roee import compute_gate_partition

            if "num_clusters" not in algo_params:
                algo_params["num_clusters"] = len(graph.pzs)
            result = compute_gate_partition(graph, **algo_params)
            gate_partition_for_run = result.gate_partition_by_pz
            gate_assignment = result.gate_assignment
        else:
            msg = f"Unknown gate partition algorithm '{algo_name}'."
            raise ValueError(msg)

    graph.gate_pz_assignment = gate_assignment
    graph.current_gate_by_pz = {}

    # --- Run Simulation ---

    # Initialize PZ states
    for pz in graph.pzs:
        pz.getting_processed = []  # Track nodes being processed by this PZ

    print("\nStarted shuttling simulation...")

    # Run the main shuttling logic
    final_timesteps = run_shuttle_main(
        graph,
        dag,
        cycle_or_paths_str,
        use_dag=use_dag,
        gate_partition=gate_partition_for_run,
    )

    # --- Results ---
    end_time = datetime.now()
    cpu_time = end_time - start_time

    print(f"\nSimulation finished in {final_timesteps} timesteps.")
    print(f"Total CPU time: {cpu_time}")

    # # --- Benchmarking Output ---
    # bench_filename = f"benchmarks/{start_time.strftime('%Y%m%d_%H%M%S')}_{algorithm_name}.txt"
    # pathlib.Path("benchmarks").mkdir(exist_ok=True)
    # benchmark_output = (
    #     f"{arch}, ions{num_ions}/pos{number_of_mz_edges}: {num_ions/number_of_mz_edges if number_of_mz_edges > 0 else 0:.2f}, "
    #     f"#pzs: {len(pzs_to_use)}, ts: {final_timesteps}, cpu_time: {cpu_time.total_seconds():.2f}, "
    #     f"gates: {seq_length}, baseline: {None}, DAG-Compilation: {use_dag}, paths: {use_paths}, "
    #     f"seed: {seed}, failing_jcts: {failing_junctions}\n"
    # )
    # try:
    #     with open(bench_filename, "a") as f:
    #         f.write(benchmark_output)
    #     print(f"Benchmark results appended to {bench_filename}")
    # except Exception as e:
    #     print(f"Warning: Could not write benchmark file: {e}")
