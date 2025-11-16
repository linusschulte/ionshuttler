import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from itertools import product
from typing import Any

import h5py
import numpy as np


from outside.compilation import (
    build_node_gate_id_lookup,
    create_dag,
    create_initial_sequence,
    create_updated_sequence_destructive,
)
from outside.cycles import create_starting_config, get_ions
from outside.graph_creator import GraphCreator, PZCreator
from outside.partition import get_partition
from outside.processing_zone import ProcessingZone
from outside.shuttle import main as run_shuttle_main

LEGACY_CLI_COMMAND = "mqt-ionshuttler-heuristic"
_SIM_TIMESTEPS_RE = re.compile(r"Simulation finished in\s+(?P<value>\d+)\s+timesteps")
_CPU_TIME_RE = re.compile(r"Total CPU time:\s+(?P<value>.+)")


def _parse_cpu_time(value: str) -> timedelta:
    normalized = value.strip()
    days = 0
    if "day" in normalized:
        day_part, normalized = normalized.split(",", 1)
        day_tokens = day_part.strip().split()
        if day_tokens:
            days = int(day_tokens[0])
        normalized = normalized.strip()
    hours_str, minutes_str, seconds_str = normalized.split(":")
    hours = int(hours_str)
    minutes = int(minutes_str)
    seconds = float(seconds_str)
    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def _extract_legacy_metrics(output: str) -> tuple[int, timedelta]:
    final_timesteps: int | None = None
    cpu_time: timedelta | None = None
    for line in output.splitlines():
        if final_timesteps is None:
            timestep_match = _SIM_TIMESTEPS_RE.search(line)
            if timestep_match:
                final_timesteps = int(timestep_match.group("value"))
        if cpu_time is None:
            cpu_match = _CPU_TIME_RE.search(line)
            if cpu_match:
                cpu_time = _parse_cpu_time(cpu_match.group("value"))
    if final_timesteps is None or cpu_time is None:
        msg = "Unable to parse legacy CLI output for timesteps or CPU time."
        raise RuntimeError(f"{msg}\nFull output:\n{output}")
    return final_timesteps, cpu_time


def run_legacy_cli_with_config(config: dict[str, Any]) -> tuple[int, timedelta]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp_file:
        json.dump(config, tmp_file)
        tmp_path = pathlib.Path(tmp_file.name)
    try:
        completed = subprocess.run(
            [LEGACY_CLI_COMMAND, str(tmp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Legacy CLI command failed with exit code {exc.returncode}.\nOutput:\n{exc.stdout or ''}"
        ) from exc
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
    return _extract_legacy_metrics(completed.stdout)


def should_use_legacy_cli(config: dict[str, Any]) -> bool:
    return (
        config.get("gate_partition_algorithm") is None
        and config.get("gate_partition") is None
    )

def main(config: dict[str, Any]):
    # --- Extract Parameters from Config ---
    arch = config.get("arch")
    num_pzs_config = config.get("num_pzs", 1)
    max_ions_per_pz = config.get("max_ions_per_pz", 2)
    seed = config.get("seed", 0)
    algorithm_name = config.get("algorithm_name")
    num_ions = config.get("num_ions")
    use_dag = config.get("use_dag", True)
    use_paths = config.get("use_paths", False)
    max_timesteps = config.get("max_timesteps", 100000)
    plot_flag = config.get("plot", False)
    save_flag = config.get("save", False)
    failing_junctions = config.get("failing_junctions", 0)
    debug_gate_tracking = config.get("debug_gate_tracking", False)


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
    height = -2.5
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
    graph.max_num_parking = max_ions_per_pz
    graph.pzs = pzs_to_use  # List of ProcessingZone objects

    graph.plot = plot_flag
    graph.save = save_flag
    graph.arch = str(arch)  # For plotting/logging
    graph.debug_gate_tracking = debug_gate_tracking

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
    enforce_slice_plan = config.get("enforce_slice_plan", True)
    graph.gate_pz_assignment = {}
    graph.current_gate_by_pz = {}
    graph.locked_gates = {}
    graph.dag_gate_id_lookup = {}
    graph.initialize_slice_plan(None)
    gate_partition_for_run: dict[str, list[int]] | None = None
    gate_assignment: dict[int, str] = {}
    if graph.sequence:
        first_gate_id = graph.sequence[0]
        gate_meta = graph.gate_info[first_gate_id]
    seq_length = len(graph.sequence)
    print(f"Number of Gates: {seq_length}")

    # --- Partitioning (legacy) ---
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
            graph.dag_gate_id_lookup = build_node_gate_id_lookup(dag, gate_info=graph.gate_info)
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
        if enforce_slice_plan:
            graph.initialize_slice_plan(None)
    elif gate_partition_algorithm_cfg:
        if isinstance(gate_partition_algorithm_cfg, dict):
            algo_name = gate_partition_algorithm_cfg.get("name", "fgp_roee")
            algo_params = gate_partition_algorithm_cfg.get("params", {})
        else:
            algo_name = str(gate_partition_algorithm_cfg)
            algo_params = {}
        algo_name_lower = algo_name.lower()
        if algo_name_lower == "fgp_roee":
            from outside.fgp_roee import compute_gate_partition

            if "num_clusters" not in algo_params:
                algo_params["num_clusters"] = config.get("num_pzs", 1)
            if "capacity" not in algo_params:
                algo_params["capacity"] = config.get("max_ions_per_pz", 1)
            print(algo_params)
            result = compute_gate_partition(graph, **algo_params)
            gate_partition_for_run = result.gate_partition_by_pz
            gate_assignment = result.gate_assignment
            if enforce_slice_plan:
                graph.initialize_slice_plan(result.slice_plan)
            else:
                graph.initialize_slice_plan(None)
        elif algo_name_lower == "tdag":
            if not use_dag or dag is None:
                msg = "TDAG gate partitioning requires use_dag=True."
                raise ValueError(msg)
            from outside.tdag import compute_gate_partition_tdag

            max_qubits = algo_params.get("k", 4)
            tdag_result = compute_gate_partition_tdag(
                graph,
                dag,
                max_qubits_per_block=max_qubits,
            )
            pz_names = [pz.name for pz in graph.pzs]
            gate_partition_for_run = {pz_name: [] for pz_name in pz_names}
            gate_assignment = {}
            blocks = tdag_result.get("blocks", [])
            for idx, block in enumerate(blocks):
                pz_name = pz_names[idx % len(pz_names)]
                gate_partition_for_run[pz_name].extend(block)
                for gate_id in block:
                    gate_assignment[gate_id] = pz_name
            graph.initialize_slice_plan(None)
        else:
            msg = f"Unknown gate partition algorithm '{algo_name}'."
            raise ValueError(msg)
    else:
        graph.initialize_slice_plan(None)

    slice_plan_for_run = graph.slice_plan if enforce_slice_plan else None
    if not enforce_slice_plan:
        graph.initialize_slice_plan(None)

    graph.gate_pz_assignment = gate_assignment
    graph.current_gate_by_pz = {}

    if gate_assignment:
        print("Gate assignment to PZs:")
        for pz_name, gate_ids in gate_partition_for_run.items():
            print(f"  {pz_name}: {gate_ids}")
        if enforce_slice_plan:
            print("Enforcing slice plan based on gate partitioning.")
            for slice in slice_plan_for_run:
                print(f"  Slice: {slice}")

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
        slice_plan=slice_plan_for_run,
        max_timesteps=max_timesteps,
    )

    # --- Results ---
    end_time = datetime.now()
    cpu_time = end_time - start_time

    print(f"\nSimulation finished in {final_timesteps} timesteps.")
    print(f"Total CPU time: {cpu_time}")

    return final_timesteps, cpu_time


def execute_run(config: dict[str, Any]) -> tuple[int, timedelta]:
    config_for_run = config.copy()
    if should_use_legacy_cli(config_for_run):
        print("Using legacy CLI entrypoint (mqt-ionshuttler-heuristic) for this configuration.")
        return run_legacy_cli_with_config(config_for_run)
    return main(config_for_run)

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute heuristic shuttling schedules")
    parser.add_argument("config_file", help="Path to the JSON configuration file")
    parser.add_argument("--run_meta_study", help="Run meta study with parameter sweeps y/n", default="n")
    args = parser.parse_args()

    try:
        with pathlib.Path(args.config_file).open("r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON file {args.config_file}")
        sys.exit(1)

    if args.run_meta_study.lower() != "y":
        #run single compilation
        main(config)
        exit()

    # Use a single results file (not datetime specific)
    results_file = f"outputs/simulation_results_{config['algorithm_name']}.h5"
    pathlib.Path("outputs").mkdir(exist_ok=True)
    
    # Helper function to check if parameter set exists
    def parameter_set_exists(f, num_pz, ions_pz, grid_size, mz_trap_size, use_dag, partitioning_alg, enforce_slice):
        if 'results' not in f:
            return False
        results_group = f['results']
        for run_name in results_group.keys():
            run_group = results_group[run_name]
            if (run_group.attrs.get('num_pzs') == num_pz and
                run_group.attrs.get('ions_per_pz') == ions_pz and
                run_group.attrs.get('grid_size') == grid_size and
                run_group.attrs.get('mz_trap_size') == mz_trap_size and
                run_group.attrs.get('use_dag') == use_dag and
                run_group.attrs.get('partitioning_algorithm') == (str(partitioning_alg) if partitioning_alg else 'none') and
                run_group.attrs.get('enforce_slice_plan') == enforce_slice):
                return True
        return False
    
    # meta study overwrite:
    num_ions = [5,10,15]
    num_pzs = [2,3]
    ions_per_pz = [2,3]
    grid_sizes = [3,5,7]
    mz_trap_sizes = [1]
    dag_options = [True, False]
    partitioning_options = [None, config.get('gate_partition_algorithm')]  # None means no partitioning algorithm
    enforce_slice_plan_options = [True, False]
    
    valid_combinations = [
        (num_ion, num_pz, ions_pz, grid_size, mz_trap_size, use_dag, partitioning_alg, enforce_slice)
        for num_ion, num_pz, ions_pz, grid_size, mz_trap_size, use_dag, partitioning_alg, enforce_slice in product(
            num_ions, num_pzs, ions_per_pz, grid_sizes, mz_trap_sizes, dag_options, partitioning_options, enforce_slice_plan_options
        )
        if not (enforce_slice and partitioning_alg is None)
    ]
    
    # Open file in append mode, create if doesn't exist
    with h5py.File(results_file, 'a') as f:
        # Initialize results group if it doesn't exist
        if 'results' not in f:
            results_group = f.create_group('results')
            f.attrs['algorithm_name'] = config.get('algorithm_name', 'unknown')
            f.attrs['base_num_ions'] = config.get('num_ions', 0)
            f.attrs['seed'] = config.get('seed', 0)
            f.attrs['created_at'] = datetime.now().isoformat()
        else:
            results_group = f['results']
        
        # Count existing runs to determine next index
        existing_runs = len(list(results_group.keys()))
        result_index = existing_runs
        
        total_combinations = len(valid_combinations)
        skipped = 0
        
        print(f"Total combinations: {total_combinations}")
        print(f"Existing runs in file: {existing_runs}")
        
        for num_ion, num_pz, ions_pz, grid_size, mz_trap_size, use_dag, partitioning_alg, enforce_slice in valid_combinations:
            # Check if this parameter set already exists
            if parameter_set_exists(f, num_pz, ions_pz, grid_size, mz_trap_size, use_dag, partitioning_alg, enforce_slice):
                print(f"\nSkipping existing parameter set: {num_ion} ions on {grid_size}x{grid_size} grid, MZ {mz_trap_size}, PZs {num_pz}, Ions/PZ {ions_pz}, DAG {use_dag}, Part {partitioning_alg}, Enforce {enforce_slice}")
                skipped += 1
                continue
            
            # Update config for this run
            config["num_ions"] = num_ion
            config["arch"] = [grid_size, grid_size, mz_trap_size, mz_trap_size]
            config["num_pzs"] = num_pz
            config["max_ions_per_pz"] = ions_pz
            config["use_dag"] = use_dag
            config["enforce_slice_plan"] = enforce_slice
            
            if partitioning_alg is None:
                config.pop("gate_partition_algorithm", None)
            else:
                config["gate_partition_algorithm"] = partitioning_alg
                if isinstance(partitioning_alg, dict):
                    params = config["gate_partition_algorithm"].setdefault("params", {})
                    params["num_clusters"] = num_pz
                    params["capacity"] = ions_pz
            
            print(f"\n=== Run {result_index + 1} (New) ===")
            print(f"Grid: {grid_size}x{grid_size}, MZ trap size: {mz_trap_size}")
            print(f"PZs: {num_pz}, Ions per PZ: {ions_pz}")
            print(f"DAG: {use_dag}, Partitioning: {partitioning_alg}, Enforce slice: {enforce_slice}")
            
            run_name = f'run_{result_index:04d}'
            run_group = results_group.create_group(run_name)

            run_group.attrs['num_ions'] = num_ion
            run_group.attrs['num_pzs'] = num_pz
            run_group.attrs['ions_per_pz'] = ions_pz
            run_group.attrs['grid_size'] = grid_size
            run_group.attrs['mz_trap_size'] = mz_trap_size
            run_group.attrs['use_dag'] = use_dag
            run_group.attrs['partitioning_algorithm'] = str(partitioning_alg) if partitioning_alg else 'none'
            run_group.attrs['enforce_slice_plan'] = enforce_slice
            
            try:
                final_timesteps, cpu_time = execute_run(config)
                if final_timesteps >= config.get("max_timesteps", 100000) - 1:
                    run_group.attrs['success'] = False
                    run_group.attrs['error_message'] = f"Simulation reached max timesteps ({final_timesteps})"
                else:
                    run_group.attrs['success'] = True
                    run_group.attrs['final_timesteps'] = final_timesteps
                run_group.attrs['cpu_time_seconds'] = cpu_time.total_seconds()
                
                
                print(f"Completed: {final_timesteps} timesteps, {cpu_time.total_seconds():.2f}s CPU time")
                
            except Exception as e:
                print(f"Failed: {str(e)}")
                
                run_group.attrs['error_message'] = str(e)
                run_group.attrs['success'] = False
            
            result_index += 1
    
    print(f"\nAll simulations completed. Skipped {skipped} existing parameter sets.")
    print(f"Results saved to {results_file}")
