import argparse
import json
import pathlib
import random
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from itertools import product
from typing import Any
from qiskit import QuantumCircuit

import h5py
import numpy as np
from scipy.stats import qmc
from inside import plotting as plotting_mod, shuttle as shuttle_mod
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

PRINT_DEBUG = 0

from inside.cycles import create_starting_config, get_ion_chains
from inside.graph_creator import GraphCreator
from inside.helper import generate_pzs
from inside.shuttle import main as run_shuttle_main


import sys
from pathlib import Path
# To enable importing existing types from outside
sys.path.insert(0, str(Path(__file__).resolve().parent))

from outside.compilation import compile_qasm_file as compile_qasm
from outside.types import GateInfo
from outside.partition import get_partition
#from outside.compilation import create_initial_sequence

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

def _calculate_timestep_lower_bound(sequence: list[int], gate_info: dict[int, GateInfo], num_pzs: int) -> int:
    timestep_lower_bound = 0
    for gate_id in sequence:
        qubits = gate_info[gate_id].qubits
        timestep_lower_bound += 3 if len(qubits) > 1 else 1
    return timestep_lower_bound // max(1, num_pzs)


def _infer_num_ions_from_qasm(qasm_file_path: pathlib.Path) -> int:
    qc = QuantumCircuit.from_qasm_file(str(qasm_file_path))
    return qc.num_qubits


def _resolve_max_timesteps(config: dict[str, Any], num_ions: int | None) -> int:
    max_timesteps = config.get("max_timesteps")
    if max_timesteps is None:
        max_timesteps_factor = config.get("max_timesteps_factor")
        if max_timesteps_factor is not None:
            if num_ions is None:
                msg = "Config parameter 'num_ions' is required when using 'max_timesteps_factor'."
                raise ValueError(msg)
            max_timesteps = int(max_timesteps_factor * num_ions)
        else:
            max_timesteps = 100000
    return int(max_timesteps)


def _format_site(site: tuple[int, int]) -> str:
    return f"({site[0]}, {site[1]})"


def _edge_to_strings(edge: tuple[tuple[int, int], tuple[int, int]]) -> list[str]:
    return [_format_site(edge[0]), _format_site(edge[1])]


def _collect_ion_edges(graph: Any) -> dict[int, tuple[tuple[int, int], tuple[int, int]]]:
    ion_edges: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for u, v, data in graph.edges(data=True):
        ions = data.get("ions", [])
        edge = tuple(sorted((u, v), key=sum))
        for ion in ions:
            ion_edges[int(ion)] = edge
    return ion_edges


def _removed_gates(prev: list[tuple[int, ...]], current: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    remaining = list(current)
    removed: list[tuple[int, ...]] = []
    for gate in prev:
        if gate in remaining:
            remaining.remove(gate)
        else:
            removed.append(gate)
    return removed


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
        and config.get("use_legacy_implementation", False) is True
        #and config.get("use_dag") is True
    )

def main(config: dict[str, Any]):
    arch = config.get("arch")
    num_pzs_config = int(config.get("num_pzs", 1))
    max_ions_per_pz = int(config.get("max_ions_per_pz", 2))
    seed = int(config.get("seed", 0))
    algorithm_name = config.get("algorithm_name")
    num_ions = config.get("num_ions")
    use_dag = config.get("use_dag", True) and not config.get("enforce_slice_plan", False)
    use_paths = config.get("use_paths", False)
    max_timesteps = config.get("max_timesteps")
    plot_flag = config.get("plot", False)
    save_flag = config.get("save", False)
    failing_junctions = config.get("failing_junctions", 0)
    debug_gate_tracking = config.get("debug_gate_tracking", False)
    failing_junctions = int(config.get("failing_junctions", 0))
    timeline_output = config.get("timeline_output")
    gate_time_one_qubit = config.get("gate_time_one_qubit")
    gate_time_two_qubit = config.get("gate_time_two_qubit")
    qasm_file_path_cfg = config.get("qasm_file_path")
    gate_partition_cfg = config.get("gate_partition")
    gate_partition_algorithm_cfg = config.get("gate_partition_algorithm")
    enforce_slice_plan = config.get("enforce_slice_plan", True)
    optimize_params = config.get("optimize_params", False)
    optimization_budget = float(config.get("optimization_budget", 3.0))
    plot_move_hist = bool(config.get("plot_move_hist", False))
    max_shuttle_seconds = config.get("max_shuttle_seconds", 1800)
    cache_shortest_paths = bool(config.get("cache_shortest_paths", True))


    if arch is None or not isinstance(arch, list) or len(arch) != 4:
        msg = "Config parameter 'arch' must be a list of 4 integers [m, n, v, h]."
        raise ValueError(msg)
    if algorithm_name is None and qasm_file_path_cfg is None:
        msg = "Either 'algorithm_name' or 'qasm_file_path' must be set."
        raise ValueError(msg)
    if num_ions is None and qasm_file_path_cfg is None:
        msg = "Either 'num_ions' or 'qasm_file_path' must be set."
        raise ValueError(msg)

    m, n, v, h = map(int, arch)
    start_time = datetime.now()
    cycle_or_paths_str = "Paths" if use_paths else "Cycles"

    qasm_base_dir_string = config.get("qasm_base_dir")
    if qasm_base_dir_string is None:
        qasm_base_dir = pathlib.Path(__file__).absolute().parent.parent.parent.parent / "inputs" / "qasm_files"
    else:
        qasm_base_dir = pathlib.Path(qasm_base_dir_string)

    gate_density = config.get("gate_density")
    if gate_density:
        gate_densities_string = f"_{gate_density[0]}_{gate_density[1]}"
    else:
        gate_densities_string = ""

    if qasm_file_path_cfg is not None:
        qasm_file_path = pathlib.Path(qasm_file_path_cfg)
        if algorithm_name is None:
            algorithm_name = qasm_file_path.stem
        if num_ions is None:
            num_ions = _infer_num_ions_from_qasm(qasm_file_path)
    else:
        qasm_file_path = qasm_base_dir / f"{algorithm_name}" / f"{algorithm_name}{gate_densities_string}_{num_ions}.qasm"

    if not qasm_file_path.is_file():
        print(f"Error: QASM file not found at {qasm_file_path}")
        sys.exit(1)

    max_timesteps = _resolve_max_timesteps(config, num_ions)

    pz_definitions = generate_pzs(
        num_pzs=num_pzs_config,
        m=m,
        n=n,
        v=v,
        h=h,
        pz_edges=config.get("pz_edges"),
    )
    available_pz_names = list(pz_definitions.keys())
    pz_names_to_use = [f"pz{pz}" for pz in config.get("pz_numbers_to_use", range(1, num_pzs_config + 1))]
    if not all(name in available_pz_names for name in pz_names_to_use):
        msg = f"Some specified PZ names are invalid: {pz_names_to_use}"
        raise ValueError(msg)
    pzs_to_use = [pz_definitions[name] for name in pz_names_to_use]
    if not pzs_to_use:
        msg = "No processing zones selected."
        raise ValueError(msg)

    graph_creator = GraphCreator(
        m,
        n,
        v,
        h,
        failing_junctions,
        pzs_to_use,
        edges_to_delete=config.get("edges_to_delete"),
        nodes_to_suppress=config.get("nodes_to_suppress"),
    )
    graph = graph_creator.get_graph()

    def _normalize_edge(edge: tuple[tuple[int, int], tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
        return tuple(sorted(edge, key=sum))

    normalized_edges = {_normalize_edge(edge) for edge in graph.edges()}
    for pz in pzs_to_use:
        normalized = _normalize_edge(pz.edge_idc)
        if normalized not in normalized_edges:
            msg = f"PZ edge {pz.edge_idc} ({pz.name}) is not present in the inside graph."
            raise ValueError(msg)
        pz.edge_idc = normalized
        graph.edges[normalized]["edge_type"] = "processing"
        graph.edges[normalized]["color"] = "g"

    graph.seed = seed
    graph.max_ions_per_pz = max_ions_per_pz
    graph.pzs = pzs_to_use
    graph.plot = plot_flag
    graph.save = save_flag
    graph.arch = str(arch)
    graph.m = m
    graph.n = n
    graph.v = v
    graph.h = h
    #graph.debug_gate_tracking = debug_gate_tracking
    #graph.enable_memory_zone_manager = config.get("enable_memory_zone_manager", False)
    graph.locked_gates = {}
    #graph.gate_pz_assignment = {}
    #graph.current_gate_by_pz = {}
    #graph.dag_gate_id_lookup = {}

    if plot_flag:
        try:
            qc = QuantumCircuit.from_qasm_file(str(qasm_file_path))
            qc.draw(output="mpl", filename=f"outputs/circuits/{algorithm_name}_{num_ions}_circuit.png")
        except Exception as exc:
            print(f"Warning: Could not visualize circuit: {exc}")

    create_starting_config(graph, int(num_ions), seed=seed)
    
    
    parsed = compile_qasm(qasm_file_path)
    graph.sequence = parsed.sequence
    graph.gate_info = parsed.gate_info
    timesteps_lower_bound = _calculate_timestep_lower_bound(graph.sequence, graph.gate_info, len(graph.pzs))
    '''
    graph.state = get_ion_chains(graph)
    initial_circuit = create_initial_sequence(qasm_file_path)
    graph.sequence = initial_circuit.sequence.copy()
    graph.gate_info = initial_circuit.gate_info
    shuttle_sequence = [graph.gate_info[gate_id].qubits for gate_id in graph.sequence]
    timesteps_lower_bound = _calculate_timestep_lower_bound(shuttle_sequence, len(graph.pzs))
    '''

    partitions: dict[str, list[int]] = {}
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

    graph.map_to_pz = {qubit: pz_name for pz_name, qubits in partitions.items() for qubit in qubits}
    all_qubits = {q for gate_id in graph.sequence for q in graph.gate_info[gate_id].qubits}
    #all_qubits = {ion for gate_id in graph.sequence for ion in graph.gate_info[gate_id].qubits}
    if missing := sorted(all_qubits - set(graph.map_to_pz)):
        fallback_pz = graph.pzs[0].name
        for qubit in missing:
            graph.map_to_pz[qubit] = fallback_pz

    gate_partition_for_run: dict[str, list[int]] | None = None
    gate_assignment: dict[int, str] = {}
    partition_result: object | None = None
    partition_param_trials: list[dict[str, object]] | None = None
    gate_count_1q = sum(1 for info in graph.gate_info.values() if len(info.qubits) == 1)
    gate_count_2q = sum(1 for info in graph.gate_info.values() if len(info.qubits) == 2)
    
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
            from outside.fgp_roee import fgp_roee

            if "num_pzs" not in algo_params:
                algo_params["num_pzs"] = config.get("num_pzs", 1)
            if "capacity" not in algo_params:
                algo_params["capacity"] = config.get("max_ions_per_pz", 1)
            result = fgp_roee(graph, **algo_params)
            gate_partition_for_run = result.gate_partition_by_pz
            gate_assignment = result.gate_assignment
            partition_result = result
        elif algo_name_lower in {"fgp_tabu", "fgp_tabu_global", "fgp_tabu_global_2", "fgp_kl"}:
            if algo_name_lower == "fgp_tabu":
                from outside.fgp_tabu import fgp_tabu as gate_partitioner
            elif algo_name_lower == "fgp_tabu_global":
                from outside.fgp_tabu_global import fgp_tabu_global as gate_partitioner
            elif algo_name_lower == "fgp_tabu_global_2":
                from outside.fgp_tabu_global_2 import fgp_tabu_global as gate_partitioner
            else:
                from outside.fgp_kl import fgp_kl as gate_partitioner
            if "num_pzs" not in algo_params:
                algo_params["num_pzs"] = config.get("num_pzs", 1)
            if "capacity" not in algo_params:
                algo_params["capacity"] = config.get("max_ions_per_pz", 1)
            if algo_name_lower in {"fgp_tabu_global", "fgp_tabu_global_2"}:
                if "capacity_weight" not in algo_params:
                    algo_params["capacity_weight"] = 1.0
                if "distance_weight" not in algo_params:
                    algo_params["distance_weight"] = 1.0
                if "balance_weight" not in algo_params:
                    algo_params["balance_weight"] = 1.0
                if "relaxed_layering" not in algo_params and "relaxed_layering" in config:
                    algo_params["relaxed_layering"] = config["relaxed_layering"]
                if "max_layer_depth" not in algo_params and "max_layer_depth" in config:
                    algo_params["max_layer_depth"] = config["max_layer_depth"]
            else:
                if "lookahead_weight_factor" not in algo_params:
                    algo_params["lookahead_weight_factor"] = 1.0
            if optimize_params and algo_name_lower == "fgp_tabu":
                print(
                    f"Optimizing fgp_tabu parameters for up to {optimization_budget:.1f}s using move_distance_total proxy"
                )
                start_opt = time.time()
                best_result = None
                best_params = None
                trials: list[dict[str, object]] = []
                iteration = 0
                collected_results = []
                while True:
                    iteration += 1
                    sampled = {
                        "balance_penalty": random.uniform(0.1, 5.0),
                        "sigma": random.uniform(0.1, 5.0),
                        "lookahead_weight_factor": random.uniform(0.1, 5.0),
                        "distance_weight_factor": random.uniform(0.1, 5.0),
                    }
                    params_for_run = algo_params.copy()
                    params_for_run.update(sampled)
                    candidate = None
                    try:
                        candidate = gate_partitioner(graph, **params_for_run)
                    except Exception as exc:
                        print(f"Warning: fgp_tabu candidate failed: {exc}")
                    move_dist = getattr(candidate, "move_distance_total", None) if candidate else None
                    trials.append({"params": sampled, "move_distance_total": move_dist})
                    if candidate and move_dist is not None:
                        collected_results.append((candidate, params_for_run, move_dist))
                    current_best = getattr(best_result, "move_distance_total", None) if best_result else None
                    if candidate and move_dist is not None and (current_best is None or move_dist < current_best):
                        best_result = candidate
                        best_params = params_for_run
                    if time.time() - start_opt >= optimization_budget and best_result is not None:
                        break
                    if time.time() - start_opt >= optimization_budget and iteration > 3:
                        break
                chosen_result = best_result if best_result is not None else candidate
                if collected_results:
                    distances = [md for _, _, md in collected_results if md is not None]
                    mean_md = float(np.mean(distances))
                    std_md = float(np.std(distances))
                    mid_pack = [
                        (res, params, md)
                        for res, params, md in collected_results
                        if abs(md - mean_md) <= std_md
                    ]
                    if mid_pack:
                        res_mid, params_mid, _ = min(mid_pack, key=lambda x: x[2])
                        chosen_result = res_mid
                        best_params = params_mid
                result = chosen_result
                partition_param_trials = trials
                if best_params:
                    print(
                        f"Ran {iteration} iterations. Selected params: {best_params} "
                        f"with move_distance_total={getattr(result, 'move_distance_total', None)}"
                    )
            else:
                result = gate_partitioner(graph, **algo_params)
            gate_partition_for_run = result.gate_partition_by_pz
            gate_assignment = result.gate_assignment
            partition_result = result
        elif algo_name_lower in {"tdag", "fgp_tdag"}:
            from outside.tdag import compute_gate_partition_tdag

            capacity = algo_params.get("k", algo_params.get("capacity", 4))
            balance_penalty = algo_params.get("balance_penalty", 0.5)
            tdag_result = compute_gate_partition_tdag(
                graph,
                None,
                capacity=capacity,
                qasm_file_path=qasm_file_path,
                balance_penalty=balance_penalty,
            )
            gate_partition_for_run = tdag_result.get("gate_partition_for_run", {})
            gate_assignment = tdag_result.get("gate_assignment", {})
        elif algo_name_lower == "rehome":
            pass
        else:
            msg = f"Unknown gate partition algorithm '{algo_name}'."
            raise ValueError(msg)

    graph.gate_pz_assignment = gate_assignment

    timeline_frames: list[dict[str, object]] = []
    if timeline_output:
        def _patched_plot_state(g, *args, **kwargs):
            labels = kwargs.get("labels", args[0] if args else ("", None))
            timestep_label = labels[0] if isinstance(labels, tuple) and labels else ""
            match = re.search(r"timestep\s+(\d+)", str(timestep_label))
            if not match:
                return None
            timestep = int(match.group(1))

            ion_edges = _collect_ion_edges(g)
            ions_payload = [
                {"id": f"$q_{ion}$", "edge": _edge_to_strings(edge)}
                for ion, edge in sorted(ion_edges.items())
            ]

            buffered = list(getattr(g, "executed_gates_next", []))
            if buffered and timeline_frames:
                converted = []
                for gate in buffered:
                    edge = gate.get("edge", [])
                    edge_str = _edge_to_strings(tuple(edge)) if len(edge) == 2 else []
                    qubits = gate.get("qubits", [])
                    qubits = [f"$q_{q}$" for q in qubits]
                    converted.append(
                        {
                            "id": gate.get("id"),
                            "type": gate.get("type"),
                            "qubits": qubits,
                            "edge": edge_str,
                            "duration": gate.get("duration", 1),
                            "pz": gate.get("pz", ""),
                        }
                    )
                timeline_frames[-1].setdefault("gates", [])
                timeline_frames[-1]["gates"].extend(converted)
                g.executed_gates_next = []

            frame: dict[str, object] = {"t": timestep, "ions": ions_payload}
            timeline_frames.append(frame)
            return None

        plotting_mod.plot_state = _patched_plot_state
        shuttle_mod.plot_state = _patched_plot_state
        graph.plot = True

    def _write_timeline_snapshot(interrupted: bool) -> None:
        if not timeline_output:
            return
        algo = str(gate_partition_algorithm_cfg["name"]) if gate_partition_algorithm_cfg is not None else "none"
        out_name = timeline_output + "_" + algorithm_name + "_" + algo + ".json"
        processing_edges: list[tuple[tuple[int, int], tuple[int, int]]] = [
            edge for edge, data in graph.edges.items() if data.get("edge_type") == "processing"
        ]
        pzs_payload: dict[str, list[str]] = {}
        for pz in graph.pzs:
            normalized = tuple(sorted(pz.edge_idc, key=sum))
            if graph.has_edge(*normalized):
                pzs_payload[pz.name] = _edge_to_strings(normalized)
                continue
            match = next(
                (edge for edge, data in graph.edges.items() if data.get("pz_name") == pz.name),
                None,
            )
            if match is not None:
                pzs_payload[pz.name] = _edge_to_strings(match)
        inner_pz_edges = [_edge_to_strings(edge) for edge in processing_edges]
        removed_edges_cfg = config.get("edges_to_delete") or []
        suppressed_nodes_cfg = config.get("nodes_to_suppress") or []

        def _coerce_node(node: Any) -> tuple[int, int]:
            x, y = node
            return (int(x), int(y))

        def _coerce_edge(edge: Any) -> tuple[tuple[int, int], tuple[int, int]]:
            node_a, node_b = edge
            return (_coerce_node(node_a), _coerce_node(node_b))

        removed_edges_payload = [_edge_to_strings(_coerce_edge(edge)) for edge in removed_edges_cfg]
        suppressed_nodes_payload = [_format_site(_coerce_node(node)) for node in suppressed_nodes_cfg]
        architecture = {
            "grid": {"rows": m, "cols": n},
            "sites": {"vertical": v, "horizontal": h},
            "pzs": pzs_payload,
            "innerPZEdges": inner_pz_edges,
            "removedEdges": removed_edges_payload,
            "suppressedNodes": suppressed_nodes_payload,
        }
        payload = {
            "architecture": architecture,
            "grid": architecture["grid"],
            "sites": architecture["sites"],
            "pzs": architecture["pzs"],
            "innerPZEdges": architecture["innerPZEdges"],
            "removedEdges": architecture["removedEdges"],
            "suppressedNodes": architecture["suppressedNodes"],
            "timeline": timeline_frames,
            "interrupted": interrupted,
        }
        out_path = pathlib.Path(out_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, separators=(",", ":")))
        state = "partial" if interrupted else "complete"
        print(f"Wrote {state} timeline JSON to {out_path} ({len(timeline_frames)} frames).")

    print("\nStarted inside shuttling simulation...")
    #final_timesteps = run_shuttle_main(graph, graph.sequence.copy(), cycle_or_paths_str)
    interrupted = False
    timed_out = False
    try:
        final_timesteps, timed_out = run_shuttle_main(
            graph,
            cycle_or_paths_str,
            max_timesteps=max_timesteps,
            max_seconds=max_shuttle_seconds,
            cache_shortest_paths=cache_shortest_paths,
        )
    except KeyboardInterrupt:
        interrupted = True
        final_timesteps = -1
        raise
    finally:
        _write_timeline_snapshot(interrupted)
    cpu_time = datetime.now() - start_time

    cost_before = getattr(partition_result, "cost_before", None)
    cost_after = getattr(partition_result, "cost_after", None)
    time_slices_info = getattr(partition_result, "time_slices", [])
    qubit_assignments = getattr(partition_result, "assignments", [])
    move_distance_total = getattr(partition_result, "move_distance_total", None)
    _ = enforce_slice_plan, use_dag, gate_partition_for_run, plot_move_hist

    return (
        final_timesteps,
        timed_out,
        cpu_time,
        timesteps_lower_bound,
        cost_before,
        cost_after,
        time_slices_info,
        qubit_assignments,
        move_distance_total,
        partition_param_trials,
        gate_count_1q,
        gate_count_2q,
    )


def execute_run(
    config: dict[str, Any],
) -> tuple[
    int,
    bool,
    timedelta,
    int,
    float | None,
    float | None,
    list[list[int]],
    list[list[int]],
    float | None,
    list[dict[str, object]] | None,
    int,
    int,
]:
    config_for_run = config.copy()
    if config_for_run.get("max_timesteps") is None and config_for_run.get("max_timesteps_factor") is not None:
        num_ions = config_for_run.get("num_ions")
        if num_ions is None and config_for_run.get("qasm_file_path") is not None:
            num_ions = _infer_num_ions_from_qasm(pathlib.Path(config_for_run["qasm_file_path"]))
            config_for_run["num_ions"] = num_ions
        config_for_run["max_timesteps"] = _resolve_max_timesteps(config_for_run, num_ions)
    if should_use_legacy_cli(config_for_run):
        print("Using legacy CLI entrypoint (mqt-ionshuttler-heuristic) for this configuration.")
        final_ts, cpu_time = run_legacy_cli_with_config(config_for_run)
        return final_ts, False, cpu_time, 0, None, None, [], [], None, None, 0, 0
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
    parser.add_argument("--use_legacy_implementation", help="If possible, run legacy implementation y/n", default="n")
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

    config["use_legacy_implementation"] = args.use_legacy_implementation.lower() == "y"

    if args.run_meta_study.lower() != "y":
        #run single compilation
        main(config)
        exit()

    # Use a single results file (not datetime specific)
    results_file = f"outputs/simulation_results.h5"
    pathlib.Path("outputs").mkdir(exist_ok=True)
    
    # Helper function to check if parameter set exists
    def parameter_set_exists(f, run_params: dict) -> bool:
        if 'results' not in f:
            return False
        results_group = f['results']
        def _normalize_attr(value):
            if isinstance(value, np.ndarray):
                return value
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except Exception:
                    pass
            if value == 'None':
                return None
            return value

        for run_name in results_group.keys():
            run_group = results_group[run_name]
            # Check all stored attributes match the run parameters
            match = all(
                np.array_equal(run_group.attrs.get(k), v) if isinstance(run_group.attrs.get(k), np.ndarray)
                else _normalize_attr(run_group.attrs.get(k)) == _normalize_attr(v)
                for k, v in run_params.items()
            )
            if match:
                return True
        return False

    def _write_json_dataset(group: h5py.Group, name: str, payload: object) -> None:
        """Persist small structured payloads (lists/dicts) as UTF-8 JSON datasets."""
        try:
            if name in group:
                del group[name]
            dtype = h5py.string_dtype(encoding="utf-8")
            group.create_dataset(name, data=np.array(json.dumps(payload), dtype=dtype))
        except Exception as exc:  # pragma: no cover - best-effort diagnostic
            print(f"Warning: could not store dataset '{name}': {exc}")

    qasm_folder = config.get("qasm_folder")
    max_qasm_lines = config.get("max_qasm_lines")
    qasm_files: list[pathlib.Path] | None = None
    if qasm_folder:
        qasm_dir = pathlib.Path(qasm_folder)
        if not qasm_dir.is_dir():
            print(f"Error: QASM folder does not exist or is not a directory: {qasm_dir}")
            sys.exit(1)
        qasm_files = sorted(qasm_dir.glob("*.qasm"))
        qasm_files.extend(sorted(qasm_dir.glob("*.qasm3")))
        if not qasm_files:
            print(f"No QASM files found in {qasm_dir}")
            sys.exit(1)

    #################################################################################################################
    # Meta study configuration
    plot_move_hist = False
    
    #unique_id = "generated_0.71_0.7_num_ions_4pzs"
    #unique_id = "balance_distance_sweep_20ions_4pzs"
    #unique_id = "num_pzs_mean_std_allswept_4pzs"
    

    # Declare partitioning algorithm parameters
    fgp_tabu = {
        'name': 'fgp_tabu',
        'params': {
            #'balance_penalty': np.linspace(0.1, 5, 40),  #[0.6],
            #'sigma': np.linspace(0.1, 5, 20),  #[5.0],
            #'lookahead_weight_factor': np.linspace(0.1, 5, 20),  #[0.6],
            #'distance_weight_factor': np.linspace(0.1, 5, 20)  #[1.5],
        },
        '_sampling': {
            'method': 'lhs',
            'num_samples': 10,
        },
    }
    fgp_tabu_global = {
        'name': 'fgp_tabu_global',
        'params': {
            'balance_penalty': [0.5], # np.linspace(0.1, 5, 40),  #[0.5],
            'distance_weight_factor': [1.0], #np.linspace(0.1, 5, 20),  #[1.0],
            #'max_iterations_factor': [25], 
            'max_iterations': [0, 100, 1000, 10000],
            #'tabu_list_length': [200],
            'seed': range(5),
            'candidate_list_length': [None],
            'relaxed_layering': [False],
        },
    }
    fgp_tabu_global_2_capacity = {
        'name': 'fgp_tabu_global_2',
        'params': {
            'balance_penalty': [0], #[0, 0.01, 0.05, 0.1, 0.5], #[0.6],
            'distance_weight_factor': [1.0],  #[1.5],
            'capacity_weight': [0.1, 0.15, 0.2, 0.25],#[0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2], #[0, 0.01, 0.05, 0.1, 0.5],  #[1.0],
            'max_iterations_factor': [25,50,100],#[0, 25, 50, 100, 200],# 100], 
            #'tabu_list_length': [200],
            'seed': range(3),
            'candidate_list_length': [None],
            'relaxed_layering': [True]#, False],
        },
    }
    fgp_tabu_global_2_dense = {
        'name': 'fgp_tabu_global_2',
        'params': {
            'balance_penalty': [0], #[0, 0.01, 0.05, 0.1, 0.5], #[0.6],
            'distance_weight_factor': [1.0],  #[1.5],
            'capacity_weight': [0.5],#[0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2], #[0, 0.01, 0.05, 0.1, 0.5],  #[1.0],
            'max_iterations_factor': [50],#[0, 25, 50, 100, 200],# 100], 
            #'max_iterations': [10000],#[0, 100, 1000, 10000],
            #'tabu_list_length': [200],
            'seed': range(1),
            'candidate_list_length': [None],
            'relaxed_layering': [True]#, False],
        },
    }
    fgp_tabu_global_2_racetrack = {
        'name': 'fgp_tabu_global_2',
        'params': {
            'balance_penalty': [0], #[0, 0.01, 0.05, 0.1, 0.5], #[0.6],
            'distance_weight_factor': [1.0],  #[1.5],
            'capacity_weight': [0.1],#[0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2], #[0, 0.01, 0.05, 0.1, 0.5],  #[1.0],
            'max_iterations_factor': [50],#[0, 25, 50, 100, 200],# 100], 
            #'max_iterations': [0, 100, 1000, 10000],
            #'tabu_list_length': [200],
            'seed': range(1),
            'candidate_list_length': [None],
            'relaxed_layering': [True]#, False],
        },
    }
    fgp_kl = {
        'name': 'fgp_kl',
        'params': {
            'lookahead_weight_factor': [2.902851755464501],  #[0.85],
            'balance_penalty': [4.516752],  #[4.5],
            'sigma': [1.6113372786564444],  #[4.5]
            'distance_weight_factor': [2.1438317591],  #[3.75],
        },
        'sampling': {'method': 'lhs', 'num_samples': 30},
    }
    fgp_roee = {'name': 'fgp_roee', 'params': {'sigma': [0.01,5.0]}, 'sampling': {'method': 'lhs', 'num_samples': 20},}
    rehome = {'name': 'rehome', 'params': {}}


    meta_study_config = {
        "algorithm_name": ["qft_nativegates_quantinuum_qiskit_opt2", "qaoa_nativegates_quantinuum_opt2", "qpeexact_nativegates_quantinuum_qiskit_opt2", "random_nativegates_quantinuum_qiskit_opt2"],
        #"algorithm_name": ["qft_nativegates_quantinuum_qiskit_opt2"],
        #"algorithm_name" : ["random_nativegates_quantinuum_qiskit_opt2"],
        # Core architecture parameters
        'num_ions': [30],
        #'num_pzs': [8],
        #'ions_per_pz': [2],
        'grid_size': [4],
        'mz_trap_size': [4],
        #'pz_numbers_to_use': [[1,9,6,3,11,8]], 
        'use_dag': [False],
        'enforce_slice_plan': [False],
        'enable_memory_zone_manager': [False],
        'save' : [False],
        'plot' : [False],
        'optimize_params': [False],

        #'gate_density': [(0.0,1.0), (0.1,0.9), (0.2,0.8), (0.3,0.7), (0.4,0.6), (0.5,0.5), (0.6,0.4), (0.7,0.3), (0.8,0.2), (0.9,0.1), (1.0,0.0)],
        #'gate_density': [(0.1,0.1), (0.25,0.25), (0.5,0.5), (0.75, 0.75), (1.0, 1.0)], 
        
        # Partitioning algorithm configurations
        'partitioning_algorithms': [
            #{'name': 'none'},  # No partitioning
            #fgp_tabu,
            #fgp_tabu_global,
            #fgp_tabu_global_2_dense,
            #fgp_tabu_global_2_racetrack,
            fgp_tabu_global_2_capacity,
            #fgp_tabu_global_long,
            #fgp_tabu_global_test,
            #fgp_tabu_global_mini,
            #fgp_tabu_global_mini_quota,
            #fgp_tabu_global_slack
            #fgp_kl,
            #fgp_roee,
            #rehome
        ]
    }
    # Default to config values when not explicitly swept in meta_study_config.
    if "num_pzs" not in meta_study_config:
        meta_study_config["num_pzs"] = [config.get("num_pzs", 1)]
    if "ions_per_pz" not in meta_study_config:
        meta_study_config["ions_per_pz"] = [config.get("max_ions_per_pz", 2)]

    dag_string = "WITHDAG" if True in meta_study_config.get("use_dag", []) else "NODAG"
    #unique_id = f"INSIDE_fgp_tabu_global_circ-random_final_dense_4"
    if "num_pzs" in meta_study_config and "ions_per_pz" in meta_study_config:
        arch_string = Path(args.config_file).stem +"_"+f"{str(meta_study_config['num_pzs'])}pzs_{str(meta_study_config['ions_per_pz'])}perpz"
    else:
        arch_string = Path(args.config_file).stem +"_"+f"{str(config['num_pzs'])}pzs_{str(config['max_ions_per_pz'])}perpz"
    
    ##############################################################
    unique_id = f"INSIDE_HyperStudy" + "_" + arch_string
    unique_id += f"_{dag_string}"
    clear_prev = False


    if unique_id != "":
        #stamp = datetime.now().strftime("%Y%m%d_%H")
        results_file = f"outputs/results/simulation_results_{unique_id}.h5"




    # Clear previous results if requested
    if clear_prev and pathlib.Path(results_file).exists():
        print(f"Removing existing results file: {results_file}")
        pathlib.Path(results_file).unlink()

    # Generate all valid combinations
    valid_combinations = []
    
    # Extract base parameter ranges
    base_params = {k: v for k, v in meta_study_config.items() if k != 'partitioning_algorithms'}
    base_param_names = [k for k in base_params.keys()]
    base_param_values = [base_params[k] for k in base_param_names]

    # Fallback params from JSON config (e.g., gate_partition_algorithm)
    fallback_algo_params: dict[str, dict[str, Any]] = {}
    gpa = config.get("gate_partition_algorithm")
    if isinstance(gpa, dict) and gpa.get("name"):
        fallback_algo_params[gpa["name"]] = gpa.get("params", {})

    
    # Iterate through all base parameter combinations
    for base_combo in product(*base_param_values):
        base_dict = dict(zip(base_param_names, base_combo))

        # If num_pzs is omitted, infer it from the provided PZ numbers for this combo
        if "num_pzs" not in base_dict and "pz_numbers_to_use" in base_dict:
            base_dict["num_pzs"] = len(base_dict["pz_numbers_to_use"])
        
        # For each partitioning algorithm configuration
        for algo_config in meta_study_config['partitioning_algorithms']:
            algo_name = algo_config['name']
            
  
            if algo_name == 'none':
                # enforce_slice_plan makes no sense for no slice plan
                base_dict['enforce_slice_plan'] = False
                # No algorithm parameters to expand
                params_dict = base_dict.copy()
                params_dict['partitioning_algorithm'] = 'none'
                params_dict['enable_memory_zone_manager'] = False
                valid_combinations.append(params_dict)
            else:
                # Generate all combinations of algorithm parameters
                algo_params = dict(algo_config.get('params', {}))
                # Supply missing params from JSON config for matching algorithm
                for param_name, param_value in fallback_algo_params.get(algo_name, {}).items():
                    if param_name not in algo_params:
                        algo_params[param_name] = param_value if isinstance(param_value, list) else [param_value]
                if algo_params:
                    algo_param_names = list(algo_params.keys())
                    algo_param_values = [algo_params[k] for k in algo_param_names]
                    sampling_cfg = algo_config.get('sampling')
                    params_to_sample = algo_config.get('params')
                    if sampling_cfg and sampling_cfg.get('method') == 'lhs':
                        num_samples = int(sampling_cfg.get('num_samples', 10))
                        lower_bounds = np.array([min(vals) for vals in params_to_sample.values()], dtype=float)
                        upper_bounds = np.array([max(vals) for vals in params_to_sample.values()], dtype=float)
                        #print("LHS Sampling:", params_to_sample.keys(), "Samples:", num_samples)
                        sampler = qmc.LatinHypercube(d=len(params_to_sample))
                        lhs_sample = sampler.random(num_samples)
                        scaled = qmc.scale(lhs_sample, lower_bounds, upper_bounds)
                        for sample_vals in scaled:
                            params_dict = base_dict.copy()
                            params_dict['partitioning_algorithm'] = algo_name
                            for param_name, param_value in zip(algo_param_names, sample_vals):
                                params_dict[f'algo_{param_name}'] = float(param_value)
                            valid_combinations.append(params_dict)
                    else:
                        for algo_combo in product(*algo_param_values):
                            params_dict = base_dict.copy()
                            params_dict['partitioning_algorithm'] = algo_name
                            
                            # Add algorithm-specific parameters with prefix
                            for param_name, param_value in zip(algo_param_names, algo_combo):
                                params_dict[f'algo_{param_name}'] = param_value
                            
                            valid_combinations.append(params_dict)
                else:
                    # Algorithm has no parameters
                    params_dict = base_dict.copy()
                    params_dict['partitioning_algorithm'] = algo_name
                    valid_combinations.append(params_dict)
    
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
        if qasm_files:
            total_combinations *= len(qasm_files)
        skipped = 0
        
        print(f"Total combinations: {total_combinations}")
        print(f"Existing runs in file: {existing_runs}")
        
        best_timesteps = None
        best_params = None
        move_dist_records: list[float] = []

        for run_params in valid_combinations:
            run_qasm_files = qasm_files if qasm_files else [None]
            for qasm_file in run_qasm_files:
                run_params_for_file = run_params.copy()
                inferred_num_ions = None
                if qasm_file is not None:
                    circuit_name = qasm_file.stem
                    try:
                        inferred_num_ions = _infer_num_ions_from_qasm(qasm_file)
                    except Exception as exc:
                        print(f"Skipping {qasm_file}: could not infer qubits ({exc})")
                        skipped += 1
                        continue
                    if max_qasm_lines is not None:
                        try:
                            with qasm_file.open("r", encoding="utf-8", errors="ignore") as handle:
                                line_count = sum(1 for _ in handle)
                        except Exception as exc:
                            print(f"Skipping {qasm_file}: could not count lines ({exc})")
                            skipped += 1
                            continue
                        if line_count > max_qasm_lines:
                            print(
                                f"Skipping {qasm_file}: line_count={line_count} exceeds max_qasm_lines={max_qasm_lines}"
                            )
                            skipped += 1
                            continue
                    target_qubits = run_params.get("num_ions", config.get("num_ions"))
                    if isinstance(target_qubits, (list, tuple, set)):
                        allowed_qubits = {int(val) for val in target_qubits}
                    elif target_qubits is None:
                        allowed_qubits = set()
                    else:
                        allowed_qubits = {int(target_qubits)}
                    if allowed_qubits and inferred_num_ions not in allowed_qubits:
                        skipped += 1
                        continue
                    run_params_for_file["algorithm_name"] = circuit_name
                    run_params_for_file["circuit_name"] = circuit_name
                    run_params_for_file["qasm_file"] = str(qasm_file)
                    run_params_for_file["num_ions"] = int(inferred_num_ions)

                # Check if this parameter set already exists
                if parameter_set_exists(f, run_params_for_file):
                    print(f"\nSkipping existing parameter set: {run_params_for_file}")
                    skipped += 1
                    continue

                # Update config for this run
                # Define mapping from run_params keys to config keys and their transformations
                param_mapping = {
                    'num_ions': lambda v: v,
                    'num_pzs': lambda v: v,
                    'ions_per_pz': ('max_ions_per_pz', lambda v: v),
                    'use_dag': lambda v: v,
                    'enforce_slice_plan': lambda v: v,
                    'save': lambda v: v,
                    'plot': lambda v: v,
                    'grid_size': ('arch', lambda v: [v, v, run_params['mz_trap_size'], run_params['mz_trap_size']]),
                    'gate_density': lambda v: v,
                    'enable_memory_zone_manager': lambda v: v,
                    'pz_numbers_to_use': lambda v: v,
                    'optimize_params': lambda v: v,
                    'algorithm_name': lambda v: v,
                    'gate_time_one_qubit': lambda v: v,
                    'gate_time_two_qubit': lambda v: v
                }
                
                # Apply direct parameter mappings
                for param_key, param_value in run_params_for_file.items():
                    if param_key in param_mapping:
                        mapping = param_mapping[param_key]
                        if isinstance(mapping, tuple):
                            config_key, transform = mapping
                            config[config_key] = transform(param_value)
                        else:
                            config[param_key] = mapping(param_value)

                if qasm_file is not None:
                    config["qasm_file_path"] = str(qasm_file)
                    if inferred_num_ions is not None:
                        config["num_ions"] = int(inferred_num_ions)
                else:
                    config.pop("qasm_file_path", None)
                
                # Handle partitioning algorithm
                if run_params_for_file['partitioning_algorithm'] == 'none':
                    config.pop("gate_partition_algorithm", None)
                else:
                    # Extract algorithm parameters from run_params
                    algo_params = {
                        key.replace('algo_', ''): value 
                        for key, value in run_params_for_file.items() 
                        if key.startswith('algo_')
                    }
                    algo_params['num_pzs'] = run_params_for_file['num_pzs']
                    algo_params['capacity'] = run_params_for_file['ions_per_pz']
                    
                    config["gate_partition_algorithm"] = {
                        "name": run_params_for_file['partitioning_algorithm'],
                        "params": algo_params
                    }
                
                print(f"\n=== Run {result_index + skipped - existing_runs + 1} / {total_combinations} new ===")
                print(f"Config: {run_params_for_file}")
                
                run_name = f'run_{result_index:04d}'
                run_group = results_group.create_group(run_name)

                # Store all run parameters as attributes
                for key, value in run_params_for_file.items():
                    if value == None:
                        value = 'None'
                    run_group.attrs[key] = value
                if "algorithm_name" not in run_params_for_file and config.get("algorithm_name") is not None:
                    run_group.attrs["algorithm_name"] = config["algorithm_name"]
                
                try:
                    (
                        final_timesteps,
                        timed_out,
                        cpu_time,
                        timesteps_lower_bound,
                        cost_before,
                        cost_after,
                        time_slices_info,
                        qubit_assignments,
                        move_distance_total,
                        partition_param_trials,
                        gate_count_1q,
                        gate_count_2q,
                    ) = execute_run(config)

                    print("move_distance_total:", move_distance_total)

                    max_timesteps = _resolve_max_timesteps(config, config.get("num_ions"))
                    if timed_out:
                        run_group.attrs['success'] = False
                        run_group.attrs['error_message'] = (
                            f"Compilation time limit of {config["max_shuttle_seconds"]}s reached at timestep {final_timesteps}/{max_timesteps}"
                        )
                    elif final_timesteps >= max_timesteps - 1:
                        run_group.attrs['success'] = False
                        run_group.attrs['error_message'] = (
                            f"Simulation reached max timesteps ({final_timesteps})"
                        )
                    else:
                        run_group.attrs['success'] = True
                        run_group.attrs['final_timesteps'] = final_timesteps
                    run_group.attrs['cpu_time_seconds'] = cpu_time.total_seconds()
                    run_group.attrs['timesteps_lower_bound'] = timesteps_lower_bound
                    run_group.attrs['cost_before'] = cost_before if cost_before is not None else np.nan
                    run_group.attrs['cost_after'] = cost_after if cost_after is not None else np.nan
                    run_group.attrs['gate_count_1q'] = int(gate_count_1q)
                    run_group.attrs['gate_count_2q'] = int(gate_count_2q)
                    if time_slices_info:
                        _write_json_dataset(run_group, "time_slices", time_slices_info)
                    if qubit_assignments:
                        _write_json_dataset(run_group, "qubit_assignments", qubit_assignments)
                    run_group.attrs["move_distance_total"] = (
                        float(move_distance_total) if move_distance_total is not None else np.nan
                    )
                    if partition_param_trials:
                        _write_json_dataset(run_group, "partition_param_trials", partition_param_trials)
                    if move_distance_total is not None:
                        move_dist_records.append(float(move_distance_total))

                    if run_group.attrs['success']:
                        print(f" - Successful!, {cpu_time.total_seconds():.2f}s CPU time")
                        if best_timesteps is None or final_timesteps < best_timesteps:
                            best_timesteps = final_timesteps
                            best_params = run_params_for_file.copy()
                    else:
                        print(f" - FAILED. {cpu_time.total_seconds():.2f}s CPU time")

                except KeyboardInterrupt:
                    print("Meta study interrupted by user. Current configuration will be retried on the next run.")
                    # Remove the partially created group so the run can resume later
                    del results_group[run_name]
                    raise
                except Exception as e:
                    print(f"Failed: {str(e)}")
                    
                    run_group.attrs['error_message'] = str(e)
                    run_group.attrs['success'] = False
                
                result_index += 1
    
    print(f"\nAll simulations completed. Skipped {skipped} existing parameter sets.")
    print(f"Results saved to {results_file}")
    if best_params is not None:
        print(f"Best run achieved {best_timesteps} timesteps with parameters: {best_params}")
    if plot_move_hist and plt is not None and move_dist_records:
        try:
            pathlib.Path("outputs/plots").mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(move_dist_records, bins=20, edgecolor="black", alpha=0.8)
            ax.set_xlabel("move_distance_total")
            ax.set_ylabel("count")
            ax.set_title("Distribution of move_distance_total (meta study)")
            fig.tight_layout()
            plot_path = pathlib.Path("outputs/plots/move_distance_hist.png")
            fig.savefig(plot_path)
            plt.close(fig)
            print(f"Saved move_distance_total histogram to {plot_path}")
        except Exception as exc:
            print(f"Warning: failed to plot move_distance_total histogram: {exc}")
