from __future__ import annotations

import math
import os
import re
from typing import TYPE_CHECKING

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dagdependency
from qiskit.dagcircuit import DAGDependency
from qiskit.transpiler.passes import RemoveBarriers, RemoveFinalMeasurements

from .cycles import get_state_idxs
from .graph_utils import create_dist_dict, update_distance_map
from .scheduling import pick_pz_for_2_q_gate
from .types import GateInfo, ParsedCircuit

if TYPE_CHECKING:
    from pathlib import Path

    from qiskit.dagcircuit import DAGDepNode

    from .graph import Graph


def is_qasm_file(file_path: Path) -> bool:
    with file_path.open(encoding="utf-8") as file:
        # Read the first line of the file (7th line, specific to MQT Bench)
        first_line = ""
        for _f in range(7):
            prev_line = first_line
            first_line = file.readline()
        # Check if the first line contains the OPENQASM identifier
        return "OPENQASM" in first_line or "OPENQASM" in prev_line


def extract_qubits_from_gate(gate_line: str) -> list[int]:
    """Extract qubit numbers from a gate operation line."""
    # Regular expression to match qubits (assuming they are in the format q[<number>])
    pattern = re.compile(r"q\[(\d+)\]")
    matches = pattern.findall(gate_line)

    # Convert matched qubit numbers to integers
    return [int(match) for match in matches]


def parse_qasm(filename: Path) -> ParsedCircuit:
    """Parse a QASM file and return per-gate metadata indexed by a unique gate id."""
    sequence: list[int] = []
    gate_info: dict[int, GateInfo] = {}
    with filename.open(encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(("//", "#")):
                continue

            # Check if line represents a gate operation
            if line.startswith(("OPENQASM", "include", "qreg", "creg", "gate", "barrier", "measure")):
                continue

            qubits = extract_qubits_from_gate(line)
            if not qubits:
                continue

            gate_id = len(sequence)
            sequence.append(gate_id)
            gate_info[gate_id] = GateInfo(qubits=tuple(qubits), qasm=line)

    return ParsedCircuit(sequence=sequence, gate_info=gate_info)


def compile_qasm_file(filename: Path) -> ParsedCircuit:
    """Compile a QASM file and return the compiled sequence metadata."""
    # Check if the file is a valid QASM file
    if not is_qasm_file(filename):
        msg = "Invalid QASM file format"
        raise ValueError(msg)
    # Parse the QASM file to extract the qubits used for each gate
    return parse_qasm(filename)


def get_front_layer(dag: DAGDependency) -> list[DAGDepNode]:
    """Get the front layer of the DAG."""
    front_layer = []
    for node in dag.get_nodes():
        # If a node has no predecessors, it's in the front layer
        if not dag.direct_predecessors(node.node_id):
            front_layer.append(node)
    return front_layer


def remove_node(dag: DAGDependency, node: DAGDepNode) -> None:
    """Execute a node and update the DAG (remove the node and its edges)."""
    # if dag.direct_successors(node.node_id):
    #    for successor in dag.direct_successors(node.node_id):
    #        dag._multi_graph.remove_edge(node.node_id, successor)
    dag._multi_graph.remove_node(node.node_id)


def build_node_gate_id_lookup(
    dag: DAGDependency,
    gate_info: dict[int, GateInfo],
) -> dict[int, int]:
    """Return a mapping from DAG node id to parsed gate id matching gate metadata."""
    from collections import defaultdict, deque

    gate_buckets: dict[tuple[tuple[int, ...], str], deque[int]] = defaultdict(deque)
    for gate_id in sorted(gate_info):
        meta = gate_info[gate_id]
        qasm_base = re.split(r'[\(\s\[]', meta.qasm)[0]
        gate_buckets[(meta.qubits, qasm_base)].append(gate_id)

    #print("GATE BUCKETS:", {k: list(v) for k, v in gate_buckets.items()})

    node_lookup: dict[int, int] = {}
    for node in dag.topological_nodes():  # type: ignore[attr-defined]
        #print("NODE:", node.node_id, node.op, node.qindices)
        if getattr(node, "type", None) != "op":
            continue
        qubits = tuple(node.qindices)
        qasm_repr = getattr(node.op, "qasm", None)
        qasm_str = qasm_repr() if callable(qasm_repr) else node.op.name
        key = (qubits, qasm_str)
        #print("KEY:", key)
        if key not in gate_buckets or not gate_buckets[key]:
            msg = (
                f"No gate metadata matches DAG node {node.node_id} "
                f"(op={node.op}, qubits={qubits})."
            )
            raise ValueError(msg)
        node_lookup[node.node_id] = gate_buckets[key].popleft()
    return node_lookup


def find_best_gate(
    graph: Graph,
    front_layer: list[DAGDepNode],
    dist_map: dict[int, dict[str, int]],
    gate_info_map: dict[DAGDepNode, str],
) -> DAGDepNode:
    """Find the best gate to execute based on distance."""
    min_gate_cost = math.inf
    for _, gate_node in enumerate(front_layer):
        qubit_indices = gate_node.qindices
        pz_of_node = gate_info_map[gate_node]
        pz = graph.pzs_name_map[pz_of_node]
        if gate_node in pz.getting_processed:
            return gate_node
        gate_cost = max(dist_map[qs][pz_of_node] for qs in qubit_indices)
        # if both ions of 2-qubit gate are in pz execute 2-qubit gate
        if len(qubit_indices) == 2 and gate_cost == 0:
            return gate_node
        if gate_cost < min_gate_cost:
            min_gate_cost = gate_cost
            best_gate = gate_node
    return best_gate


def manual_copy_dag(dag: DAGDependency) -> DAGDependency:
    new_dag = DAGDependency()
    # Recreate quantum registers in the new DAG
    for qreg in dag.qregs.values():
        new_dag.add_qreg(qreg)
    # Iterate over all operation nodes in the original DAG and copy them
    for node in dag.get_nodes():
        new_dag.add_op_node(node.op, node.qargs, node.cargs)
    return new_dag


def create_dag(filename: Path) -> DAGDependency:
    qc = QuantumCircuit.from_qasm_file(str(filename))
    # Remove barriers
    qc = RemoveBarriers()(qc)
    # Remove measurement operations
    qc = RemoveFinalMeasurements()(qc)
    return circuit_to_dagdependency(qc)


def create_initial_sequence(filename: Path) -> ParsedCircuit:
    # assert file is a qasm file
    assert is_qasm_file(filename), "The file is not a valid QASM file."
    return parse_qasm(filename)


def create_updated_sequence_destructive(
    graph: Graph,
    filename: Path,
    dag_dep: DAGDependency,
    use_dag: bool,
) -> tuple[list[int], DAGDependency | None, dict[int, GateInfo]]:
    """Create an updated gate-id sequence, optionally using DAG ordering information."""
    assert is_qasm_file(filename), "The file is not a valid QASM file."

    parsed_circuit = parse_qasm(filename)

    if not use_dag:
        return parsed_circuit.sequence.copy(), None, parsed_circuit.gate_info

    working_dag = manual_copy_dag(dag_dep)
    graph.dist_dict = create_dist_dict(graph)
    state = get_state_idxs(graph)
    dist_map = update_distance_map(graph, state)

    node_to_gate_id = build_node_gate_id_lookup(working_dag, graph.gate_info)
    if graph.debug_gate_tracking:
        print("Debug: DAG node to gate ID mapping (first 5):")
        for idx, (node_id, gate_id) in enumerate(node_to_gate_id.items()):
            gate_meta = parsed_circuit.gate_info[gate_id]
            print(
                f"  #{idx} node {node_id}: gate_id {gate_id}, qubits {gate_meta.qubits}, qasm '{gate_meta.qasm}'"
            )
            if idx >= 4:
                break
    if len(node_to_gate_id) != len(parsed_circuit.sequence):
        msg = (
            "Mismatch between parsed gates and DAG operations: "
            f"{len(parsed_circuit.sequence)} parsed vs {len(node_to_gate_id)} DAG nodes."
        )
        raise ValueError(msg)

    updated_sequence: list[int] = []
    while True:
        first_gates = get_front_layer(working_dag)
        if not first_gates:
            break

        pz_info_map = map_front_gates_to_pzs(
            graph,
            front_layer_nodes=first_gates,
            gate_id_lookup=node_to_gate_id,
        )
        gate_info_map = {value: key for key, values in pz_info_map.items() for value in values}

        for pz_name in pz_info_map:
            if pz_info_map[pz_name]:
                first_gate_to_execute = find_best_gate(graph, pz_info_map[pz_name], dist_map, gate_info_map)
                remove_node(working_dag, first_gate_to_execute)
                gate_id = node_to_gate_id.get(first_gate_to_execute.node_id)
                if gate_id is None:
                    msg = f"Unable to map DAG node {first_gate_to_execute.node_id} to a gate id."
                    raise KeyError(msg)
                updated_sequence.append(gate_id)

    if graph.debug_gate_tracking:
        print("Debug: Parsed vs DAG-derived sequence IDs (first 20):")
        for idx, (parsed_id, dag_id) in enumerate(
            zip(parsed_circuit.sequence, updated_sequence)
        ):
            print(
                f"  index {idx}: parsed {parsed_id}, dag {dag_id}, "
                f"qubits parsed {parsed_circuit.gate_info[parsed_id].qubits}, "
                f"qubits dag {parsed_circuit.gate_info[dag_id].qubits}"
            )
            if idx >= 19:
                break

    return updated_sequence, dag_dep, parsed_circuit.gate_info


def get_front_layer_non_destructive(dag: DAGDependency, virtually_processed_nodes: set[int]) -> list[DAGDepNode]:
    """Get the front layer of the DAG without modifying it."""
    front_layer = []

    for node in dag.get_nodes():
        # Skip nodes we've already processed
        if node.node_id in virtually_processed_nodes:
            continue

        # Check if all predecessors have been processed
        predecessors = dag.direct_predecessors(node.node_id)
        if not predecessors or all(pred in virtually_processed_nodes for pred in predecessors):
            front_layer.append(node)

    return front_layer


def map_front_gates_to_pzs(
    graph: Graph,
    front_layer_nodes: list[DAGDepNode],
    gate_id_lookup: dict[int, int],
) -> dict[str, list[DAGDepNode]]:
    """Create list of all front layer gates at each processing zone."""
    gates_of_pz_info: dict[str, list[DAGDepNode]] = {pz.name: [] for pz in graph.pzs}
    for seq_node in front_layer_nodes:
        gate_id = gate_id_lookup[seq_node.node_id]
        gate_info = graph.gate_info[gate_id]
        qubits = gate_info.qubits
        preferred_pz = graph.preferred_pz_for_gate(gate_id)
        if preferred_pz is not None and preferred_pz not in graph.pzs_name_map:
            msg = f"Preferred processing zone '{preferred_pz}' for gate {gate_id} does not exist."
            raise ValueError(msg)

        if len(qubits) == 1:
            ion = qubits[0]
            pz = preferred_pz or graph.map_to_pz[ion]
        elif len(qubits) == 2:
            if preferred_pz is not None:
                pz = preferred_pz
                graph.locked_gates[gate_id] = pz
            else:
                if gate_id not in graph.locked_gates:
                    pz = pick_pz_for_2_q_gate(graph, qubits[0], qubits[1])
                    graph.locked_gates[gate_id] = pz
                else:
                    pz = graph.locked_gates[gate_id]
        else:
            msg = f"Unsupported gate arity {len(qubits)} for gate id {gate_id}."
            raise ValueError(msg)

        gates_of_pz_info[pz].append(seq_node)
    # print('\ngates of pz info: ', {pz: [node.qindices for node in nodes] for pz, nodes in gates_of_pz_info.items()}, '\n')
    return gates_of_pz_info


def remove_processed_gates(
    graph: Graph,
    dag: DAGDependency,
    gate_id_lookup: dict[int, int],
    removed_nodes: dict[str, DAGDepNode],
) -> None:
    """
    Remove the processed gates of each processing zone from both the DAG and sequence.

    Args:
        graph: Graph object containing the gate sequence
        dag: DAG representing dependencies between gates
        first_gates_by_pz: Dictionary mapping processing zones to their first gates
    """
    # Track which gates are removed
    removed_gate_ids: list[int] = []

    # Process each processing zone's first gate
    for _pz_name, first_gate in removed_nodes.items():
        gate_id = gate_id_lookup.get(first_gate.node_id)
        if gate_id is None:
            continue

        if gate_id in graph.sequence:
            graph.sequence.remove(gate_id)
            removed_gate_ids.append(gate_id)
            if graph.debug_gate_tracking:
                print(f"Debug: Removed gate id {gate_id} from sequence for PZ {_pz_name}")

        # Remove the gate from the DAG
        node_id = first_gate.node_id
        if dag.get_node(node_id):
            dag._multi_graph.remove_node(node_id)
            # print(f"Removed node {node_id} from DAG for PZ {pz_name}")
        gate_id_lookup.pop(first_gate.node_id, None)


def get_all_first_gates_and_update_sequence_non_destructive(
    graph: Graph,
    dag: DAGDependency,
    gate_id_lookup: dict[int, int] | None = None,
    max_rounds: int = 5,
) -> dict[str, DAGDepNode]:
    """Get the first gates from the DAG for each processing zone (only first round, so they are simultaneously processable).
    Continue finding the subsequent "first gates" and update the sequence accordingly.
    Creates a compiled list of gates (ordered) for each pz from the DAG Dependency."""

    if gate_id_lookup is None:
        gate_id_lookup = getattr(graph, "dag_gate_id_lookup", build_node_gate_id_lookup(dag, graph.gate_info))

    ordered_sequence: list[int] = []
    processed_nodes: set[int] = set()  # Track nodes we've "virtually removed"
    # Dictionary to store the first gate for each processing zone
    first_nodes_by_pz: dict[str, DAGDepNode] = {}

    # update dist map
    state = get_state_idxs(graph)
    dist_map = update_distance_map(graph, state)
    for round_recalc_fl in range(max_rounds):
        # Get front layer excluding already processed nodes
        front_layer_nodes = get_front_layer_non_destructive(dag, processed_nodes)

        # If front layer is empty, done
        if not front_layer_nodes:
            break

        pz_info_map = map_front_gates_to_pzs(graph, front_layer_nodes, gate_id_lookup)
        gate_info_map = {value: key for key, values in pz_info_map.items() for value in values}

        # Track gates processed in this round to ensure maximum parallelism
        round_processed_gates: list[DAGDepNode] = []

        # Process one gate for each processing zone that has available gates
        for pz_name in pz_info_map:
            if pz_info_map[pz_name]:
                # Find the best gate for this processing zone
                best_gate = find_best_gate(graph, pz_info_map[pz_name], dist_map, gate_info_map)

                # Save the first gate that can be processed for each pz (only of first round, since otherwise can not be simultaneously processed)
                if round_recalc_fl == 0 and pz_name not in first_nodes_by_pz:
                    first_nodes_by_pz[pz_name] = best_gate

                # Add to the processed list for this round
                round_processed_gates.append(best_gate)

                gate_id = gate_id_lookup.get(best_gate.node_id)
                if gate_id is None:
                    msg = f"No gate id mapped for node {best_gate.node_id}"
                    raise KeyError(msg)

                # Update the ordered sequence
                ordered_sequence.append(gate_id)

                # Mark as processed
                processed_nodes.add(best_gate.node_id)

        # Remove all processed gates from the original sequence
        for gate_node in round_processed_gates:
            gate_id = gate_id_lookup.get(gate_node.node_id)
            if gate_id is None:
                continue
            if gate_id in graph.sequence:
                graph.sequence.remove(gate_id)

    # Update the final sequence
    graph.sequence = ordered_sequence + graph.sequence

    return first_nodes_by_pz
