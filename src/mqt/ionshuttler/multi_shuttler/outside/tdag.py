from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

from qiskit.dagcircuit import DAGDependency, DAGDepNode

from .compilation import (
    build_node_gate_id_lookup,
    create_dag,
    create_initial_sequence,
    manual_copy_dag,
)

if TYPE_CHECKING:
    from .graph import Graph
    from .types import GateInfo


def compute_gate_partition_tdag(
    graph: "Graph",
    dag: DAGDependency | None,
    capacity: int = 4,
    qasm_file_path: Path | None = None,
    balance_penalty: float = 0.25,
) -> dict[str, list[list[int]] | dict[str, list[int]] | dict[int, str]]:
    """Partition gates using the TDAG strategy described in the GLSVLSI'23 paper.

    If no ``dag`` is provided, a fresh DAG is built from ``qasm_file_path`` and
    the graph's ``sequence``/``gate_info`` are populated when missing.
    Blocks are clustered across PZs by preferring qubit overlap and lightly
    penalizing load (``balance_bias``).
    """

    parsed_circuit = None
    if dag is None:
        if qasm_file_path is None:
            msg = "TDAG partitioning requires either a DAGDependency instance or a QASM file path."
            raise ValueError(msg)
        dag = create_dag(qasm_file_path)
    if (not getattr(graph, "_sequence", None)) or (not getattr(graph, "_gate_info", None)):
        if qasm_file_path is None:
            msg = "TDAG partitioning requires gate metadata; provide qasm_file_path when none is set on the graph."
            raise ValueError(msg)
        parsed_circuit = parsed_circuit or create_initial_sequence(qasm_file_path)
        graph.sequence = parsed_circuit.sequence.copy()
        graph.gate_info = parsed_circuit.gate_info
    if capacity <= 0:
        msg = "max_qubits_per_block must be a positive integer"
        raise ValueError(msg)
    if not graph.sequence:
        return {"blocks": [], "gate_partition_for_run": {}, "gate_assignment": {}}

    working_dag = manual_copy_dag(dag)
    node_to_gate = build_node_gate_id_lookup(working_dag, graph.gate_info)
    gate_to_node = {gate_id: node_id for node_id, gate_id in node_to_gate.items()}
    active_sequence = list(graph.sequence)

    blocks: list[list[int]] = []
    while active_sequence:
        deps_by_node = _compute_k_limited_dependencies(working_dag, capacity)
        qubit_groups = _enumerate_groups(working_dag, deps_by_node, capacity)
        candidates = _build_candidate_blocks(active_sequence, graph.gate_info, qubit_groups)
        if not candidates:
            fallback_gate = active_sequence[0]
            fallback_group = frozenset(graph.gate_info[fallback_gate].qubits)
            candidates = [(fallback_group, [fallback_gate])]

        best_group, best_gate_ids = max(
            candidates,
            key=lambda candidate: _score_candidate(graph, candidate[1]),
        )
        if not best_gate_ids:
            best_gate_ids = [active_sequence[0]]

        blocks.append(best_gate_ids)
        _remove_from_dag_and_sequence(
            working_dag,
            active_sequence,
            best_gate_ids,
            gate_to_node,
            node_to_gate,
        )

    # Cluster blocks onto PZs with overlap preference and balancing bias
    pz_names = [pz.name for pz in graph.pzs]
    gate_partition_for_run: dict[str, list[int]] = {name: [] for name in pz_names}
    gate_assignment: dict[int, str] = {}
    pz_qubits: dict[str, set[int]] = {name: set() for name in pz_names}
    pz_load: dict[str, int] = {name: 0 for name in pz_names}

    if pz_names:
        for block in blocks:
            block_qubits = set(q for gid in block for q in graph.gate_info[gid].qubits)
            best_pz = None
            best_score = float("-inf")
            for pz_name in pz_names:
                overlap = len(block_qubits & pz_qubits[pz_name])
                score = overlap - balance_penalty * pz_load[pz_name]
                if score > best_score or (
                    score == best_score and pz_load[pz_name] < pz_load.get(best_pz, float("inf"))
                ):
                    best_score = score
                    best_pz = pz_name

            target = best_pz or pz_names[0]
            gate_partition_for_run[target].extend(block)
            pz_load[target] += len(block)
            pz_qubits[target].update(block_qubits)
            for gate_id in block:
                gate_assignment[gate_id] = target

    return {
        "blocks": blocks,
        "gate_partition_for_run": gate_partition_for_run,
        "gate_assignment": gate_assignment,
    }


def _compute_k_limited_dependencies(
    dag: DAGDependency,
    max_qubits_per_block: int,
) -> dict[int, set[int]]:
    """Collect dependencies for each node and truncate to ``k`` qubits."""

    dependencies: dict[int, set[int]] = {}
    for node in _topological_op_nodes(dag):
        current = set(node.qindices)
        for predecessor in dag.direct_predecessors(node.node_id):
            if getattr(predecessor, "type", None) != "op":
                continue
            current.update(dependencies.get(predecessor.node_id, set(predecessor.qindices)))

        if len(current) > max_qubits_per_block:
            current = set(sorted(current)[:max_qubits_per_block])
        dependencies[node.node_id] = current
    return dependencies


def _enumerate_groups(
    dag: DAGDependency,
    deps_by_node: Mapping[int, set[int]],
    max_qubits_per_block: int,
) -> set[frozenset[int]]:
    """Enumerate unique qubit groups via DFS, pruning when groups exceed ``k`` qubits."""

    groups: set[frozenset[int]] = set()

    def dfs(node: DAGDepNode, active_group: set[int]) -> None:
        merged = set(active_group)
        merged.update(deps_by_node.get(node.node_id, set()))
        if len(merged) > max_qubits_per_block:
            return
        if merged:
            groups.add(frozenset(merged))
        for successor in dag.direct_successors(node.node_id):
            if getattr(successor, "type", None) != "op":
                continue
            dfs(successor, merged)

    for start_node in _topological_op_nodes(dag):
        dfs(start_node, set())

    return groups


def _build_candidate_blocks(
    active_sequence: list[int],
    gate_info: Mapping[int, "GateInfo"],
    groups: Iterable[frozenset[int]],
) -> list[tuple[frozenset[int], list[int]]]:
    """Map qubit groups to ordered gate lists drawn from the active sequence."""

    candidates: list[tuple[frozenset[int], list[int]]] = []
    if not groups:
        return candidates

    for group in groups:
        gate_ids: list[int] = []
        for gate_id in active_sequence:
            qubits = gate_info[gate_id].qubits
            if not qubits:
                continue
            if set(qubits).issubset(group):
                gate_ids.append(gate_id)
        if gate_ids:
            candidates.append((group, gate_ids))
    return candidates


def _score_candidate(graph: "Graph", gate_ids: list[int]) -> tuple[int, int]:
    two_qubit_count = sum(1 for gate_id in gate_ids if len(graph.gate_info[gate_id].qubits) == 2)
    return two_qubit_count, len(gate_ids)


def _remove_from_dag_and_sequence(
    dag: DAGDependency,
    active_sequence: list[int],
    gate_ids: Iterable[int],
    gate_to_node: dict[int, int],
    node_to_gate: dict[int, int],
) -> None:
    """Discard executed gates from the working DAG and sequence view."""

    gate_id_set = set(gate_ids)
    for gate_id in gate_ids:
        node_id = gate_to_node.pop(gate_id, None)
        if node_id is None:
            continue
        node_to_gate.pop(node_id, None)
        dag._multi_graph.remove_node(node_id)

    active_sequence[:] = [gate_id for gate_id in active_sequence if gate_id not in gate_id_set]


def _topological_op_nodes(dag: DAGDependency) -> Iterable[DAGDepNode]:
    """Yield DAG nodes in topological order, restricted to op nodes."""

    if hasattr(dag, "topological_op_nodes"):
        return dag.topological_op_nodes()
    return (node for node in dag.topological_nodes() if getattr(node, "type", None) == "op")
