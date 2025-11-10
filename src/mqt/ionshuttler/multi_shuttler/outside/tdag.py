from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Mapping

from qiskit.dagcircuit import DAGDependency, DAGDepNode

from .compilation import build_node_gate_id_lookup, manual_copy_dag

if TYPE_CHECKING:
    from .graph import Graph
    from .types import GateInfo


def compute_gate_partition_tdag(
    graph: "Graph",
    dag: DAGDependency,
    max_qubits_per_block: int = 4,
) -> dict[str, list[list[int]]]:
    """Partition gates using the TDAG strategy described in the GLSVLSI'23 paper."""

    if dag is None:
        msg = "TDAG partitioning requires a DAGDependency instance."
        raise ValueError(msg)
    if max_qubits_per_block <= 0:
        msg = "max_qubits_per_block must be a positive integer"
        raise ValueError(msg)
    if not graph.sequence:
        return {"blocks": []}

    working_dag = manual_copy_dag(dag)
    node_to_gate = build_node_gate_id_lookup(working_dag, graph.gate_info)
    gate_to_node = {gate_id: node_id for node_id, gate_id in node_to_gate.items()}
    active_sequence = list(graph.sequence)

    blocks: list[list[int]] = []
    while active_sequence:
        deps_by_node = _compute_k_limited_dependencies(working_dag, max_qubits_per_block)
        qubit_groups = _enumerate_groups(working_dag, deps_by_node, max_qubits_per_block)
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

    return {"blocks": blocks}


def _compute_k_limited_dependencies(
    dag: DAGDependency,
    max_qubits_per_block: int,
) -> dict[int, set[int]]:
    """Collect dependencies for each node and truncate to ``k`` qubits."""

    dependencies: dict[int, set[int]] = {}
    for node in dag.topological_op_nodes():
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

    for start_node in dag.topological_op_nodes():
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
