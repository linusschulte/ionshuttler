from __future__ import annotations

import pathlib
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING

from .compilation import get_all_first_gates_and_update_sequence_non_destructive, remove_processed_gates
from .cycles import get_ions
from .graph_utils import get_idc_from_idx, get_idx_from_idc
from .plotting import plot_state
from .scheduling import (
    create_cycles_for_moves,
    create_gate_info_list,
    create_move_list,
    create_priority_queue,
    find_movable_cycles,
    find_out_of_entry_moves,
    get_partitioned_priority_queues,
    preprocess,
    rotate_free_cycles,
    update_entry_and_exit_cycles,
)

if TYPE_CHECKING:
    from qiskit.dagcircuit import DAGDependency

    from .graph import Graph
    from .types import Edge


def check_duplicates(graph: Graph) -> None:
    edge_idxs_occupied = []
    for edge_idc in graph.state.values():
        edge_idxs_occupied.append(get_idx_from_idc(graph.idc_dict, edge_idc))
    # Count occurrences of each integer
    counts = Counter(edge_idxs_occupied)

    for idx, count in counts.items():
        edge_idc = get_idc_from_idx(graph.idc_dict, idx)
        if graph.get_edge_data(edge_idc[0], edge_idc[1])["edge_type"] != "parking_edge" and count > 1:
            message = f"More than one ion in edge {edge_idc}, arch: {graph.arch}, circuit depth: {len(graph.sequence)}, seed: {graph.seed}!"
            raise AssertionError(message)

        if (
            graph.get_edge_data(edge_idc[0], edge_idc[1])["edge_type"] == "parking_edge"
            and count > graph.max_num_parking
        ):
            message = f"More than {graph.max_num_parking} chains in parking edge {edge_idc}!"
            raise AssertionError(message)


def build_plot_annotations(graph: Graph, in_process_ions: list[int]) -> tuple[str, str]:
    current_map = graph.current_gate_by_pz
    if not current_map:
        return ("Current gates by PZ: none", "Gates in progress: none")

    title_parts = []
    progress_parts = []
    in_process_set = set(in_process_ions)
    for pz_name in sorted(current_map):
        gate_id = current_map[pz_name]
        try:
            qubits = graph.gate_qubits(gate_id)
            qasm = graph.gate_qasm(gate_id)
        except KeyError:
            qubits = ()
            qasm = "unknown"
        qubit_str = ",".join(str(q) for q in qubits) if qubits else "-"
        title_parts.append(f"{pz_name}: gate {gate_id} (q={qubit_str})")
        if qubits and in_process_set and all(q in in_process_set for q in qubits):
            progress_parts.append(f"{gate_id}: {qasm}")
    title_text = "Current gates by PZ: " + "; ".join(title_parts)
    if progress_parts:
        xlabel_text = "Gates in progress: " + "; ".join(progress_parts)
    else:
        xlabel_text = "Gates in progress: none"
    return title_text, xlabel_text


def find_pz_order(graph: Graph, gate_info_list: dict[str, list[int]]) -> list[str]:
    # find next processing zone that will execute a gate
    pz_order = []
    for gate_id in graph.sequence:
        qubits = graph.gate_qubits(gate_id)
        if len(qubits) == 1:
            ion = qubits[0]
            for pz in graph.pzs:
                if ion in gate_info_list[pz.name]:
                    pz_order.append(pz.name)
                    break
        elif len(qubits) == 2:
            ion1, ion2 = qubits
            for pz in graph.pzs:
                if ion1 in gate_info_list[pz.name] and ion2 in gate_info_list[pz.name]:
                    pz_order.append(pz.name)
                    break
    return pz_order


def shuttle(
    graph: Graph,
    priority_queue: dict[int, str],
    timestep: int,
    cycle_or_paths: str,
    unique_folder: pathlib.Path,
    title_text: str | None,
    xlabel_text: str | None,
) -> None:
    preprocess(graph, priority_queue)

    # Update ion chains after preprocess
    graph.state = get_ions(graph)

    check_duplicates(graph)
    part_prio_queues = get_partitioned_priority_queues(priority_queue)

    all_cycles: dict[int, list[Edge]] = {}
    # Iterate over all processing zones
    # create move list for each pz -> needed to get all cycles
    # priority queue later picks the cycles to rotate
    in_and_into_exit_moves_of_pz = {}
    for pz in graph.pzs:
        prio_queue = part_prio_queues[pz.name]
        move_list = create_move_list(graph, prio_queue, pz)
        cycles, in_and_into_exit_moves = create_cycles_for_moves(graph, move_list, cycle_or_paths, pz)
        # add cycles to all_cycles
        all_cycles = {**all_cycles, **cycles}

    out_of_entry_moves = find_out_of_entry_moves(graph, other_next_edges=[])

    for pz in graph.pzs:
        prio_queue = part_prio_queues[pz.name]
        out_of_entry_moves_of_pz = out_of_entry_moves.get(pz, None)
        if pz.name in in_and_into_exit_moves:
            in_and_into_exit_moves_of_pz = in_and_into_exit_moves[pz.name]
        update_entry_and_exit_cycles(
            graph, pz, all_cycles, in_and_into_exit_moves_of_pz, out_of_entry_moves_of_pz, prio_queue
        )

    # now general priority queue picks cycles to rotate
    chains_to_rotate = find_movable_cycles(graph, all_cycles, priority_queue, cycle_or_paths)
    rotate_free_cycles(graph, all_cycles, chains_to_rotate)

    # Update ions after rotate
    graph.state = get_ions(graph)

    labels = (
        f"timestep and seq length {timestep} {len(graph.sequence)}",
        "Sequence: %s" % [graph.sequence if len(graph.sequence) < 8 else graph.sequence[:8]],
    )

    if graph.plot is True or graph.save is True:
        plot_state(
            graph,
            labels,
            plot_ions=True,
            show_plot=graph.plot,
            save_plot=graph.save,
            plot_cycle=False,
            plot_pzs=False,
            filename=unique_folder / f"{graph.arch}_timestep_{timestep}.png",
            title_text=title_text,
            xlabel_text=xlabel_text,
        )


def main(
    graph: Graph,
    dag: DAGDependency,
    cycle_or_paths: str,
    use_dag: bool,
    gate_partition: dict[str, list[int]] | None = None,
) -> int:
    print("gate partition:", gate_partition)
    timestep = 0
    max_timesteps = 1e6
    graph.state = get_ions(graph)

    unique_folder = pathlib.Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    if graph.save is True:
        unique_folder.mkdir(exist_ok=True, parents=True)

    if any([graph.plot, graph.save]):
        plot_state(
            graph,
            labels=("Initial state", None),
            plot_ions=True,
            show_plot=graph.plot,
            save_plot=graph.save,
            plot_cycle=False,
            plot_pzs=True,
            filename=unique_folder / f"{graph.arch}_timestep_{timestep}.pdf",
        )

    assignment_map: dict[int, str] = dict(graph.gate_pz_assignment)
    if gate_partition is not None:
        assignment_map.clear()
        for pz_name, gate_ids in gate_partition.items():
            if pz_name not in graph.pzs_name_map:
                raise ValueError(f"Unknown processing zone '{pz_name}' in gate partition.")
            for gate_id in gate_ids:
                if gate_id in assignment_map and assignment_map[gate_id] != pz_name:
                    raise ValueError(f"Gate id {gate_id} assigned to multiple processing zones.")
                assignment_map[gate_id] = pz_name
    for gate_id in assignment_map:
        if gate_id not in graph.gate_info:
            raise ValueError(f"Gate id {gate_id} in partition but not present in circuit.")
    graph.gate_pz_assignment = assignment_map
    graph.current_gate_by_pz = {}

    for pz in graph.pzs:
        pz.time_in_pz_counter = 0
        pz.gate_execution_finished = True

    graph.in_process = []

    if use_dag:
        next_processable_gate_nodes = get_all_first_gates_and_update_sequence_non_destructive(graph, dag)

    locked_gates: dict[int, str] = {}
    while timestep < max_timesteps:
        for pz in graph.pzs:
            pz.rotate_entry = False
            pz.out_of_parking_cycle = None
            pz.out_of_parking_move = None

        if use_dag:
            gate_info_list: dict[str, list[int]] = {pz.name: [] for pz in graph.pzs}
            for pz_name, node in next_processable_gate_nodes.items():
                for ion in node.qindices:
                    gate_info_list[pz_name].append(ion)
        else:
            # update gate_info_list (list of next gate at pz)
            gate_info_list = create_gate_info_list(graph)

        # pz order - find next pz that processes a gate (only used to add ions in exit to front of prio queue)
        pz_executing_gate_order = find_pz_order(graph, gate_info_list)

        # # reset locked gates, prio q recalcs them (2-qubit gates get locked after each execution)
        graph.locked_gates = locked_gates
        # priority queue is dict with ions as keys and pz as values
        # (for 2-qubit gates pz may not match the pz of the individual ion)
        priority_queue, next_gate_at_pz_dict = create_priority_queue(graph, pz_executing_gate_order)
        graph.current_gate_by_pz = {
            pz_name: gate_id for pz_name, gate_id in next_gate_at_pz_dict.items() if gate_id is not None
        }
        next_gate_qubits_by_pz = {
            pz_name: () if gate_id is None else graph.gate_qubits(gate_id)
            for pz_name, gate_id in next_gate_at_pz_dict.items()
        }

        # check if ions are already in processing zone
        # -> important for 2-qubit gates
        # -> leave ion in processing zone if needed in a 2-qubit gate
        for i in range(min(len(graph.pzs), len(graph.sequence))):
            gate_id = graph.sequence[i]
            qubits = graph.gate_qubits(gate_id)

            if len(qubits) == 2:
                ion1, ion2 = qubits
                for pz in graph.pzs:
                    state1 = graph.state[ion1]
                    state2 = graph.state[ion2]
                    next_qubits = next_gate_qubits_by_pz.get(pz.name, ())
                    # append ion to in_process if it is in the correct processing zone
                    if state1 == pz.parking_edge and ion1 in next_qubits and ion2 in next_qubits:
                        graph.in_process.append(ion1)
                    if state2 == pz.parking_edge and ion1 in next_qubits and ion2 in next_qubits:
                        graph.in_process.append(ion2)

        title_text, xlabel_text = build_plot_annotations(graph, graph.in_process)

        # shuttle one timestep
        shuttle(
            graph,
            priority_queue,
            timestep,
            cycle_or_paths,
            unique_folder,
            title_text=title_text,
            xlabel_text=xlabel_text,
        )

        # reset ions in process
        graph.in_process = []

        # Check the state of each ion in the sequence
        graph.state = get_ions(graph)

        if use_dag:
            processed_nodes = {}
            gate_id_lookup = getattr(graph, "dag_gate_id_lookup", {})
            for pz_name, gate_node in next_processable_gate_nodes.items():
                pz = graph.pzs_name_map[pz_name]
                gate_id = gate_id_lookup.get(gate_node.node_id)
                if gate_id is None:
                    continue

                gate_qubits = graph.gate_qubits(gate_id)
                if len(gate_qubits) == 1:
                    ion = gate_qubits[0]
                    if get_idx_from_idc(graph.idc_dict, graph.state[ion]) == get_idx_from_idc(
                        graph.idc_dict, pz.parking_edge
                    ):
                        pz.gate_execution_finished = (
                            False  # set False, then check below if gate time is finished -> then True
                        )
                        pz.getting_processed.append(gate_node)
                        pz.time_in_pz_counter += 1
                        gate_time = 1

                        if pz.time_in_pz_counter == gate_time:
                            processed_nodes[pz_name] = gate_node
                            pz.getting_processed.remove(gate_node)
                            # remove the processing zone from the list
                            # (it can only process one ion)
                            # pzs.remove(pz)
                            # graph.in_process.append(ion)

                            pz.time_in_pz_counter = 0
                            pz.gate_execution_finished = True
                            # break
                elif len(gate_qubits) == 2:
                    ion1, ion2 = gate_qubits
                    state1 = graph.state[ion1]
                    state2 = graph.state[ion2]

                    # if both ions are in the processing zone, process the gate
                    if get_idx_from_idc(graph.idc_dict, state1) == get_idx_from_idc(
                        graph.idc_dict, pz.parking_edge
                    ) and get_idx_from_idc(graph.idc_dict, state2) == get_idx_from_idc(graph.idc_dict, pz.parking_edge):
                        pz.gate_execution_finished = (
                            False  # set False, then check below if gate time is finished -> then True
                        )
                        pz.getting_processed.append(gate_node)
                        pz.time_in_pz_counter += 1

                        gate_time = 3
                        if pz.time_in_pz_counter == gate_time:
                            processed_nodes[pz_name] = gate_node
                            # remove the processing zone from the list
                            # (it can only process one gate)
                            # pzs.remove(pz)

                            # remove the locked pz of the processed two-qubit gate
                            if gate_id in graph.locked_gates and graph.locked_gates[gate_id] == pz.name:
                                graph.locked_gates.pop(gate_id)
                            pz.time_in_pz_counter = 0
                            pz.gate_execution_finished = True
                            pz.getting_processed.remove(gate_node)
                            # break
                else:
                    msg = "Invalid gate format"
                    raise ValueError(msg)

        else:
            processed_gate_ids: list[int] = []
            previous_gate_processed = True
            pzs = graph.pzs.copy()
            next_gate_ids = graph.sequence[: min(len(graph.pzs), len(graph.sequence))]
            # go through the first gates in the sequence (as many as pzs or sequence length)
            # for now, gates are processed in order
            # (can only be processed in parallel if previous gates are processed)
            for gate_id in next_gate_ids:
                # only continue if previous ion was processed
                if not previous_gate_processed:
                    break
                gate_qubits = graph.gate_qubits(gate_id)
                gate_processed = False
                # wenn auf weg zu pz in anderer pz -> wird processed?
                # Problem nur für 2-qubit gate?
                for pz in pzs:
                    if len(gate_qubits) == 1:
                        ion = gate_qubits[0]
                        if get_idx_from_idc(graph.idc_dict, graph.state[ion]) == get_idx_from_idc(
                            graph.idc_dict, pz.parking_edge
                        ):
                            pz.gate_execution_finished = (
                                False  # set False, then check below if gate time is finished -> then True
                            )
                            pz.time_in_pz_counter += 1
                            gate_time = 1
                            if pz.time_in_pz_counter == gate_time:
                                processed_gate_ids.insert(0, gate_id)
                                gate_processed = True
                                # remove the processing zone from the list
                                # (it can only process one ion)
                                pzs.remove(pz)
                                # graph.in_process.append(ion)

                                pz.time_in_pz_counter = 0
                                pz.gate_execution_finished = True
                                break
                    elif len(gate_qubits) == 2:
                        ion1, ion2 = gate_qubits
                        state1 = graph.state[ion1]
                        state2 = graph.state[ion2]

                        # if both ions are in the processing zone, process the gate
                        if get_idx_from_idc(graph.idc_dict, state1) == get_idx_from_idc(
                            graph.idc_dict, pz.parking_edge
                        ) and get_idx_from_idc(graph.idc_dict, state2) == get_idx_from_idc(
                            graph.idc_dict, pz.parking_edge
                        ):
                            pz.gate_execution_finished = (
                                False  # set False, then check below if gate time is finished -> then True
                            )
                            pz.time_in_pz_counter += 1
                            gate_time = 3
                            if pz.time_in_pz_counter == gate_time:
                                processed_gate_ids.insert(0, gate_id)
                                gate_processed = True
                                # remove the processing zone from the list
                                # (it can only process one gate)
                                pzs.remove(pz)  # noqa: B909

                                # remove the locked pz of the processed two-qubit gate
                                if gate_id in graph.locked_gates and graph.locked_gates[gate_id] == pz.name:
                                    graph.locked_gates.pop(gate_id)
                                pz.time_in_pz_counter = 0
                                pz.gate_execution_finished = True
                                break
                    else:
                        msg = "Invalid gate format"
                        raise ValueError(msg)
                previous_gate_processed = gate_processed

        # Remove processed ions from the sequence (and dag if use_dag)
        if use_dag:
            if processed_nodes:
                remove_processed_gates(graph, dag, gate_id_lookup, processed_nodes)
                next_processable_gate_nodes = get_all_first_gates_and_update_sequence_non_destructive(graph, dag)
                for pz_name, node in next_processable_gate_nodes.items():
                    gate_id = gate_id_lookup.get(node.node_id)
                    if gate_id is not None:
                        locked_gates[gate_id] = pz_name
        else:
            for gate_id in processed_gate_ids:
                if gate_id in graph.sequence:
                    graph.sequence.remove(gate_id)

        if len(graph.sequence) == 0:
            break

        timestep += 1

    return timestep
