from __future__ import annotations

import os
import pathlib
from datetime import datetime
from typing import TYPE_CHECKING

from .cycles import get_ion_chains
from .plotting import plot_state
from .scheduling import (
    create_cycles_for_moves,
    create_gate_info_list,
    create_move_list,
    create_priority_queue,
    find_movable_cycles,
    get_partitioned_priority_queues,
    preprocess,
    rotate_free_cycles,
)

DEBUG_FLAG = bool(int(os.getenv("IONSHUTTLER_DEBUG_SHUTTLE", "0")))

if TYPE_CHECKING:
    from .graph import Graph
    from .types import Edge


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
) -> None:
    gate_info_list = create_gate_info_list(graph)
    if DEBUG_FLAG:
        print(f"Gate info list: {gate_info_list}")

    pz_executing_gate_order = find_pz_order(graph, gate_info_list)
    if DEBUG_FLAG:
        print(f"Next processing zone executing gate: {pz_executing_gate_order[:10]} ...")

    # new: stop moves (ions that are already in the correct processing zone for a two-qubit gate)
    graph.stop_moves = []

    # "swap" ions in the same processing zone if only one is needed
    for pz in graph.pzs:
        ions_at_pz = graph[pz.edge_idc[0]][pz.edge_idc[1]]["ions"]
        if len(ions_at_pz) >= graph.max_ions_per_pz:
            #ion1, ion2 = ions_at_pz
            for ion in ions_at_pz:
                if ion not in gate_info_list[pz.name]:
                    # ion not needed in this pz, will be swapped out
                    graph[pz.edge_idc[0]][pz.edge_idc[1]]["ions"].remove(ion)
                    graph[pz.edge_idc[0]][pz.edge_idc[1]]["ions"].insert(0, ion)
                    if DEBUG_FLAG:
                        print(f"swapped ion {ion} within pz. Before {ions_at_pz} now: {graph[pz.edge_idc[0]][pz.edge_idc[1]]['ions']}")
                    break
            '''
            if ion2 not in gate_info_list[pz.name]:
                graph[pz.edge_idc[0]][pz.edge_idc[1]]["ions"].remove(ion2)
                graph[pz.edge_idc[0]][pz.edge_idc[1]]["ions"].insert(0, ion2)
                if DEBUG_FLAG:
                    print(f"swapped ion2 within pz. Before {[ion1, ion2]} now: {ions_at_pz}")

            # find the next processing zone that will execute a gate on ion1
            # in case it is needed elsewhere
            # solution to bug where ion1 was needed later in the current pz
            # but also needed elsewhere
            # so was never rotated because other ions where rotating into that pz
            # and swapped with ion1 (ion1 would never be rotated then)
            for pz_name in pz_executing_gate_order:
                if ion1 in gate_info_list[pz_name]:
                    ion1_needed_in_pz = pz_name
                    break

            # TODO ion1 swap also necessary?
            ions_at_pz_before = graph[pz.edge_idc[0]][pz.edge_idc[1]]["ions"].copy()
            if ion1_needed_in_pz is not None and ion1_needed_in_pz != pz.name:
                # ion1 not in gate_info_list[pz.name]:
                graph[pz.edge_idc[0]][pz.edge_idc[1]]["ions"].remove(ion1)
                graph[pz.edge_idc[0]][pz.edge_idc[1]]["ions"].insert(0, ion1)
                ions_at_pz_after = graph[pz.edge_idc[0]][pz.edge_idc[1]]["ions"].copy()
                if DEBUG_FLAG:
                    print(
                        f"[Broken?] swapped back ion1 within pz. Before {ions_at_pz_before} now {ions_at_pz_after}"
                    )

            # new: this could maybe be used to time the gates
            # (bug where two ions were randomly in the correct pz already while only
            # a 1-qubit gate was executed on one of them and next gate was
            # the 2-qubit gate on them -> 3rd ion was moved into them)
            if ion1 in gate_info_list[pz.name] and ion2 in gate_info_list[pz.name]:
                if DEBUG_FLAG:
                    print("Stopping moves for ions already in correct pz for 2-qubit gate:", ion1, ion2)
                graph.stop_moves.append(ion1)
                graph.stop_moves.append(ion2)
            '''

    preprocess(graph, priority_queue)

    # Update ion chains after preprocess
    graph.state = get_ion_chains(graph)
    # print(f"priority queue: {priority_queue}")
    part_prio_queues = get_partitioned_priority_queues(priority_queue)

    all_cycles: dict[int, list[Edge]] = {}
    # Iterate over all processing zones
    # create move list for each pz -> needed to get all cycles
    # priority queue later picks the cycles to rotate
    for pz in graph.pzs:
        prio_queue = part_prio_queues[pz.name]
        move_list = create_move_list(graph, prio_queue, pz)
        # print(f"Priority queue for {pz.name}: {prio_queue}")
        # print(f"Move list for {pz.name}: {move_list}")
        cycles = create_cycles_for_moves(graph, move_list, cycle_or_paths, pz)
        # add cycles to all_cycles
        all_cycles = {**all_cycles, **cycles}
    # print(f"All cycles: {all_cycles}")

    # now general priority queue picks cycles to rotate
    chains_to_rotate = find_movable_cycles(graph, all_cycles, priority_queue, cycle_or_paths)
    # print(f"Chains to rotate: {chains_to_rotate}")

    rotate_free_cycles(graph, all_cycles, chains_to_rotate)

    # new: postprocess?
    # -> ions can already move into processing zone if they pass a junction
    # Update ion chains after rotate
    graph.state = get_ion_chains(graph)
    preprocess(graph, priority_queue)

    labels = (f"timestep {timestep}", f"remaining sequence: {graph.sequence}")

    if graph.plot or graph.save:
        plot_state(
            graph,
            labels,
            plot_ions=True,
            show_plot=graph.plot,
            save_plot=graph.save,
            plot_cycle=False,
            plot_pzs=False,
            filename=unique_folder / f"{graph.arch}_timestep_{timestep}.png",
        )


def main(graph: Graph, cycle_or_paths: str, max_timesteps: int | None = None) -> int:
    timestep = 0
    max_timesteps = int(max_timesteps) if max_timesteps is not None else int(1e6)
    graph.state = get_ion_chains(graph)

    unique_folder = pathlib.Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    if graph.save is True:
        unique_folder.mkdir(exist_ok=True, parents=True)

    if graph.plot or graph.save:
        plot_state(
            graph,
            labels=("Initial state", None),
            plot_ions=True,
            show_plot=graph.plot,
            save_plot=graph.save,
            plot_cycle=False,
            plot_pzs=True,
            filename=unique_folder / f"{graph.arch}_timestep_{timestep}.png",
        )

    graph.in_process = []
    graph.locked_gates = {}
    graph.executed_gates_next = []
    while timestep < max_timesteps:
        if timestep % 10 == 0:
            print(f"\rTimestep: {timestep}/{int(max_timesteps)}", end="", flush=True)
        if DEBUG_FLAG:
            print("------------------------------------------------------")
            print(f"\nStarting timestep {timestep}")
            if not graph.gate_pz_assignment:
                print("locked_gates", graph.locked_gates)
            print(f"Upcoming sequence: {[(gate_id, graph.gate_info[gate_id].qubits) for gate_id in graph.sequence[:10]]}")

        # priority queue is dict with ions as keys and pz as values
        # (for 2-qubit gates pz may not match the pz of the individual ion)
        priority_queue, next_gate_at_pz = create_priority_queue(graph, graph.sequence)

        # check if ions are already in processing zone ->
        # important for 2-qubit gates
        # -> leave ion in processing zone if needed in a 2-qubit gate
        for i in range(min(len(graph.pzs), len(graph.sequence))):
            # only continue if previous ion was processed
            gate_id = graph.sequence[i]
            qubits = graph.gate_qubits(gate_id)

            if len(qubits) == 2:
                ion1, ion2 = qubits
                for pz in graph.pzs:
                    state1 = graph.state[ion1]
                    state2 = graph.state[ion2]
                    next_gate_id = next_gate_at_pz.get(pz.name)
                    next_qubits = graph.gate_qubits(next_gate_id) if next_gate_id is not None else ()
                    # append ion to in_process if it is in the correct processing zone
                    if state1 == pz.edge_idc and ion1 in next_qubits and ion2 in next_qubits:
                        graph.in_process.append(ion1)
                        # print(f"Added ion {ion1} to in_process")
                    if state2 == pz.edge_idc and ion1 in next_qubits and ion2 in next_qubits:
                        graph.in_process.append(ion2)
                        # print(f"Added ion {ion2} to in_process")

        # print('in process before shuttling:', graph.in_process)

        # shuttle one timestep
        shuttle(graph, priority_queue, timestep, cycle_or_paths, unique_folder)

        # reset ions in process
        graph.in_process = []

        # Check the state of each ion in the sequence
        graph.state = get_ion_chains(graph)
        processed_gates: list[int] = []
        previous_ion_processed = True
        pzs = graph.pzs.copy()
        # go through the first gates in the sequence (as many as pzs or sequence length)
        # for now, gates are processed in order
        # (can only be processed in parallel if previous gates are processed)
        for i in range(min(len(graph.pzs), len(graph.sequence))):
            # only continue if previous ion was processed
            if not previous_ion_processed:
                break
            gate_id = graph.sequence[i]
            qubits = graph.gate_qubits(gate_id)
            ion_processed = False
            if DEBUG_FLAG:
                print(f"---> checking out gate {i}: {gate_id} {qubits}")
            # wenn auf weg zu pz in anderer pz -> wird processed?
            # Problem nur für 2-qubit gate? -> TODO fix
            for pz in pzs:
                if len(qubits) == 1:
                    ion = qubits[0]
                    if graph.state[ion] == pz.edge_idc:
                        if DEBUG_FLAG:
                            print(f"Ion {ion} at Processing Zone {pz.name}")
                        processed_gates.insert(0, gate_id)
                        graph.executed_gates_next.append(
                            {
                                "id": f"t{timestep}_q{ion}",
                                "type": "ONE_QUBIT",
                                "qubits": [ion],
                                "edge": [pz.edge_idc[0], pz.edge_idc[1]],
                                "duration": 1,
                                "pz": pz.name,
                            }
                        )
                        ion_processed = True
                        # remove the processing zone from the list
                        # (it can only process one ion)
                        pzs.remove(pz)
                        # graph.in_process.append(ion)
                        break
                elif len(qubits) == 2:
                    ion1, ion2 = qubits
                    state1 = graph.state[ion1]
                    state2 = graph.state[ion2]

                    # The following is now done at the beginning of next timestep
                    # (otherwise would do it double
                    # -> would leave the ones from last time step in in_process
                    # -> would not move even though
                    # they are move out of pz by preprocessing)
                    # append ion to in_process if it is in the correct processing zone
                    # if state1 == pz.edge_idc and ion1 in next_gate_at_pz[pz.name]
                    # and ion2 in next_gate_at_pz[pz.name]:
                    #     graph.in_process.append(ion1)
                    # if state2 == pz.edge_idc and ion1 in next_gate_at_pz[pz.name]
                    # and ion2 in next_gate_at_pz[pz.name]: # also 1 qubit gate?
                    #     graph.in_process.append(ion2)

                    # if both ions are in the processing zone, process the gate
                    if state1 == pz.edge_idc and state2 == pz.edge_idc:
                        if DEBUG_FLAG:
                            print(f"Ions {ion1} and {ion2} at Processing Zone {pz.name}")
                        processed_gates.insert(0, gate_id)
                        graph.executed_gates_next.append(
                            {
                                "id": f"t{timestep}_q{ion1}_{ion2}",
                                "type": "TWO_QUBIT",
                                "qubits": [ion1, ion2],
                                "edge": [pz.edge_idc[0], pz.edge_idc[1]],
                                "duration": 3,
                                "pz": pz.name,
                            }
                        )
                        ion_processed = True
                        # remove the processing zone from the list
                        # (it can only process one gate)
                        pzs.remove(pz)  # noqa: B909

                        # remove the locked pz of the processed two-qubit gate
                        if gate_id in graph.locked_gates and graph.locked_gates[gate_id] == pz.name:
                            graph.locked_gates.pop(gate_id)
                        break
                else:
                    msg = "Invalid gate format"
                    raise ValueError(msg)
            previous_ion_processed = ion_processed

        if DEBUG_FLAG:
            print("Processed gates this timestep:", processed_gates)

        # Remove processed ions from the sequence
        for gate_id in processed_gates:
            graph.sequence.remove(gate_id)

        if len(graph.sequence) == 0:
            if DEBUG_FLAG:
                print(f"\n ----- Final Timesteps: {timestep} -----")
            break

        timestep += 1

    print(f"\rTimestep: {timestep}/{int(max_timesteps)}", end="", flush=True)
    print()
    return timestep
