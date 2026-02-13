from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import networkx as nx

from .graph import Graph
from .types import Edge, Node

if TYPE_CHECKING:
    from .processing_zone import ProcessingZone


class GraphCreator:
    def __init__(
        self,
        m: int,
        n: int,
        ion_chain_size_vertical: int,
        ion_chain_size_horizontal: int,
        failing_junctions: int,
        pz_info: list[ProcessingZone],
        edges_to_delete: list[Edge] | None = None,
        nodes_to_suppress: list[Node] | None = None,
    ):
        self.m = m
        self.n = n
        self.ion_chain_size_vertical = ion_chain_size_vertical
        self.ion_chain_size_horizontal = ion_chain_size_horizontal
        self.failing_junctions = failing_junctions
        self.pz_info = pz_info
        self.edges_to_delete = self._coerce_edges(edges_to_delete or [])
        self.nodes_to_suppress = self._coerce_nodes(nodes_to_suppress or [])
        self.m_extended = self.m + (self.ion_chain_size_vertical - 1) * (self.m - 1)
        self.n_extended = self.n + (self.ion_chain_size_horizontal - 1) * (self.n - 1)
        self.networkx_graph = self.create_graph()

    def create_graph(self) -> Graph:
        networkx_graph: Graph = nx.grid_2d_graph(self.m_extended, self.n_extended, create_using=Graph)
        # color all edges black
        nx.set_edge_attributes(networkx_graph, values=dict.fromkeys(networkx_graph.edges(), "k"), name="color")
        # num_edges needed for outer pz (length of one-way connection - exit/entry)
        self._set_trap_nodes(networkx_graph)
        self._remove_edges(networkx_graph)
        self._remove_nodes(networkx_graph)
        networkx_graph.junction_nodes = []
        self._set_junction_nodes(networkx_graph)
        # if self.pz == 'mid':
        #     self._remove_mid_part(networkx_graph)
        self._remove_junctions(networkx_graph, self.failing_junctions)
        nx.set_edge_attributes(networkx_graph, values=dict.fromkeys(networkx_graph.edges(), "trap"), name="edge_type")
        self._set_processing_zone_edges(networkx_graph)
        self._delete_edges(networkx_graph)
        self._suppress_nodes(networkx_graph)
        nx.set_node_attributes(networkx_graph, values=dict.fromkeys(networkx_graph.nodes(), 1), name="weight")

        return networkx_graph

    def _set_trap_nodes(self, networkx_graph: Graph) -> None:
        for node in networkx_graph.nodes():
            networkx_graph.add_node(node, node_type="trap_node", color="b")

    def _remove_edges(self, networkx_graph: Graph) -> None:
        self._remove_horizontal_edges(networkx_graph)
        self._remove_vertical_edges(networkx_graph)

    def _remove_nodes(self, networkx_graph: Graph) -> None:
        self._remove_horizontal_nodes(networkx_graph)

    def _remove_horizontal_edges(self, networkx_graph: Graph) -> None:
        for i in range(0, self.m_extended - self.ion_chain_size_vertical, self.ion_chain_size_vertical):
            for k in range(1, self.ion_chain_size_vertical):
                for j in range(self.n_extended - 1):
                    networkx_graph.remove_edge((i + k, j), (i + k, j + 1))

    def _remove_vertical_edges(self, networkx_graph: Graph) -> None:
        for i in range(0, self.n_extended - self.ion_chain_size_horizontal, self.ion_chain_size_horizontal):
            for k in range(1, self.ion_chain_size_horizontal):
                for j in range(self.m_extended - 1):
                    networkx_graph.remove_edge((j, i + k), (j + 1, i + k))

    def _remove_horizontal_nodes(self, networkx_graph: Graph) -> None:
        for i in range(0, self.m_extended - self.ion_chain_size_vertical, self.ion_chain_size_vertical):
            for k in range(1, self.ion_chain_size_vertical):
                for j in range(0, self.n_extended - self.ion_chain_size_horizontal, self.ion_chain_size_horizontal):
                    for s in range(1, self.ion_chain_size_horizontal):
                        networkx_graph.remove_node((i + k, j + s))

    def _set_junction_nodes(self, networkx_graph: Graph) -> None:
        for i in range(0, self.m_extended, self.ion_chain_size_vertical):
            for j in range(0, self.n_extended, self.ion_chain_size_horizontal):
                networkx_graph.add_node((i, j), node_type="junction_node", color="g")
                networkx_graph.junction_nodes.append((i, j))

    def _remove_junctions(self, networkx_graph: Graph, num_nodes_to_remove: int) -> None:
        """
        Removes a specified number of nodes from the graph, excluding nodes of type 'exit_node' or 'entry_node'.
        """
        #  Filter out nodes that are of type 'exit_node' or 'entry_node'
        nodes_to_remove = [
            node
            for node, data in networkx_graph.nodes(data=True)
            if data.get("node_type") not in {"exit_node", "entry_node"}
        ]

        # Shuffle the list of nodes to remove
        random.seed(0)
        random.shuffle(nodes_to_remove)

        # Remove the specified number of nodes
        for node in nodes_to_remove[:num_nodes_to_remove]:
            networkx_graph.remove_node(node)

    def _delete_edges(self, networkx_graph: Graph) -> None:
        for edge in self.edges_to_delete:
            edge_idc = self._normalize_edge(edge)
            if networkx_graph.has_edge(*edge_idc):
                networkx_graph.remove_edge(*edge_idc)

    def _suppress_nodes(self, networkx_graph: Graph) -> None:
        for node in self.nodes_to_suppress:
            if node not in networkx_graph:
                continue

            neighbors = list(networkx_graph.neighbors(node))
            if len(neighbors) == 2:
                node_a, node_b = neighbors
                attrs_a = networkx_graph.edges[node, node_a]
                attrs_b = networkx_graph.edges[node, node_b]
                merged = self._merge_edge_attributes_many([attrs_a, attrs_b])

                if networkx_graph.has_edge(node_a, node_b):
                    existing = networkx_graph.edges[node_a, node_b]
                    merged = self._merge_edge_attributes_many([existing, merged])
                    networkx_graph.edges[node_a, node_b].update(merged)
                else:
                    networkx_graph.add_edge(node_a, node_b, **merged)

            networkx_graph.remove_node(node)

    def _merge_edge_attributes_many(self, attrs_list: list[dict[str, Any]]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        keys: set[str] = set()
        for attrs in attrs_list:
            keys.update(attrs.keys())

        for key in keys:
            values = [attrs.get(key) for attrs in attrs_list if attrs.get(key) is not None]
            if not values:
                continue

            if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
                merged[key] = sum(values) / len(values)
                continue

            try:
                counts: dict[Any, int] = {}
                order: list[Any] = []
                for value in values:
                    if value not in counts:
                        counts[value] = 0
                        order.append(value)
                    counts[value] += 1
                max_count = max(counts.values())
                tied = [value for value in order if counts[value] == max_count]
                merged[key] = tied[0]
            except TypeError:
                merged[key] = values[0]

        return merged

    def _normalize_edge(self, edge: Edge) -> Edge:
        node_a, node_b = edge
        return (node_a, node_b) if node_a <= node_b else (node_b, node_a)

    def _coerce_nodes(self, nodes: list[Node] | list[list[int]] | list[tuple[int, int]]) -> list[Node]:
        coerced: list[Node] = []
        for node in nodes:
            x, y = node
            coerced.append((int(x), int(y)))
        return coerced

    def _coerce_edges(self, edges: list[Edge] | list[list[Node]] | list[tuple[Node, Node]]) -> list[Edge]:
        coerced: list[Edge] = []
        for edge in edges:
            node_a, node_b = edge
            coerced.append((self._coerce_nodes([node_a])[0], self._coerce_nodes([node_b])[0]))
        return coerced

    def _set_processing_zone_edges(self, networkx_graph: Graph) -> None:
        for pz in self.pz_info:
            edge = tuple(sorted(pz.edge_idc, key=sum))
            if networkx_graph.has_edge(*edge):
                networkx_graph.edges[edge]["edge_type"] = "processing"
                networkx_graph.edges[edge]["color"] = "g"

    def get_graph(self) -> Graph:
        return self.networkx_graph
