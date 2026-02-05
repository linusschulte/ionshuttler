from __future__ import annotations

from .processing_zone import ProcessingZone
from .types import Edge


def _boundary_edge_candidates(m: int, n: int, v: int, h: int) -> list[Edge]:
    if m < 2 or n < 2:
        msg = "Inside PZ placement requires m >= 2 and n >= 2."
        raise ValueError(msg)
    offset = 1 if m > 2 or n > 2 else 0

    max_row = (m - 1) * v
    max_col = (n - 1) * h

    # Corner-adjacent boundary edges, used first.
    prioritized: list[Edge] = [
        ((0, offset), (0, 1+offset)),
        ((max_row, max_col-offset), (max_row, max_col - 1 -offset)),
        ((offset, max_col), (1+offset, max_col)),
        ((max_row-offset, 0), (max_row - 1-offset, 0)),
    ]

    # Deterministic fallback pool if more than four PZs are requested.
    all_boundary: list[Edge] = []
    for col in range(0, max_col, h):
        all_boundary.append(((0, col), (0, col + h)))
        all_boundary.append(((max_row, col), (max_row, col + h)))
    for row in range(0, max_row, v):
        all_boundary.append(((row, 0), (row + v, 0)))
        all_boundary.append(((row, max_col), (row + v, max_col)))

    dedup: list[Edge] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for edge in [*prioritized, *all_boundary]:
        normalized = tuple(sorted(edge, key=sum))
        if normalized in seen:
            continue
        dedup.append(normalized)
        seen.add(normalized)
    return dedup


def _round_robin_edges_by_line(lines: dict[int, list[Edge]]) -> list[Edge]:
    """Take one edge per line in rounds to keep per-line counts balanced."""
    line_ids = sorted(lines)
    queue = {line_id: list(edges) for line_id, edges in lines.items()}
    selected: list[Edge] = []
    while any(queue[line_id] for line_id in line_ids):
        for line_id in line_ids:
            if queue[line_id]:
                selected.append(queue[line_id].pop(0))
    return selected


def _even_then_odd(edges: list[Edge]) -> list[Edge]:
    return [*edges[::2], *edges[1::2]]


def _all_line_balanced_candidates(m: int, n: int, v: int, h: int) -> list[Edge]:
    """Horizontals first (balanced per row), then verticals (balanced per column)."""
    max_row = (m - 1) * v
    max_col = (n - 1) * h

    horizontal_lines: dict[int, list[Edge]] = {}
    for row in range(0, max_row + 1, v):
        row_edges = [((row, col), (row, col + h)) for col in range(0, max_col, h)]
        horizontal_lines[row] = _even_then_odd(row_edges)

    vertical_lines: dict[int, list[Edge]] = {}
    for col in range(0, max_col + 1, h):
        col_edges = [((row, col), (row + v, col)) for row in range(0, max_row, v)]
        vertical_lines[col] = _even_then_odd(col_edges)

    horizontals = _round_robin_edges_by_line(horizontal_lines)
    verticals = _round_robin_edges_by_line(vertical_lines)
    return [*horizontals, *verticals]


def generate_pzs(num_pzs: int, m: int, n: int, v: int, h: int) -> dict[str, ProcessingZone]:
    if num_pzs < 1:
        msg = "num_pzs must be >= 1."
        raise ValueError(msg)

    if num_pzs <= 4:
        candidates = _boundary_edge_candidates(m, n, v, h)
    else:
        candidates = _all_line_balanced_candidates(m, n, v, h)
    if num_pzs > len(candidates):
        msg = f"Requested {num_pzs} PZs, but only {len(candidates)} candidate edges are available."
        raise ValueError(msg)

    pz_definitions: dict[str, ProcessingZone] = {}
    for idx, edge in enumerate(candidates[:num_pzs], start=1):
        processing_zone = edge[0]
        pz_definitions[f"pz{idx}"] = ProcessingZone(
            name=f"pz{idx}",
            edge_idc=edge,
            processing_zone=processing_zone,
        )
    return pz_definitions


def recalculate_architecture_config(meta_study_config: dict, population_density: float) -> dict:
    return meta_study_config
