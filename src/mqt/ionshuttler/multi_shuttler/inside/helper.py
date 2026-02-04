from __future__ import annotations

from .processing_zone import ProcessingZone
from .types import Edge


def _boundary_edge_candidates(m: int, n: int, v: int, h: int) -> list[Edge]:
    if m < 2 or n < 2:
        msg = "Inside PZ placement requires m >= 2 and n >= 2."
        raise ValueError(msg)

    max_row = (m - 1) * v
    max_col = (n - 1) * h

    # Corner-adjacent boundary edges, used first.
    prioritized: list[Edge] = [
        ((0, 1), (0, 2)),
        ((1, max_col), (2, max_col)),
        ((max_row, max_col-1), (max_row, max_col - 2)),
        ((max_row-1, 0), (max_row - 2, 0)),
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


def generate_pzs(num_pzs: int, m: int, n: int, v: int, h: int) -> dict[str, ProcessingZone]:
    if num_pzs < 1:
        msg = "num_pzs must be >= 1."
        raise ValueError(msg)

    candidates = _boundary_edge_candidates(m, n, v, h)
    if num_pzs > len(candidates):
        msg = f"Requested {num_pzs} PZs, but only {len(candidates)} boundary edges are available."
        raise ValueError(msg)

    pz_definitions: dict[str, ProcessingZone] = {}
    for idx, edge in enumerate(candidates[:num_pzs], start=1):
        pz_definitions[f"pz{idx}"] = ProcessingZone(name=f"pz{idx}", edge_idc=edge)
    return pz_definitions


def recalculate_architecture_config(meta_study_config: dict, population_density: float) -> dict:
    return meta_study_config
