# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Focused renderers for the six computational manuscript figures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from .analysis import RawRow

mpl.use("Agg")

METHOD_STYLES = {
    "NoDD": ("#222222", "o", "No DD"),
    "IdealizedHahn": ("#7b3294", "s", "Idealized Hahn"),
    "NearestHahn": ("#008837", "^", "Nearest Hahn"),
    "PulseOnlySADD": ("#e08214", "D", "Pulse-only SADD"),
    "FullSADD": ("#2166ac", "o", "Full SADD"),
}
MANUSCRIPT_PANEL_DETUNINGS = (10.0, 1.0, 0.1)


def render_all(
    aggregate: Sequence[RawRow],
    per_case: Sequence[RawRow],
    temporal: Sequence[RawRow],
    spatial: Sequence[RawRow],
    schedules: Sequence[RawRow],
    opportunities: Sequence[RawRow],
    objective: Sequence[RawRow],
    output_dir: Path,
) -> tuple[Path, ...]:
    """Render the six computational result figures in the manuscript.

    Returns:
        Paths to the six rendered PDFs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        output_dir / "figure_2_operating_regime.pdf",
        output_dir / "figure_3_rerouting_benefit.pdf",
        output_dir / "figure_4_profile_awareness.pdf",
        output_dir / "figure_5_proxy_applicability.pdf",
        output_dir / "figure_6_compilation_cost.pdf",
        output_dir / "figure_7_objective_fidelity.pdf",
    )
    _operating_regime(aggregate, per_case, paths[0])
    _rerouting_benefit(aggregate, paths[1])
    _profile_awareness(aggregate, paths[2])
    _proxy_applicability(temporal, spatial, paths[3])
    _compilation_cost(schedules, opportunities, paths[4])
    _objective_fidelity(objective, paths[5])
    return paths


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _operating_regime(
    rows: Sequence[RawRow],
    per_case: Sequence[RawRow],
    path: Path,
) -> None:
    selected = [row for row in rows if row["scenario"] == "operating_envelope" and row["method"] == "FullSADD"]
    detunings = sorted({float(row["detuning"]) for row in selected})
    controls = sorted({float(row["control"]) for row in selected})
    grid = np.full((len(controls), len(detunings)), np.nan)
    for row in selected:
        grid[controls.index(float(row["control"])), detunings.index(float(row["detuning"]))] = float(
            row["mean_log10_ratio"]
        )
    magnitude = max(2.0, float(np.nanmax(np.abs(grid))))
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))
    axis = axes[0]
    image = axis.imshow(
        grid, origin="lower", aspect="equal", cmap="RdBu_r", norm=TwoSlopeNorm(0, -magnitude, magnitude)
    )
    axis.set_xticks(range(len(detunings)), [f"{value:g}" for value in detunings])
    axis.set_yticks(range(len(controls)), [f"{value:g}" for value in controls])
    axis.set_xlabel(r"Detuning $\delta\omega_0$")
    axis.set_ylabel(r"Control-error scale $s_{\rm ctrl}$")
    figure.colorbar(image, ax=axis, label=r"mean $\log_{10}[(1-F)/(1-F_{\rm NoDD})]$")
    axis.set_title("Operating regime: Full SADD vs no DD")
    _absolute_cross_section(
        axes[1],
        per_case,
        varying="detuning",
        fixed="control",
        fixed_value=1.0,
    )
    axes[1].set_xlabel(r"Detuning $\delta\omega_0$")
    axes[1].set_ylabel("Geometric-mean infidelity")
    axes[1].set_title(r"Cross-section at $s_{\rm ctrl}=1$")
    _absolute_cross_section(
        axes[2],
        per_case,
        varying="control",
        fixed="detuning",
        fixed_value=1.0,
    )
    axes[2].set_xlabel(r"Control-error scale $s_{\rm ctrl}$")
    axes[2].set_title(r"Cross-section at $\delta\omega_0=1$")
    axes[2].legend(fontsize="x-small")
    figure.tight_layout()
    _save(figure, path)


def _absolute_cross_section(
    axis: plt.Axes,
    rows: Sequence[RawRow],
    *,
    varying: str,
    fixed: str,
    fixed_value: float,
) -> None:
    operating = [row for row in rows if row["scenario"] == "operating_envelope" and row["profile"] == "flat"]
    available = sorted({float(row[fixed]) for row in operating})
    selected_fixed = min(available, key=lambda value: abs(value - fixed_value))
    for method, (color, marker, label) in METHOD_STYLES.items():
        method_rows = [
            row for row in operating if row["method"] == method and np.isclose(float(row[fixed]), selected_fixed)
        ]
        x_values = sorted({float(row[varying]) for row in method_rows})
        y_values = []
        for x_value in x_values:
            infidelities = [
                max(float(row["mean_infidelity"]), 1e-15)
                for row in method_rows
                if np.isclose(float(row[varying]), x_value)
            ]
            y_values.append(float(np.exp(np.mean(np.log(infidelities)))))
        axis.plot(x_values, y_values, color=color, marker=marker, label=label)
    axis.set_xscale("log")
    axis.set_yscale("log")


def _rerouting_benefit(rows: Sequence[RawRow], path: Path) -> None:
    selected = [row for row in rows if row["scenario"] == "control_heating"]
    detunings = _ordered_panel_detunings({float(row["detuning"]) for row in selected})
    figure, axes = plt.subplots(1, len(detunings), figsize=(4.1 * len(detunings), 3.4), squeeze=False, sharey=True)
    for axis, detuning in zip(axes[0], detunings, strict=True):
        for method in ("NearestHahn", "PulseOnlySADD", "FullSADD"):
            method_rows = sorted(
                (row for row in selected if row["method"] == method and np.isclose(float(row["detuning"]), detuning)),
                key=lambda row: float(row["heating"]),
            )
            if not method_rows:
                continue
            color, marker, label = METHOD_STYLES[method]
            axis.plot(
                [float(row["heating"]) for row in method_rows],
                [float(row["mean_log10_ratio"]) for row in method_rows],
                color=color,
                marker=marker,
                label=label,
            )
        axis.axhline(0.0, color="black", linewidth=0.7)
        _set_heating_axis(axis)
        axis.set_title(rf"$\delta\omega_0={detuning:g}$")
        axis.set_xlabel(r"Motional-error scale $s_{\rm heat}$")
    axes[0, 0].set_ylabel("Mean log-infidelity ratio")
    axes[0, -1].legend(fontsize="small")
    _save(figure, path)


def _profile_awareness(rows: Sequence[RawRow], path: Path) -> None:
    selected = [row for row in rows if row["scenario"] == "profile_awareness" and row["method"] == "FullSADD"]
    detunings = _ordered_panel_detunings({float(row["detuning"]) for row in selected})
    figure, axes = plt.subplots(1, len(detunings), figsize=(4.1 * len(detunings), 3.4), squeeze=False, sharey=True)
    for axis, detuning in zip(axes[0], detunings, strict=True):
        for profile, label, style in (("aware", "Profile-aware", "-"), ("agnostic", "Profile-agnostic", "--")):
            profile_rows = sorted(
                (row for row in selected if row["profile"] == profile and np.isclose(float(row["detuning"]), detuning)),
                key=lambda row: float(row["heating"]),
            )
            axis.plot(
                [float(row["heating"]) for row in profile_rows],
                [float(row["mean_log10_ratio"]) for row in profile_rows],
                color=METHOD_STYLES["FullSADD"][0],
                marker="o",
                linestyle=style,
                markerfacecolor=None if profile == "aware" else "white",
                label=label,
            )
        axis.axhline(0.0, color="black", linewidth=0.7)
        _set_heating_axis(axis)
        axis.set_title(rf"$\delta\omega_0={detuning:g}$")
        axis.set_xlabel(r"Motional-error scale $s_{\rm heat}$")
    axes[0, 0].set_ylabel("Mean log-infidelity ratio")
    axes[0, -1].legend(fontsize="small")
    _save(figure, path)


def _ordered_panel_detunings(available: set[float]) -> tuple[float, ...]:
    """Return manuscript detunings in their displayed panel order."""
    selected = tuple(detuning for detuning in MANUSCRIPT_PANEL_DETUNINGS if detuning in available)
    return selected or tuple(sorted(available, reverse=True))


def _set_heating_axis(axis: plt.Axes) -> None:
    """Use the manuscript heating scale without a negative plotting margin."""
    axis.set_xscale("symlog", linthresh=1e-4)
    axis.set_xlim(left=0.0)


def _proxy_applicability(temporal: Sequence[RawRow], spatial: Sequence[RawRow], path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(5.2, 6.2))
    axes[0].plot(
        [float(row["tau_seconds"]) for row in temporal],
        [float(row["mean_R_chi"]) for row in temporal],
        marker="o",
        color=METHOD_STYLES["FullSADD"][0],
    )
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].axhline(float(temporal[0]["quasistatic_R_J"]), color="gray", linestyle="--")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"Temporal correlation time $\tau_c$ [s]")
    axes[0].set_ylabel(r"Susceptibility ratio $R_\chi$")
    axes[0].set_title("Temporal applicability")
    axes[1].plot(
        [float(row["ell_sites"]) for row in spatial],
        [float(row["mean_R_chi"]) for row in spatial],
        marker="o",
        color=METHOD_STYLES["FullSADD"][0],
    )
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].axhline(float(spatial[0]["rank_one_R_chi"]), color="gray", linestyle="--")
    axes[1].axvline(float(spatial[0]["architecture_span_sites"]), color="gray", linestyle=":")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"Spatial correlation length $\ell_c$ [sites]")
    axes[1].set_ylabel(r"Susceptibility ratio $R_\chi$")
    axes[1].set_title("Spatial applicability")
    figure.tight_layout()
    _save(figure, path)


def _compilation_cost(
    schedules: Sequence[RawRow],
    opportunities: Sequence[RawRow],
    path: Path,
) -> None:
    full = [row for row in schedules if row["method"] == "FullSADD"]
    nearest = {str(row["case"]): row for row in schedules if row["method"] == "NearestHahn"}
    figure, axes = plt.subplots(2, 1, figsize=(5.2, 6.0))
    x = np.asarray([float(row["opportunities_evaluated"]) for row in full])
    y = np.asarray([float(row["runtime_seconds"]) for row in full])
    axes[0].scatter(x, y, marker="o", color=METHOD_STYLES["FullSADD"][0], label="Full SADD")
    axes[0].scatter(
        x,
        [float(nearest[str(row["case"])]["runtime_seconds"]) for row in full],
        marker="o",
        facecolors="none",
        edgecolors=METHOD_STYLES["NearestHahn"][0],
        label="Nearest Hahn",
    )
    positive = (x > 0) & (y > 0)
    if np.count_nonzero(positive) >= 2:
        slope, intercept = np.polyfit(np.log10(x[positive]), np.log10(y[positive]), 1)
        fit_x = np.geomspace(float(np.min(x[positive])), float(np.max(x[positive])), 100)
        axes[0].plot(fit_x, 10**intercept * fit_x**slope, color="black", linestyle="--", label=rf"$N^{{{slope:.2f}}}$")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Evaluated opportunities")
    axes[0].set_ylabel("Compilation time [s]")
    axes[0].legend(fontsize="small")
    runtimes = np.asarray([float(row["runtime_seconds"]) for row in opportunities])
    positive_runtimes = runtimes[runtimes > 0]
    bin_count = min(20, max(3, len(positive_runtimes)))
    if len(positive_runtimes) and float(np.min(positive_runtimes)) < float(np.max(positive_runtimes)):
        bins: int | np.ndarray = np.geomspace(
            float(np.min(positive_runtimes)),
            float(np.max(positive_runtimes)),
            bin_count + 1,
        )
    else:
        bins = bin_count
    axes[1].hist(positive_runtimes, bins=bins, color=METHOD_STYLES["FullSADD"][0])
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Runtime per control opportunity [s]")
    axes[1].set_ylabel("Count")
    figure.tight_layout()
    _save(figure, path)


def _objective_fidelity(rows: Sequence[RawRow], path: Path) -> None:
    families = ("ising", "qft", "random", "qpe", "ghz")
    points = [row for row in rows if row["kind"] == "case"]
    correlations = sorted((row for row in rows if row["kind"] == "correlation"), key=lambda row: float(row["control"]))
    figure, axes = plt.subplots(3, 2, figsize=(6.0, 8.2))
    scatter_axes = tuple(axes.flat[:5])
    values = [float(row[key]) for row in points for key in ("rj", "ri")]
    low, high = (min(values) - 0.1, max(values) + 0.1) if values else (-1.0, 0.2)
    size_colors = {4: "#440154", 5: "#31688e", 6: "#35b779", 8: "#fde725"}
    for axis, family in zip(scatter_axes, families, strict=True):
        selected = [row for row in points if row["family"] == family]
        for row in selected:
            size = int(row["num_ions"])
            axis.scatter(float(row["rj"]), float(row["ri"]), color=size_colors[size], label=f"n={size}")
        axis.plot((low, high), (low, high), color="gray", linestyle="--")
        axis.axhline(0.0, color="black", linewidth=0.6)
        axis.axvline(0.0, color="black", linewidth=0.6)
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(family.upper())
        axis.set_xlabel(r"$R_J$")
        axis.set_ylabel(r"$R_I$")
    correlation_axis = axes.flat[5]
    correlation_axis.plot(
        [float(row["control"]) for row in correlations],
        [float(row["pearson_r"]) for row in correlations],
        marker="o",
        color=METHOD_STYLES["FullSADD"][0],
    )
    correlation_axis.fill_between(
        [float(row["control"]) for row in correlations],
        [float(row["ci95_low"]) for row in correlations],
        [float(row["ci95_high"]) for row in correlations],
        color=METHOD_STYLES["FullSADD"][0],
        alpha=0.2,
    )
    correlation_axis.set_xscale("symlog", linthresh=0.03)
    correlation_axis.set_ylim(-1.05, 1.05)
    correlation_axis.set_xlabel(r"Control-error scale $s_{\rm ctrl}$")
    correlation_axis.set_ylabel(r"corr$(R_J,R_I)$")
    figure.tight_layout()
    _save(figure, path)
