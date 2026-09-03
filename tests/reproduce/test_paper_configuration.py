# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Regression checks for the manuscript's frozen reproduction settings."""

from __future__ import annotations

import tomllib

import matplotlib.pyplot as plt
import pytest

from reproduce.paper import plots
from reproduce.paper.run import DEFAULT_CONFIG


def test_manuscript_panel_order_and_noise_settings() -> None:
    with DEFAULT_CONFIG.open("rb") as handle:
        config = tomllib.load(handle)

    simulation = config["simulation"]
    sweeps = config["sweeps"]
    assert plots._ordered_panel_detunings({0.1, 1.0, 10.0}) == (10.0, 1.0, 0.1)
    assert simulation["dephasing_correlation_length_sites"] == float("inf")
    assert sweeps["profile_heating"] == [0.0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]
    assert sweeps["profile_detuning"] == pytest.approx(10.0)
    assert sweeps["spatial_correlation_sites"] == [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]


def test_heating_axis_can_start_at_zero_on_a_symlog_scale() -> None:
    figure, axis = plt.subplots()
    axis.plot([0.0, 0.1], [0.0, 1.0])
    plots._set_heating_axis(axis)

    assert axis.get_xlim()[0] == pytest.approx(0.0)
    plt.close(figure)
