"""Test-suite guards.

The topology tool now runs a real optimization whenever it can (warm started from a
candidate mesh, or from the requirements when there is none). That takes minutes on a GPU,
so the whole suite runs with TO_AGENT_MODE=off and gets the deterministic mock. Tests that
exercise the live path patch `tools.topology._import_adapter` and re-enable the mode
themselves (see test_topology_tool.py).
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _disable_live_topology():
    previous = os.environ.get("TO_AGENT_MODE")
    os.environ["TO_AGENT_MODE"] = "off"
    yield
    if previous is None:
        os.environ.pop("TO_AGENT_MODE", None)
    else:
        os.environ["TO_AGENT_MODE"] = previous
