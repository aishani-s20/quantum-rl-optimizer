# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Quantum Circuit Optimization Environment.
"""

import os
import functools

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies with 'uv sync'"
    ) from e

from quantum_openenv_env.models import QuantumAction, QuantumObservation
from quantum_openenv_env.server.quantum_openenv_env_environment import QuantumCircuitOptimizationEnvironment

# Read QUANTUM_TASK from environment variable (default: "random")
# When inference.py starts a container with env_vars={"QUANTUM_TASK": "easy"},
# this ensures the environment is instantiated with the correct task.
_task = os.getenv("QUANTUM_TASK", "random")

# Create a factory function (not a bare class) so we can pass task= argument
def _env_factory() -> QuantumCircuitOptimizationEnvironment:
    return QuantumCircuitOptimizationEnvironment(task=_task)

app = create_app(
    _env_factory,
    QuantumAction,
    QuantumObservation,
    env_name="quantum_openenv_env",
    max_concurrent_envs=100,
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()