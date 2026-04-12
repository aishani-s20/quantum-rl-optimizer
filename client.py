# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantum Openenv Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from quantum_openenv_env.models import QuantumAction, QuantumObservation


class QuantumOpenenvEnv(
    EnvClient[QuantumAction, QuantumObservation, State]
):
    """
    Client for the Quantum Openenv Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with QuantumOpenenvEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.gate_count)
        ...
        ...     result = client.step(QuantumAction(target_index=2, action_type=1))
        ...     print(result.observation.gate_count)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = QuantumOpenenvEnv.from_docker_image("quantum_openenv_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(QuantumAction(target_index=0, action_type=2))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: QuantumAction) -> Dict:
        """
        Convert QuantumAction to JSON payload for step message.
        """
        return {
            "target_index": action.target_index,
            "action_type": action.action_type,  # Added the new action type!
        }

    def _parse_result(self, payload: Dict) -> StepResult[QuantumObservation]:
        """
        Parse server response into StepResult[QuantumObservation].
        """
        obs_data = payload.get("observation", {})
        
        observation = QuantumObservation(
            circuit=obs_data.get("circuit", []),
            gate_count=obs_data.get("gate_count", 0),
            num_qubits=obs_data.get("num_qubits", 0),
            done=obs_data.get("done", False),
            reward=obs_data.get("reward", 0.0),
            prompt=obs_data.get("prompt", ""),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )