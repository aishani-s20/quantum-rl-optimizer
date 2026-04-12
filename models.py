# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Quantum Circuit Optimization Environment.

The quantum_openenv_env environment is an RL environment where the agent
learns to optimize quantum circuits by applying mathematical identities and cancellations.

Supports multi-qubit quantum circuits with individual gate-qubit targeting.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field
from typing import Dict, Any


class QuantumGate(BaseModel):
    """
    Represents a single quantum gate in a circuit.
    
    Each gate has a name (e.g., 'H', 'X', 'Y', 'Z', 'CNOT') and target qubits.
    For single-qubit gates, target_qubits is a list with one element (e.g., [0]).
    For two-qubit gates like CNOT, target_qubits is [control, target] (e.g., [0, 1]).
    """

    name: str = Field(..., description="Gate name (e.g., 'H', 'X', 'Y', 'Z', 'CNOT', 'CX', 'CZ', 'SWAP')")
    target_qubits: list[int] = Field(..., description="List of target qubits. For single-qubit: [q], for two-qubit: [control, target]")


class QuantumAction(Action):
    target_index: int = Field(
        ..., 
        ge=0, # Prevents negative numbers in the UI
        description="Index in the circuit to attempt the action"
    )
    action_type: int = Field(
        default=1, 
        ge=1, # Minimum action type is 1
        le=4, # Maximum action type is 4
        description="1: Cancel identical gates, 2: Swap commuting gates, 3: Replace H-X-H with Z, 4: Replace CNOT-SWAP with CZ"
    )


class QuantumObservation(Observation):
    """
    Observation from the Quantum Circuit Optimization environment.
    
    Contains the current state of the multi-qubit quantum circuit and metadata.
    """

    circuit: list[QuantumGate] = Field(
        default_factory=list, 
        description="List of QuantumGate objects representing the current circuit"
    )
    gate_count: int = Field(
        default=0, 
        description="Total number of gates in the current circuit"
    )
    num_qubits: int = Field(
        default=0,
        description="Number of qubits in the system"
    )
    done: bool = Field(
        default=False,
        description="Whether the environment has reached a terminal state"
    )
    reward: float = Field(
        default=0.0,
        description="The reward achieved on the last step"
    )
    prompt: str = Field(
        default="",
        description="Human-readable game rules and current circuit state for the UI playground."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional episode metadata (e.g., initial gate count, seed)"
    )