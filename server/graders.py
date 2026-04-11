# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Standalone graders for the Quantum Circuit Optimization Environment.
"""

from quantum_openenv_env.models import QuantumObservation

def grade_easy(observation: QuantumObservation) -> float:
    final_count = observation.gate_count
    metadata = observation.metadata or {}
    initial_count = metadata.get("initial_count", final_count)
    if initial_count == 0:
        return 1.0
    compression_ratio = (initial_count - final_count) / initial_count
    return max(0.0, min(1.0, compression_ratio))

def grade_medium(observation: QuantumObservation) -> float:
    final_count = observation.gate_count
    metadata = observation.metadata or {}
    initial_count = metadata.get("initial_count", final_count)
    if initial_count == 0:
        return 1.0
    compression_ratio = (initial_count - final_count) / initial_count
    return max(0.0, min(1.0, compression_ratio))

def grade_hard(observation: QuantumObservation) -> float:
    final_count = observation.gate_count
    metadata = observation.metadata or {}
    initial_count = metadata.get("initial_count", final_count)
    if initial_count == 0:
        return 1.0
    compression_ratio = (initial_count - final_count) / initial_count
    return max(0.0, min(1.0, compression_ratio))