# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Standalone graders for the Quantum Circuit Optimization Environment.
"""

from models import QuantumObservation


def grade_easy(observation: QuantumObservation) -> float:
    """Independent grader for Easy Task."""
    final_count = observation.gate_count
    metadata = observation.metadata or {}
    initial_count = metadata.get("initial_count", final_count)
    if initial_count == 0:
        return 1.0
    compression_ratio = (initial_count - final_count) / initial_count
    return max(0.0, min(1.0, compression_ratio))


def grade_medium(observation: QuantumObservation) -> float:
    """Independent grader for Medium Task."""
    final_count = observation.gate_count
    metadata = observation.metadata or {}
    initial_count = metadata.get("initial_count", final_count)
    if initial_count == 0:
        return 1.0
    compression_ratio = (initial_count - final_count) / initial_count
    return max(0.0, min(1.0, compression_ratio))


def grade_hard(observation: QuantumObservation) -> float:
    """Independent grader for Hard Task."""
    final_count = observation.gate_count
    metadata = observation.metadata or {}
    initial_count = metadata.get("initial_count", final_count)
    if initial_count == 0:
        return 1.0
    compression_ratio = (initial_count - final_count) / initial_count
    return max(0.0, min(1.0, compression_ratio))