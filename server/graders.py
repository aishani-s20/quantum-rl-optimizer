# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Standalone graders for the Quantum Circuit Optimization Environment.
Scores are strictly within (0.0, 1.0) — never exactly 0.0 or 1.0.
"""


def _strict(score: float) -> float:
    """Clamp score to strictly (0.0, 1.0) as required by the platform."""
    return max(0.01, min(0.99, score))


def grade_easy(observation) -> float:
    metadata = getattr(observation, 'metadata', {}) or {}
    final_count = getattr(observation, 'gate_count', 0)
    initial_count = metadata.get("initial_count", final_count)
    if initial_count == 0:
        return _strict(0.99)
    compression = (initial_count - final_count) / initial_count
    return _strict(compression)


def grade_medium(observation) -> float:
    metadata = getattr(observation, 'metadata', {}) or {}
    final_count = getattr(observation, 'gate_count', 0)
    initial_count = metadata.get("initial_count", final_count)
    if initial_count == 0:
        return _strict(0.99)
    compression = (initial_count - final_count) / initial_count
    return _strict(compression / 0.20)


def grade_hard(observation) -> float:
    metadata = getattr(observation, 'metadata', {}) or {}
    final_count = getattr(observation, 'gate_count', 0)
    initial_count = metadata.get("initial_count", final_count)
    if initial_count == 0:
        return _strict(0.99)
    compression = (initial_count - final_count) / initial_count
    return _strict(compression / 0.35)