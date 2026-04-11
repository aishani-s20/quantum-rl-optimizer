# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Standalone graders for the Quantum Circuit Optimization Environment.
"""

# No package imports — fully self-contained so platform can import without installing

def grade_easy(observation) -> float:
    metadata = getattr(observation, 'metadata', {}) or {}
    final_count = getattr(observation, 'gate_count', 0)
    initial_count = metadata.get("initial_count", final_count)
    if initial_count == 0:
        return 1.0
    return max(0.0, min(1.0, (initial_count - final_count) / initial_count))

def grade_medium(observation) -> float:
    metadata = getattr(observation, 'metadata', {}) or {}
    final_count = getattr(observation, 'gate_count', 0)
    initial_count = metadata.get("initial_count", final_count)
    if initial_count == 0:
        return 1.0
    compression = (initial_count - final_count) / initial_count
    return max(0.0, min(1.0, compression / 0.20))

def grade_hard(observation) -> float:
    metadata = getattr(observation, 'metadata', {}) or {}
    final_count = getattr(observation, 'gate_count', 0)
    initial_count = metadata.get("initial_count", final_count)
    if initial_count == 0:
        return 1.0
    compression = (initial_count - final_count) / initial_count
    return max(0.0, min(1.0, compression / 0.35))