# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Graders for the Quantum Circuit Optimization Environment.

Each grader measures a different aspect of performance matching its difficulty tier:
  - Easy:   Pure compression ratio. Any gate removal earns proportional credit.
  - Medium: Compression + bonus for using advanced identity actions (3 or 4).
  - Hard:   Weighted blend of compression and step efficiency. Harder threshold.

All scores are strictly within (0.01, 0.99) as required by the platform.
"""


def _strict(score: float) -> float:
    """Clamp to strictly (0.0, 1.0) — platform rejects exactly 0.0 or 1.0."""
    return max(0.01, min(0.99, float(score)))


def grade_easy(observation) -> float:
    """
    Easy grader: pure compression ratio.

    Score = (initial_gates - final_gates) / initial_gates
    Any reduction in gate count earns proportional credit.
    No bonus mechanics — agent just needs to find and cancel obvious pairs.
    """
    metadata = getattr(observation, 'metadata', {}) or {}
    final_count = getattr(observation, 'gate_count', 0)
    initial_count = metadata.get("initial_count", final_count)

    if initial_count == 0:
        return _strict(0.5)

    compression = (initial_count - final_count) / initial_count
    return _strict(compression)


def grade_medium(observation) -> float:
    """
    Medium grader: compression ratio + bonus for advanced identity usage.

    Score = compression_ratio + 0.15 bonus if agent used action 3 (H-X-H→Z)
            or action 4 (CNOT-SWAP→CZ) at least once during the episode.

    This rewards agents that discover algebraic identities beyond simple
    gate cancellation — a meaningfully harder skill than the easy task.
    """
    metadata = getattr(observation, 'metadata', {}) or {}
    final_count = getattr(observation, 'gate_count', 0)
    initial_count = metadata.get("initial_count", final_count)

    if initial_count == 0:
        return _strict(0.5)

    compression = (initial_count - final_count) / initial_count

    # Bonus for using advanced identity actions (tracked in metadata by environment)
    used_advanced = metadata.get("used_advanced_actions", False)
    bonus = 0.15 if used_advanced else 0.0

    return _strict(compression + bonus)


def grade_hard(observation) -> float:
    """
    Hard grader: weighted blend of compression efficiency and step efficiency.

    Score = 0.7 * compression_ratio + 0.3 * step_efficiency
    where step_efficiency = 1 - (steps_taken / max_steps)

    This penalises agents that compress the circuit but waste many steps —
    exactly the behaviour frontier models exhibit on hard tasks
    (thrashing with invalid swaps before finding cancellations).
    """
    metadata = getattr(observation, 'metadata', {}) or {}
    final_count = getattr(observation, 'gate_count', 0)
    initial_count = metadata.get("initial_count", final_count)
    steps_taken = metadata.get("step", 1)
    max_steps = 150

    if initial_count == 0:
        return _strict(0.5)

    compression = (initial_count - final_count) / initial_count
    step_efficiency = max(0.0, 1.0 - (steps_taken / max_steps))

    score = 0.7 * compression + 0.3 * step_efficiency
    return _strict(score)