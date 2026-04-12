# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Graders for the Quantum Circuit Optimization Environment.

Each grader measures a different skill matching its difficulty tier:

  Easy:   Pure compression ratio. Any gate removal earns proportional credit.

  Medium: Compression + 0.15 bonus for using advanced identity actions (3 or 4).
          Rewards agents that discover algebraic identities, not just cancellations.

  Hard:   Weighted blend of compression (70%) and step efficiency (30%).
          Step efficiency ONLY contributes if compression >= 5% — this prevents
          a zero-compression episode from scoring well just from low step count.
          Penalises thrashing (many invalid actions before finding cancellations).

All scores clamped strictly to (0.01, 0.99) — platform rejects exactly 0.0 or 1.0.
"""


def _strict(score: float) -> float:
    """Clamp to strictly (0.0, 1.0) — platform rejects exactly 0.0 or 1.0."""
    return max(0.01, min(0.99, float(score)))


def grade_easy(observation) -> float:
    """
    Easy grader: pure compression ratio.
    Score = (initial_gates - final_gates) / initial_gates.
    Any reduction in gate count earns proportional credit.
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
    Medium grader: compression + bonus for advanced identity usage.
    Score = compression_ratio + 0.15 if agent used action 3 or 4, else compression_ratio.
    Rewards discovering H-X-H=Z or CNOT-SWAP=CZ identities beyond simple cancellation.
    """
    metadata = getattr(observation, 'metadata', {}) or {}
    final_count = getattr(observation, 'gate_count', 0)
    initial_count = metadata.get("initial_count", final_count)

    if initial_count == 0:
        return _strict(0.5)

    compression = (initial_count - final_count) / initial_count
    used_advanced = metadata.get("used_advanced_actions", False)
    bonus = 0.15 if used_advanced else 0.0

    return _strict(compression + bonus)


def grade_hard(observation) -> float:
    """
    Hard grader: weighted blend of compression and step efficiency.

    Score = 0.7 * compression + 0.3 * step_efficiency
    where step_efficiency = 1 - (steps_taken / max_steps)

    IMPORTANT: step_efficiency only contributes when compression >= 5%.
    Without this gate, an agent making zero progress still scores ~0.20
    from step efficiency alone — which incorrectly makes hard easier than medium.

    This directly penalises the behaviour frontier models exhibit most:
    thrashing through invalid actions, exhausting the step budget without progress.
    """
    metadata = getattr(observation, 'metadata', {}) or {}
    final_count = getattr(observation, 'gate_count', 0)
    initial_count = metadata.get("initial_count", final_count)
    steps_taken = metadata.get("step", 1)
    max_steps = 150

    if initial_count == 0:
        return _strict(0.5)

    compression = (initial_count - final_count) / initial_count

    # Step efficiency only rewards efficient agents that actually compressed the circuit.
    # An agent that does nothing should not score well just because it used few steps.
    if compression < 0.05:
        return _strict(compression)

    step_efficiency = max(0.0, 1.0 - (steps_taken / max_steps))
    score = 0.7 * compression + 0.3 * step_efficiency
    return _strict(score)