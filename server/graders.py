# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Graders for the Quantum Circuit Optimization Environment.

Each grader measures a different skill matching its difficulty tier:

  Easy:   Pure compression ratio. Any gate removal earns proportional credit.

  Medium: Compression + 0.15 bonus for using advanced identity actions (3 or 4).
          Rewards agents that discover algebraic identities beyond simple cancellations.

  Hard:   Strict compression ratio with a 5% minimum threshold.
          No step-efficiency bonus — only raw gate reduction counts.
          An agent that removes nothing scores the minimum (0.01).
          This ensures hard is always harder than medium to pass.

All scores clamped strictly to (0.01, 0.99) — platform rejects exactly 0.0 or 1.0.

NOTE: The class methods grade_easy / grade_medium / grade_hard in
      quantum_openenv_env_environment.py must be kept in sync with these
      standalone functions. Both are called by different parts of the platform.
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
    Score = compression + 0.15 if agent used action 3 or 4, else compression.
    Rewards discovering H-X-H=Z or CNOT-SWAP=CZ beyond simple cancellation.
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
    Hard grader: strict compression ratio only.

    No step-efficiency bonus. Only absolute gate reduction matters.

    The 5% minimum threshold means an agent must remove at least 5% of
    gates to score above the minimum. Agents that thrash with invalid
    actions and get lucky with 1-2 cancellations near the end score
    only their raw compression — no artificial inflation.

    Why not use step-efficiency here:
      A 6-qubit ~70-gate circuit is hard enough that even finding 5%
      compression is meaningful. Adding step efficiency would reward
      agents that do nothing (low step count) over agents that try but
      fail — which is backwards. Pure compression is the honest metric.
    """
    metadata = getattr(observation, 'metadata', {}) or {}
    final_count = getattr(observation, 'gate_count', 0)
    initial_count = metadata.get("initial_count", final_count)

    if initial_count == 0:
        return _strict(0.5)

    compression = (initial_count - final_count) / initial_count

    # Hard minimum: must achieve at least 5% compression to score above floor.
    # This prevents lucky 1-cancellation runs from passing the threshold.
    if compression < 0.05:
        return _strict(0.01)

    return _strict(compression)