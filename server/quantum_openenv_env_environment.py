# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Quantum Circuit Optimization Environment Implementation.

Architecture:
- Dynamically generated circuits across 3 difficulty tiers to challenge frontier models.
- Instance-isolated PRNG (seeding) for strict reproducibility in server environments.
- Relative Compression Grading: Evaluates agents on compression ratio rather than
  an absolute theoretical minimum, mirroring real-world NP-Hard quantum optimization constraints.
- Advanced action tracking: medium grader rewards agents that discover
  algebraic identities (H-X-H=Z, CNOT-SWAP=CZ) beyond simple cancellations.

GRADER SYNC NOTE:
  The class methods grade_easy / grade_medium / grade_hard below must always
  stay in sync with the standalone functions in graders.py. Both are called
  by different parts of the platform. Any change to one must be made to both.
"""

import os
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State

from quantum_openenv_env.models import QuantumAction, QuantumGate, QuantumObservation


# ============================================================================
# Dynamic Task Configurations
# ============================================================================

class TaskConfig:
    def __init__(self, name: str, num_qubits: int, num_pairs: int, num_noise: int, use_entangling: bool):
        self.name = name
        self.num_qubits = num_qubits
        self.num_pairs = num_pairs
        self.num_noise = num_noise
        self.use_entangling = use_entangling

    def generate_circuit(self, rng: random.Random) -> list[QuantumGate]:
        single_gates = ["H", "X", "Y", "Z"]
        multi_gates = ["CNOT", "SWAP"]
        circuit = []

        for _ in range(self.num_noise):
            if self.use_entangling and self.num_qubits > 1 and rng.random() > 0.5:
                q1, q2 = rng.sample(range(self.num_qubits), 2)
                circuit.append(QuantumGate(name=rng.choice(multi_gates), target_qubits=[q1, q2]))
            else:
                q = rng.randint(0, self.num_qubits - 1)
                circuit.append(QuantumGate(name=rng.choice(single_gates), target_qubits=[q]))

        for _ in range(self.num_pairs):
            if self.use_entangling and self.num_qubits > 1 and rng.random() > 0.5:
                gate_name = rng.choice(multi_gates)
                qubits = rng.sample(range(self.num_qubits), 2)
            else:
                gate_name = rng.choice(single_gates)
                qubits = [rng.randint(0, self.num_qubits - 1)]

            gate1 = QuantumGate(name=gate_name, target_qubits=qubits)
            gate2 = QuantumGate(name=gate_name, target_qubits=qubits)

            insert_idx_1 = rng.randint(0, len(circuit))
            circuit.insert(insert_idx_1, gate1)
            insert_idx_2 = rng.randint(insert_idx_1, len(circuit))
            circuit.insert(insert_idx_2, gate2)

        return circuit


TASK_CONFIGS = {
    "easy":   TaskConfig("easy",   num_qubits=2, num_pairs=8,  num_noise=4,  use_entangling=False),
    "medium": TaskConfig("medium", num_qubits=4, num_pairs=12, num_noise=8,  use_entangling=True),
    "hard":   TaskConfig("hard",   num_qubits=6, num_pairs=25, num_noise=20, use_entangling=True),
}

TASKS = ["easy", "medium", "hard"]


# ============================================================================
# Graders (imported from graders.py)
# ============================================================================

from quantum_openenv_env.server.graders import grade_easy, grade_medium, grade_hard

GRADERS = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}


# ============================================================================
# Environment
# ============================================================================

class QuantumCircuitOptimizationEnvironment(Environment):
    """
    Quantum Circuit Optimization RL Environment.

    The agent acts as a quantum compiler, reducing circuit depth by applying
    mathematical identities and commutativity rules across 3 difficulty tiers.

    Observation:
        circuit                - Current list of QuantumGate objects
        gate_count             - Number of gates remaining
        num_qubits             - System qubit count
        done                   - Episode terminal flag
        reward                 - Last step reward
        prompt                 - Human-readable state for the web UI playground
        metadata               - task, initial_count, step, seed, used_advanced_actions

    Action types:
        1 - Cancel identical self-inverse gate pairs          (+1.0)
        2 - Swap adjacent commuting gates (different qubits)  (-0.05)
        3 - Replace H-X-H sequence with Z gate                (+2.0)
        4 - Replace CNOT-SWAP sequence with CZ gate           (+1.0)
        Invalid actions                                        (-0.1)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    SELF_INVERSE_GATES = {
        "H", "X", "Y", "Z", "CNOT", "CX", "CZ", "SWAP",
        "CCX", "TOFFOLI", "CSWAP", "FREDKIN"
    }

    def __init__(self, task: str = "random", seed: int = None):
        # Read from Docker env_var if task not explicitly set
        if task == "random":
            task = os.getenv("QUANTUM_TASK", "random")

        self.mode = task
        if self.mode != "random" and self.mode not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task: {task}. Must be 'random' or one of {list(TASK_CONFIGS.keys())}"
            )

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.current_seed = seed
        self.rng = random.Random(self.current_seed) if self.current_seed is not None else random.Random()

        self.task_name = "easy"
        self.task_config = TASK_CONFIGS["easy"]
        self._circuit: list[QuantumGate] = []
        self._initial_gate_count = 0
        self._used_advanced_actions = False

    # ============================================================================
    # OpenEnv API
    # ============================================================================

    def reset(self, seed: int = None, **kwargs) -> QuantumObservation:
        """Reset the environment to a fresh circuit for the configured task."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self._used_advanced_actions = False

        if seed is not None:
            self.current_seed = seed
            self.rng = random.Random(self.current_seed)

        if self.mode == "random":
            self.task_name = self.rng.choice(TASKS)
        else:
            self.task_name = self.mode

        self.task_config = TASK_CONFIGS[self.task_name]
        self._circuit = self.task_config.generate_circuit(self.rng)
        self._initial_gate_count = len(self._circuit)

        return QuantumObservation(
            circuit=self._circuit,
            gate_count=len(self._circuit),
            num_qubits=self.task_config.num_qubits,
            done=False,
            reward=0.0,
            prompt=self._generate_prompt(),
            metadata={
                "task": self.task_name,
                "reset_count": self._reset_count,
                "initial_count": self._initial_gate_count,
                "seed": self.current_seed,
                "used_advanced_actions": False,
            },
        )

    def step(self, action: QuantumAction, **kwargs) -> QuantumObservation:  # type: ignore[override]
        """Execute one action in the environment."""
        self._state.step_count += 1
        target_index = action.target_index
        action_type = action.action_type

        reward = -0.1
        action_result = "invalid"

        if target_index < 0 or target_index >= len(self._circuit):
            return self._build_observation(reward, "invalid_index")

        gate_at_index = self._circuit[target_index]
        active_qubits = set(gate_at_index.target_qubits)

        # ACTION 1: Cancel Identical Self-Inverse Gates
        if action_type == 1:
            next_gate_index = None
            for j in range(target_index + 1, len(self._circuit)):
                next_qubits = set(self._circuit[j].target_qubits)
                if active_qubits.intersection(next_qubits):
                    next_gate_index = j
                    break

            if (next_gate_index is not None and
                    self._circuit[next_gate_index].name == gate_at_index.name and
                    self._circuit[next_gate_index].target_qubits == gate_at_index.target_qubits and
                    gate_at_index.name in self.SELF_INVERSE_GATES):
                self._circuit.pop(next_gate_index)
                self._circuit.pop(target_index)
                reward = 1.0
                action_result = "cancelled_identical"

        # ACTION 2: Swap Commuting Gates
        elif action_type == 2:
            if target_index + 1 < len(self._circuit):
                next_gate = self._circuit[target_index + 1]
                next_qubits = set(next_gate.target_qubits)
                if not active_qubits.intersection(next_qubits):
                    self._circuit[target_index], self._circuit[target_index + 1] = (
                        self._circuit[target_index + 1],
                        self._circuit[target_index],
                    )
                    reward = -0.05
                    action_result = "swapped_commuting"

        # ACTION 3: Replace H-X-H with Z  (advanced identity)
        elif action_type == 3:
            if target_index + 2 < len(self._circuit):
                g1 = self._circuit[target_index]
                g2 = self._circuit[target_index + 1]
                g3 = self._circuit[target_index + 2]

                if (g1.name == "H" and g2.name == "X" and g3.name == "H" and
                        g1.target_qubits == g2.target_qubits == g3.target_qubits):
                    self._circuit.pop(target_index + 2)
                    self._circuit.pop(target_index + 1)
                    self._circuit[target_index] = QuantumGate(
                        name="Z", target_qubits=g1.target_qubits
                    )
                    reward = 2.0
                    action_result = "identity_hxh_to_z"
                    self._used_advanced_actions = True

        # ACTION 4: Replace CNOT-SWAP with CZ  (advanced identity)
        elif action_type == 4:
            if target_index + 1 < len(self._circuit):
                g1 = self._circuit[target_index]
                g2 = self._circuit[target_index + 1]

                if (g1.name == "CNOT" and g2.name == "SWAP" and
                        set(g1.target_qubits) == set(g2.target_qubits)):
                    self._circuit.pop(target_index + 1)
                    self._circuit[target_index] = QuantumGate(
                        name="CZ", target_qubits=g1.target_qubits
                    )
                    reward = 1.0
                    action_result = "identity_cnot_swap_to_cz"
                    self._used_advanced_actions = True

        return self._build_observation(reward, action_result)

    @property
    def state(self) -> State:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        """Return metadata shown in the HF Space web UI and consumed by platform agent."""
        return EnvironmentMetadata(
            name="Quantum Circuit Optimizer",
            description=(
                "RL environment where an agent acts as a quantum compiler, "
                "reducing circuit depth by applying gate cancellation, "
                "commutativity swaps, and algebraic identities "
                "(H·X·H = Z, CNOT·SWAP = CZ) across 3 difficulty tiers "
                "(2-qubit easy → 4-qubit medium → 6-qubit hard with deep entanglement)."
            ),
            version="0.1.0",
        )

    # ============================================================================
    # Grader methods — MUST stay in sync with graders.py standalone functions
    # ============================================================================

    @staticmethod
    def _strict(score: float) -> float:
        """Clamp to strictly (0.0, 1.0) — platform rejects exactly 0.0 or 1.0."""
        return max(0.01, min(0.99, float(score)))

    def grade_easy(self) -> float:
        """Pure compression ratio — any gate removal earns proportional credit."""
        if self._initial_gate_count == 0:
            return self._strict(0.5)
        compression = (self._initial_gate_count - len(self._circuit)) / self._initial_gate_count
        return self._strict(compression)

    def grade_medium(self) -> float:
        """Compression ratio + 0.15 bonus for using advanced identity actions."""
        if self._initial_gate_count == 0:
            return self._strict(0.5)
        compression = (self._initial_gate_count - len(self._circuit)) / self._initial_gate_count
        bonus = 0.15 if self._used_advanced_actions else 0.0
        return self._strict(compression + bonus)

    def grade_hard(self) -> float:
        """
        Strict compression ratio only. No step-efficiency bonus.

        Must achieve >= 5% compression to score above the minimum.
        Ensures hard is always genuinely harder than medium:
        - No artificial inflation from step efficiency
        - Random lucky cancellations (< 5%) score the minimum (0.01)
        - Only sustained, meaningful compression earns a real score

        SYNC: Keep identical to grade_hard() in graders.py.
        """
        if self._initial_gate_count == 0:
            return self._strict(0.5)
        compression = (self._initial_gate_count - len(self._circuit)) / self._initial_gate_count
        if compression < 0.05:
            return self._strict(0.01)
        return self._strict(compression)

    def grade(self) -> float:
        """Grade current state using the active task's grader method."""
        return {
            "easy":   self.grade_easy,
            "medium": self.grade_medium,
            "hard":   self.grade_hard,
        }[self.task_name]()

    # ============================================================================
    # Internal helpers
    # ============================================================================

    def _build_observation(self, reward: float, action_result: str) -> QuantumObservation:
        max_steps_reached = self._state.step_count >= 150
        is_done = max_steps_reached or self._is_circuit_dead_end()

        return QuantumObservation(
            circuit=self._circuit,
            gate_count=len(self._circuit),
            num_qubits=self.task_config.num_qubits,
            done=is_done,
            reward=reward,
            prompt=self._generate_prompt(),
            metadata={
                "task": self.task_name,
                "action_result": action_result,
                "step": self._state.step_count,
                "initial_count": self._initial_gate_count,
                "seed": self.current_seed,
                "used_advanced_actions": self._used_advanced_actions,
            },
        )

    def _is_circuit_dead_end(self) -> bool:
        if len(self._circuit) == 0:
            return True

        for i in range(len(self._circuit)):
            curr_gate = self._circuit[i]
            active_qubits = set(curr_gate.target_qubits)
            for j in range(i + 1, len(self._circuit)):
                next_qubits = set(self._circuit[j].target_qubits)
                if active_qubits.intersection(next_qubits):
                    next_gate = self._circuit[j]
                    if (next_gate.name == curr_gate.name and
                            next_gate.target_qubits == curr_gate.target_qubits and
                            curr_gate.name in self.SELF_INVERSE_GATES):
                        return False
                    break

        for i in range(len(self._circuit) - 1):
            if not set(self._circuit[i].target_qubits).intersection(
                    set(self._circuit[i + 1].target_qubits)):
                return False

        return True

    def _generate_prompt(self) -> str:
        """Generates a human-readable prompt for the Web UI playground."""
        prompt_text = (
            f"Quantum Circuit Optimizer ({self.task_name.upper()})\n\n"
            f"A quantum circuit on {self.task_config.num_qubits} qubits has been generated. "
            "Your goal is to compress it by finding logical reductions.\n\n"
            "ACTIONS:\n\n"
            "1: Cancel identical self-inverse gates (H, X, Y, Z, CNOT, SWAP).\n\n"
            "2: Swap adjacent commuting gates (gates not sharing qubits).\n\n"
            "3: Replace an H-X-H sequence with a Z gate.\n\n"
            "4: Replace a CNOT-SWAP sequence with a CZ gate.\n\n"
            "CURRENT CIRCUIT STATE:\n\n"
        )

        if not self._circuit:
            prompt_text += "[Empty Circuit - Optimization Complete!]"
        else:
            gate_strings = []
            for i, gate in enumerate(self._circuit):
                qubits = ",".join(str(q) for q in gate.target_qubits)
                gate_strings.append(f"[{i}]{gate.name}({qubits})")
            prompt_text += " ".join(gate_strings)

        return prompt_text