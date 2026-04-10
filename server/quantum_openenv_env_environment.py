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
"""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import QuantumAction, QuantumGate, QuantumObservation
except ImportError:
    from models import QuantumAction, QuantumGate, QuantumObservation


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
    "easy": TaskConfig("easy", num_qubits=2, num_pairs=8, num_noise=4, use_entangling=False),
    "medium": TaskConfig("medium", num_qubits=4, num_pairs=12, num_noise=8, use_entangling=True),
    "hard": TaskConfig("hard", num_qubits=6, num_pairs=25, num_noise=20, use_entangling=True),
}


# ============================================================================
# Graders
# ============================================================================

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

# Exporting for inference.py and Hackathon Platform
GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}

# Explicitly define TASKS list for the platform's static analyzer
TASKS = ["easy", "medium", "hard"]


# ============================================================================
# Environment
# ============================================================================

class QuantumCircuitOptimizationEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    SELF_INVERSE_GATES = {"H", "X", "Y", "Z", "CNOT", "CX", "CZ", "SWAP", "CCX", "TOFFOLI", "CSWAP", "FREDKIN"}

    def __init__(self, task: str = "random", seed: int = None):
        self.mode = task
        if self.mode != "random" and self.mode not in TASK_CONFIGS:
            raise ValueError(f"Unknown task: {task}. Must be 'random' or one of {list(TASK_CONFIGS.keys())}")
        
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.current_seed = seed
        self.rng = random.Random(self.current_seed) if self.current_seed is not None else random.Random()
        
        self.task_name = "easy"
        self.task_config = TASK_CONFIGS["easy"]
        self._circuit = []
        self._initial_gate_count = 0

    def reset(self, seed: int = None) -> QuantumObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        
        if seed is not None:
            self.current_seed = seed
            self.rng = random.Random(self.current_seed)
        
        if self.mode == "random":
            self.task_name = self.rng.choice(list(TASK_CONFIGS.keys()))
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
            metadata={
                "task": self.task_name, 
                "reset_count": self._reset_count,
                "initial_count": self._initial_gate_count,
                "seed": self.current_seed
            },
        )

    def step(self, action: QuantumAction) -> QuantumObservation:  # type: ignore[override]
        self._state.step_count += 1
        target_index = action.target_index
        action_type = action.action_type
        
        reward = -0.1  
        action_result = "invalid"

        if target_index < 0 or target_index >= len(self._circuit):
            return self._build_observation(reward, "invalid_index")

        gate_at_index = self._circuit[target_index]
        active_qubits = set(gate_at_index.target_qubits)

        # ---------------------------------------------------------
        # ACTION 1: Cancel Identical Gates
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # ACTION 2: Swap Commuting Gates
        # ---------------------------------------------------------
        elif action_type == 2:
            if target_index + 1 < len(self._circuit):
                next_gate = self._circuit[target_index + 1]
                next_qubits = set(next_gate.target_qubits)
                
                if not active_qubits.intersection(next_qubits):
                    self._circuit[target_index], self._circuit[target_index + 1] = self._circuit[target_index + 1], self._circuit[target_index]
                    reward = -0.05 
                    action_result = "swapped_commuting"

        # ---------------------------------------------------------
        # ACTION 3: Replace H-X-H with Z
        # ---------------------------------------------------------
        elif action_type == 3:
            if target_index + 2 < len(self._circuit):
                g1 = self._circuit[target_index]
                g2 = self._circuit[target_index + 1]
                g3 = self._circuit[target_index + 2]
                
                if (g1.name == "H" and g2.name == "X" and g3.name == "H" and
                    g1.target_qubits == g2.target_qubits == g3.target_qubits):
                    
                    self._circuit.pop(target_index + 2)
                    self._circuit.pop(target_index + 1)
                    self._circuit[target_index] = QuantumGate(name="Z", target_qubits=g1.target_qubits)
                    reward = 2.0
                    action_result = "identity_hxh_to_z"

        # ---------------------------------------------------------
        # ACTION 4: Replace CNOT-SWAP with CZ
        # ---------------------------------------------------------
        elif action_type == 4:
            if target_index + 1 < len(self._circuit):
                g1 = self._circuit[target_index]
                g2 = self._circuit[target_index + 1]
                
                if (g1.name == "CNOT" and g2.name == "SWAP" and 
                    set(g1.target_qubits) == set(g2.target_qubits)):
                    
                    self._circuit.pop(target_index + 1)
                    self._circuit[target_index] = QuantumGate(name="CZ", target_qubits=g1.target_qubits)
                    reward = 1.0
                    action_result = "identity_cnot_swap_to_cz"

        return self._build_observation(reward, action_result)

    def _build_observation(self, reward: float, action_result: str) -> QuantumObservation:
        max_steps_reached = self._state.step_count >= 150
        is_done = max_steps_reached or self._is_circuit_dead_end()

        return QuantumObservation(
            circuit=self._circuit,
            gate_count=len(self._circuit),
            num_qubits=self.task_config.num_qubits,
            done=is_done,
            reward=reward,
            metadata={
                "task": self.task_name,
                "action_result": action_result,
                "step": self._state.step_count,
                "initial_count": self._initial_gate_count,
                "seed": self.current_seed
            },
        )

    def _is_circuit_dead_end(self) -> bool:
        # 1. If the circuit is fully optimized to 0, the episode is done (Victory!)
        if len(self._circuit) == 0:
            return True
            
        # 2. Check if a simple cancellation is immediately available
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
                        return False # A valid cancellation exists
                    break 
                    
        # 3. Check if ANY valid swap is possible. If they can swap, it's not a dead end.
        for i in range(len(self._circuit) - 1):
            if not set(self._circuit[i].target_qubits).intersection(set(self._circuit[i+1].target_qubits)):
                return False # A valid swap exists, agent might be building an identity
                
        # If no cancellations and no swaps are possible, it truly is a dead end.
        return True

    def grade(self) -> float:
        grader = GRADERS[self.task_name]
        return grader(QuantumObservation(
            circuit=self._circuit,
            gate_count=len(self._circuit),
            num_qubits=self.task_config.num_qubits,
            metadata={"initial_count": self._initial_gate_count}
        ))

    @property
    def state(self) -> State:
        return self._state