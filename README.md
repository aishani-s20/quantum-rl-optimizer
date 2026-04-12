---
title: Quantum Circuit Optimizer
emoji: 🌌
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# 🌌 Quantum Circuit Optimization Environment

> **An advanced, physics-grounded Reinforcement Learning environment for the Meta OpenEnv Hackathon.**  
> Challenge agents to act as quantum compilers — optimizing multi-qubit circuits through mathematical identities and commutativity rules.

---

## Key Features

- **NP-Hard Problem Space:** Moves beyond static text puzzles into multi-dimensional spatial reasoning.
- **Deterministic Reproducibility (Seed Logic):** Fully supports the OpenEnv framework's episode seed. The engine guarantees the **exact same circuit** is generated for a given seed across different model runs, ensuring flawless grader reproducibility.
- **Three Differentiated Graders:** Each difficulty tier measures a genuinely different skill — pure compression on easy, identity-discovery bonus on medium, and step-efficiency weighting on hard.

---

## Motivation: The Quantum Compiler Challenge

In the real world, quantum computers suffer from rapid **decoherence**. Every quantum gate introduces noise, so shorter circuits yield higher-fidelity results. However, optimal quantum circuit compression is an **NP-Hard problem**.

While traditional frameworks like **Qiskit, Cirq, and tket** rely on hardcoded human heuristics to identify redundant gates and exploit commutativity, this environment turns that exact physics problem into a rigorous testing ground for Artificial Intelligence. It is designed to evaluate whether RL and LLM agents can independently learn and execute these compiler heuristics from scratch.

Current LLM benchmarks rely on static toy puzzles. This environment bridges the gap by requiring agents to generalize real-world quantum physics rules — such as swapping spatially separated, commuting gates to bring distant self-inverse identities together. **Memorization is impossible**; agents must dynamically reason about multi-dimensional spatial gate layouts and plan over long horizons.

---

## Environment Specifications

### Observation Space

The environment provides the agent with a complete topological view of the quantum state at every step.

| Field | Type | Description |
|---|---|---|
| `circuit` | `List[Gate]` | Current gate sequence. Each gate has a `name` (e.g. `"H"`, `"CNOT"`) and `target_qubits`. |
| `gate_count` | `int` | Current number of gates in the circuit. |
| `num_qubits` | `int` | Total number of qubits in the system. |
| `done` | `bool` | `True` if the circuit is fully optimized, dead-ended, or the step limit (150) is reached. |
| `reward` | `float` | Reward received from the previous action. |
| `metadata` | `dict` | Episode tracking data — see breakdown below. |

#### Metadata Fields

| Key | Type | Description |
|---|---|---|
| `task` | `str` | Active task name: `"easy"`, `"medium"`, or `"hard"`. |
| `initial_count` | `int` | Gate count at episode start. Used by all graders to compute compression ratio. |
| `step` | `int` | Current step number. Used by the hard grader for step-efficiency scoring. |
| `seed` | `int \| None` | RNG seed used to generate this circuit. Pass the same value to `reset()` to reproduce it exactly. |
| `used_advanced_actions` | `bool` | `True` if the agent successfully used action 3 (H-X-H→Z) or action 4 (CNOT-SWAP→CZ) this episode. Used by the medium grader bonus. |

---

### Action Space

The agent submits a JSON payload specifying where and how to modify the circuit.

| Field | Type | Description |
|---|---|---|
| `target_index` | `int` | Index of the primary gate in the circuit array to target. |
| `action_type` | `int` | Quantum physics rule to apply (1–4). See below. |

#### Available Action Types

| ID | Name | Description | Reward |
|---|---|---|---|
| `1` | **Cancel Identical Gates** | Removes self-inverse gate pairs (X·X = I, H·H = I, CNOT·CNOT = I, etc.) on the same qubits, not blocked by overlapping intermediate gates. | `+1.0` |
| `2` | **Swap Commuting Gates** | Swaps the target gate with the next adjacent gate **only if** their qubit sets do not intersect. Enables bringing distant cancellable pairs together. | `-0.05` |
| `3` | **H-X-H Identity Collapse** | Replaces a `H → X → H` sequence on the same qubit with a single `Z` gate (net: 2 gates removed). | `+2.0` |
| `4` | **Entanglement Compression** | Replaces an adjacent `CNOT → SWAP` on the same qubits with a single `CZ` gate (net: 1 gate removed). | `+1.0` |

> **Invalid actions** (out-of-bounds index, illegal non-commuting swap, pattern not present) incur a `-0.10` penalty. Circuit state remains unchanged.

---

## Tasks & Difficulty Levels

| Task | Qubits | Initial Gates | Entanglement | Key Challenge |
|---|---|---|---|---|
| `easy` | 2 | ~20 | None (single-qubit only) | Identify and cancel local self-inverse gate pairs. |
| `medium` | 4 | ~30 | Low (CNOT, SWAP) | Swap to unblock cancellations; discover H-X-H and CNOT-SWAP identities. |
| `hard` | 6 | ~70 | High (deep entanglement) | Long-horizon spatial reasoning; must compress efficiently with minimal wasted steps. |

Set `QUANTUM_TASK=random` to have the environment randomly select a difficulty tier on each `reset()`.

---

## Grader & Evaluation

Each grader measures a **different skill** matching its difficulty tier. All scores are strictly within `(0.01, 0.99)`.

| Task | Grader Formula | Full Score Requires |
|---|---|---|
| **Easy** | `score = (initial − final) / initial` | Any consistent gate removal earns proportional credit. |
| **Medium** | `score = compression + 0.15` if agent used action 3 or 4, else `score = compression` | Gate removal **and** discovering at least one algebraic identity. |
| **Hard** | `score = 0.7 × compression + 0.3 × step_efficiency` where `step_efficiency = 1 − (steps / 150)` | High compression **and** achieving it with few wasted steps. |

The hard grader directly penalises the behaviour frontier models exhibit most — thrashing through invalid swaps before finding cancellations, which exhausts the step budget without progress.

> **Why not use the theoretical minimum gate count?** Computing the absolute minimum for a randomized multi-qubit circuit is NP-Hard. Relative compression grading is the standard approach used in real quantum compiler benchmarks, and is the only approach that scales to arbitrary circuit depth.

---

## Baseline Scores

| Model | Task | Score | Result | Notes |
|---|---|---|---|---|
| Qwen 2.5 72B Instruct (Zero-Shot) | `easy` | ~0.22 | Pass | Identifies local cancellations reliably. |
| Qwen 2.5 72B Instruct (Zero-Shot) | `medium` | ~0.08 | Pass | Occasional cancellations; rarely discovers identities; no bonus awarded. |
| Qwen 2.5 72B Instruct (Zero-Shot) | `hard` | ~0.04 | Fail | Thrashes with invalid swaps; step budget exhausted before meaningful compression. |

> Success threshold: `score ≥ 0.10`. The hard task is an **unsolved benchmark** for zero-shot reasoning models. Advanced scaffolding (ReAct, Tree-of-Thought) is required for reliable performance.

---

## Setup and Usage Instructions

### 1. Prerequisites

```bash
pip install openenv-core
uv sync
```

### 2. Environment Variables

Create a `.env` file in the root directory:

```env
HF_TOKEN="your_huggingface_read_token"
API_BASE_URL="https://router.huggingface.co/v1"
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
QUANTUM_TASK="random"
IMAGE_NAME="quantum_env"
```

| Variable | Description |
|---|---|
| `HF_TOKEN` | HuggingFace API token (read access) |
| `API_BASE_URL` | Inference endpoint (HF router or custom) |
| `MODEL_NAME` | Model to run inference with |
| `QUANTUM_TASK` | Task: `easy`, `medium`, `hard`, or `random` |
| `IMAGE_NAME` | Docker image name for the environment server |

### 3. Build & Validate

```bash
docker build -t quantum_env .
openenv validate .
```

### 4. Run Inference

```bash
uv run python inference.py
```

The script runs **easy → medium → hard** sequentially, each in its own container instance, and prints a results summary table at the end. All 3 tasks are always evaluated.

### 5. Reproducing Baseline via Seed

To reproduce the exact same circuit for a given episode, pass a seed to `reset()`:

```python
# Same seed always produces the same initial circuit
result = await env.reset(seed=42)
```

The environment uses `random.Random(seed)` internally — fully isolated per instance, safe for concurrent WebSocket sessions.

---

## Project Structure

```
.
├── server/
│   ├── __init__.py
│   ├── app.py                               # FastAPI server entry point
│   ├── graders.py                           # Task-specific grader functions
│   └── quantum_openenv_env_environment.py   # Core environment + physics engine
├── __init__.py
├── client.py                                # OpenEnv WebSocket client
├── models.py                                # Typed Pydantic models
├── inference.py                             # Baseline LLM inference script (all 3 tasks)
├── openenv.yaml                             # OpenEnv spec manifest
├── Dockerfile                               # Container definition
├── pyproject.toml
└── README.md
```

---

## License

This project is released under the MIT license found in the `LICENSE` file.