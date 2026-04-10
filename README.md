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

> **An advanced, physics-grounded Reinforcement Learning environment for the Meta OpenEnv Hackathon.** > Challenge agents to act as quantum compilers — optimizing multi-qubit circuits through mathematical identities and commutativity rules.

---

## 🏆 Key Features for Judges
* **NP-Hard Problem Space:** Moves beyond static text puzzles into multi-dimensional spatial reasoning.
* **Deterministic Reproducibility (Seed Logic):** Fully supports the OpenEnv framework's episode seed. The engine guarantees the **exact same complex circuit** is generated for a given episode across different model runs, ensuring flawless grader reproducibility.
* **Relative Compression Grader:** Dynamically scores agents based on mathematical compression ratios, adapting perfectly to any circuit depth.

---

## 🚀 Motivation: The Quantum Compiler Challenge

In the real world, quantum computers suffer from rapid **decoherence**. Every quantum gate introduces noise, so shorter circuits yield higher-fidelity results. However, optimal quantum circuit compression is an NP-Hard problem.

Current LLM benchmarks often rely on static toy puzzles. This environment bridges the gap by requiring agents to apply real-world quantum physics rules—such as swapping spatially separated, commuting gates to bring distant self-inverse identities together. This precludes simple memorization; agents must dynamically reason about multi-dimensional spatial gate layouts and plan over long horizons.

---

## 🛠️ Environment Specifications

### 👁️ Observation Space

The environment provides the agent with a complete topological view of the quantum state at every step.

| Field | Type | Description |
|---|---|---|
| `circuit` | `List[Gate]` | Current sequence of gates. Each gate includes a `name` and `target_qubits`. |
| `gate_count` | `int` | Current number of gates in the circuit. |
| `num_qubits` | `int` | Total number of qubits in the system. |
| `done` | `bool` | `True` if the circuit is fully optimized, dead-ended, or the step limit is reached. |
| `reward` | `float` | Reward received from the previous action. |
| `metadata` | `dict` | Instance-specific tracking data: `task`, `initial_count`, `seed`. |

---

### 🎮 Action Space

The agent submits a JSON payload specifying where and how to modify the circuit.

| Field | Type | Description |
|---|---|---|
| `target_index` | `int` | The index of the primary gate in the circuit array to target. |
| `action_type` | `int` | The specific quantum physics rule to apply (1–4). See below. |

#### Available Action Types

| ID | Name | Description | Reward |
|---|---|---|---|
| `1` | **Cancel Identical Gates** | Removes self-inverse gate pairs (e.g., X·X = I) targeting the same qubits, provided they are not blocked by overlapping intermediate gates. | `+1.0` |
| `2` | **Swap Commuting Gates** | Swaps the target gate with the next adjacent gate **only if** their target qubits do not intersect. This enables agents to bring distant cancellable gates together. | `-0.05` |
| `3` | **Identity Collapse (H-X-H)** | Replaces a 3-gate sequence `H → X → H` on the same qubit with a single `Z` gate. | `+2.0` |
| `4` | **Entanglement Compression** | Replaces an adjacent `CNOT → SWAP` sequence sharing exact qubits with a single `CZ` gate. | `+1.0` |

> **Note:** Invalid actions (e.g., out-of-bounds index, illegal non-commuting swaps) incur a `-0.10` penalty to discourage hallucination, and the circuit state remains unchanged.

---

## 📊 Tasks & Difficulty Levels

The environment natively supports dynamic scaling of qubits and circuit depth. By setting `QUANTUM_TASK=random`, the environment dynamically generates a fresh, randomized circuit topology from a pool of valid gate pairs and noise injections.

| Task | Qubits | Initial Gates | Entanglement | Expected Difficulty |
|---|---|---|---|---|
| `easy` | 2 | ~20 | None (Single Qubit) | **Low:** Agents can easily spot local cancellations. |
| `medium` | 4 | ~30 | Low (CNOT, SWAP) | **Moderate:** Requires basic spatial swapping to clear blocker gates. |
| `hard` | 6 | ~70 | High (Deep Entanglement) | **Extreme:** Demands rigorous long-horizon spatial reasoning across many qubits. |

---

## 🏆 Grader & Evaluation

Because calculating the absolute theoretical minimum length of a randomized multi-qubit circuit is NP-Hard, the environment utilizes a **Relative Compression Grader**:

$$\text{Score} = \max\left(0.0, \min\left(1.0, \frac{\text{Initial Count} - \text{Final Count}}{\text{Initial Count}}\right)\right)$$

- A score of **1.0** indicates the agent perfectly compressed the circuit down to 0 gates.
- The **success threshold** is `0.10` — meaning a 10% reduction in overall circuit depth is considered a passing score for a given episode.

---

## 📈 Baseline Scores

This environment is designed to serve as a rigorous boundary test for frontier reasoning models. All baseline evaluations are fully reproducible using the environment's deterministic seed logic.

| Model | Task | Result | Notes |
|---|---|---|---|
| Qwen 2.5 72B Instruct (Zero-Shot) | `easy` | 🟢 **Passing Baseline** | Successfully identifies and executes local cancellations (Score: ~0.15–0.30). |
| Qwen 2.5 72B Instruct (Zero-Shot) | `medium` | ⚠️ **Borderline** | Attempts basic spatial swapping but frequently gets trapped by blocking gates. Usually falls just short of the 0.10 success threshold (Score: ~0.00–0.08). |
| Qwen 2.5 72B Instruct (Zero-Shot) | `hard` | 🛡️ **Benchmark Limit** | Provides a highly complex layout that tests the absolute limits of current LLMs, establishing a rigorous 0.0 baseline. **(100% reproducible via episode seeds)**. |

> **Conclusion:** This environment successfully establishes an unsolved benchmark for testing algorithmic spatial planning, proving that advanced scaffolding (e.g., Tree-of-Thought or ReAct loops) is required for deep quantum compilation.
> 
> 🎯 **Judge's Note on Reproducibility:** You can reliably reproduce these exact baseline constraints. The environment fully supports OpenEnv episode seeding, guaranteeing the exact same initial circuit generation for any given seed across different runs.

---

## 💻 Setup and Usage Instructions

### 1. Prerequisites

Ensure you have **Docker** and **uv** installed, then install the OpenEnv core dependencies:

```bash
uv pip install openenv-core
uv sync
```

### 2. Environment Variables

Create a ```.env``` file in the root directory:

```bash
HF_TOKEN="your_huggingface_read_token"
API_BASE_URL="[https://router.huggingface.co/v1](https://router.huggingface.co/v1)"
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
QUANTUM_TASK="random"
```

| Variable | Description |
|---|---|
| `HF_TOKEN` | Your HuggingFace API token (read access) |
| `API_BASE_URL` | Inference endpoint (HF router or custom) |
| `MODEL_NAME` | Model to run inference with |
| `QUANTUM_TASK` | Task name: `easy`, `medium`, `hard`, or `random` |


### 3. Build & Validate
 
```bash
docker build -t quantum_env .
openenv validate .
```
 
### 4. Run Inference
 
```bash
uv run python inference.py
```
 
The inference script handles API errors gracefully and automatically parses JSON outputs into the strict Action Space schema.


### 5. Reproducing via Seed (For Judges)

To test the deterministic generation and replicate our baseline scores, you can pass a specific seed to the environment during the reset phase in your client script.

Simply modify the reset call in ```inference.py```:

```bash
# Pass any integer seed to guarantee the exact same initial circuit topography
result = await env.reset(seed=42)
```

---

## 📁 Project Structure

```
.
├── server/
│   ├── app.py                            # FastAPI WebSocket Server
│   └── quantum_openenv_env_environment.py # Core Physics Engine & Randomizer
├── client.py                             # EnvClient Translator
├── models.py                             # Strict Pydantic Data Models
├── inference.py                          # Baseline LLM Agent Script
├── openenv.yaml                          # OpenEnv spec manifest
├── Dockerfile                            # Container definition
└── README.md
```

---


## 📄 License
 
This project is released under the MIT license found in the `LICENSE` file.

---