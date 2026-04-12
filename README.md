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
> Challenge agents to optimize multi-qubit quantum circuits through mathematical identities and commutativity rules.

---

## Key Features

- **NP-Hard Problem Space:** Moves beyond static text puzzles into multi-dimensional spatial reasoning that cannot be solved by memorization.
- **Deterministic Reproducibility:** Fully supports episode seeding via `random.Random(seed)`. The same seed always produces the same circuit, guaranteeing reproducible evaluation across model runs.
- **Three Genuinely Differentiated Graders:** Each difficulty tier measures a different skill — pure compression on easy, identity-discovery bonus on medium, and a strict 5% compression gate on hard. Grading math lives in a single file (`graders.py`) with no duplication.
- **Web UI Playground:** The `prompt` field in every observation renders a human-readable circuit view directly in the HuggingFace Space interface.

---

## Motivation: The Quantum Circuit Optimization

In the real world, quantum computers suffer from rapid **decoherence**. Every quantum gate introduces noise, so shorter circuits yield higher-fidelity results. However, optimal quantum circuit compression is an **NP-Hard problem**.

While traditional frameworks like **Qiskit, Cirq, and tket** rely on hardcoded human heuristics to identify redundant gates and exploit commutativity, this environment turns that exact physics problem into a rigorous testing ground for Artificial Intelligence. It is designed to evaluate whether RL and LLM agents can independently learn and execute these compiler heuristics from scratch.

Current LLM benchmarks rely on static toy puzzles. This environment bridges the gap by requiring agents to generalize real-world quantum physics rules — such as swapping spatially separated, commuting gates to bring distant self-inverse identities together. **Memorization is impossible**; agents must dynamically reason about multi-dimensional spatial gate layouts and plan over long horizons.

---

## Environment Specifications

### Observation Space

| Field | Type | Description |
|---|---|---|
| `circuit` | `List[Gate]` | Current gate sequence. Each gate has a `name` (e.g. `"H"`, `"CNOT"`, `"SWAP"`) and `target_qubits`. |
| `gate_count` | `int` | Current number of gates remaining in the circuit. |
| `num_qubits` | `int` | Total number of qubits in the system. |
| `done` | `bool` | `True` when fully optimized, dead-ended, or step limit (150) reached. |
| `reward` | `float` | Reward from the previous action. |
| `prompt` | `str` | Human-readable circuit state for the web UI playground. |
| `metadata` | `dict` | Episode tracking data — see breakdown below. |

#### Metadata Fields

| Key | Type | Description |
|---|---|---|
| `task` | `str` | Active task: `"easy"`, `"medium"`, or `"hard"`. |
| `initial_count` | `int` | Gate count at episode start. Used by all graders to compute compression ratio. |
| `step` | `int` | Current step number within the episode. |
| `seed` | `int \| None` | RNG seed used to generate this circuit. Pass the same value to `reset()` to reproduce it. |
| `used_advanced_actions` | `bool` | `True` if the agent used action 3 (H-X-H → Z) or action 4 (CNOT-SWAP → CZ) this episode. Used by the medium grader bonus. |

---

### Action Space

| Field | Type | Description |
|---|---|---|
| `target_index` | `int ≥ 0 & < num_qubits` | Index of the primary gate to act on. Validated: must be non-negative. |
| `action_type` | `int [1–4]` | Quantum physics rule to apply. Validated: must be between 1 and 4 inclusive. |

#### Available Action Types

| ID | Name | Description | Reward |
|---|---|---|---|
| `1` | **Cancel Identical Gates** | Removes self-inverse gate pairs (X·X=I, H·H=I, CNOT·CNOT=I) on the same qubits, not blocked by overlapping intermediate gates. | `+1.0` |
| `2` | **Swap Commuting Gates** | Swaps the target gate with the next adjacent gate **only if** their qubit sets do not intersect. Enables bringing distant cancellable pairs together. | `-0.05` |
| `3` | **H-X-H Identity Collapse** | Replaces a `H → X → H` sequence on the same qubit with a single `Z` gate (net: 2 gates removed). | `+2.0` |
| `4` | **Entanglement Compression** | Replaces an adjacent `CNOT → SWAP` on the same qubits with a single `CZ` gate (net: 1 gate removed). | `+1.0` |

> **Invalid actions** (out-of-bounds index, illegal non-commuting swap, pattern not present) incur a `-0.10` penalty. Circuit state remains unchanged.

---

## Tasks & Difficulty Levels

| Task | Qubits | Pair Gates | Noise Gates | Approx. Initial Gates | Entanglement |
|---|---|---|---|---|---|
| `easy` | 2 | 8 pairs | 4 noise | ~20 | None (single-qubit only) |
| `medium` | 4 | 12 pairs | 8 noise | ~32 | Low (CNOT, SWAP) |
| `hard` | 6 | 25 pairs | 20 noise | ~70 | High (deep entanglement) |

Set `QUANTUM_TASK=random` to have the environment randomly select a difficulty tier on each `reset()`.

---

## Grader & Evaluation

Grading math lives exclusively in `server/graders.py`. The environment class methods `grade_easy`, `grade_medium`, `grade_hard` are thin one-line delegates — no duplicated math anywhere. All scores are strictly within `(0.01, 0.99)`.

| Task | Formula | Full Score Requires |
|---|---|---|
| **Easy** | `score = (initial − final) / initial` | Any consistent gate removal earns proportional credit. |
| **Medium** | `score = compression + 0.15` if `used_advanced_actions`, else `compression` | Gate removal **and** discovering at least one algebraic identity (action 3 or 4). |
| **Hard** | `score = compression` if `compression ≥ 0.05`, else `0.01` | Must remove **at least 5%** of gates. No step-efficiency bonus — raw gate reduction only. |

**Why hard uses a strict 5% gate instead of step efficiency:**
Testing showed a step-efficiency formula inflated hard scores above medium — agents that thrashed with invalid actions but accidentally landed 1–2 cancellations near the step limit scored `~0.30` from efficiency alone with zero real compression. The 5% gate ensures hard genuinely requires sustained effort.

---

## Baseline Scores

All runs use fixed seeds so the starting circuit is identical across evaluations. Temperature 0.7 reflects real-world model behavior. The variance across 3 runs shows statistical stability of the difficulty progression.

**Run configuration:**

| Parameter | Value |
|---|---|
| Model | `llama-3.1-8b-instant` (Groq API) |
| Temperature | `0.7` |
| Max steps per episode | `15` |
| Runs per task | `3` |
| Seeds | easy=42, medium=7, hard=13 |
| Success threshold | `score ≥ 0.10` |

**Results:**

| Task | Avg Score | Min | Max | Result | Notes |
|---|---|---|---|---|---|
| `easy` | **0.167** | 0.100 | 0.200 | Pass | Model reliably finds local self-inverse cancellations on the 2-qubit circuit. |
| `medium` | **0.028** | 0.010 | 0.062 | Fail | Struggles with blocker gates; rarely discovers H-X-H or CNOT-SWAP identities needed for the bonus. |
| `hard` | **0.010** | 0.010 | 0.010 | Fail | Consistently scores the floor — cannot achieve 5% compression on the 70-gate 6-qubit circuit in 15 steps. |

**Interpretation:** The difficulty progression is correct. Easy is solvable zero-shot. Medium requires identity discovery which small models rarely attempt. Hard requires sustained multi-step planning that is beyond current zero-shot LLM capability within a limited step budget. Advanced scaffolding or RL fine-tuning is required for reliable medium/hard performance.

To reproduce these results:

```bash
# Set in your .env:
# OPENAI_API_KEY=your_groq_api_key
# API_BASE_URL=https://api.groq.com/openai/v1
# MODEL_NAME=llama-3.1-8b-instant
# IMAGE_NAME=quantum_env

uv run python inference.py
```

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
# Standard name used by the evaluation platform
OPENAI_API_KEY="your_api_key_here"
API_BASE_URL="https://api.groq.com/openai/v1"
MODEL_NAME="llama-3.1-8b-instant"

# Environment settings
QUANTUM_TASK="random"
IMAGE_NAME="quantum_env"

# Optional HF fallback
HF_TOKEN="your_hf_token"
```

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | API key — injected by the platform during evaluation; use your own for local testing |
| `API_BASE_URL` | Inference endpoint (Groq, HF router, or OpenAI compatible) |
| `MODEL_NAME` | Model identifier |
| `QUANTUM_TASK` | Task: `easy`, `medium`, `hard`, or `random` |
| `IMAGE_NAME` | Docker image name for the environment server |

### 3. Build & Validate

```bash
docker build -t quantum_env .
openenv validate .
```

### 4. Run Inference (all 3 tasks)

```bash
uv run python inference.py
```

The script runs **easy → medium → hard**, 3 episodes each with fixed seeds per task, and prints a full results summary at the end. All 3 tasks are always evaluated.

### 5. Reproducing Baseline via Seed

```python
# Same seed always produces the same initial circuit
result = await env.reset(seed=42)
```

The environment uses `random.Random(seed)` per instance — fully isolated, safe for concurrent WebSocket sessions.

---

## Project Structure

```
.
├── server/
│   ├── __init__.py
│   ├── app.py                               # FastAPI server — reads QUANTUM_TASK from env var
│   ├── graders.py                           # Single source of truth for all grading math
│   └── quantum_openenv_env_environment.py   # Core environment + thin grader delegates
├── __init__.py
├── client.py                                # OpenEnv WebSocket client
├── models.py                                # Typed Pydantic models (action: ge=0, le=4)
├── inference.py                             # Baseline LLM inference script (all 3 tasks)
├── openenv.yaml                             # OpenEnv spec manifest
├── Dockerfile                               # Container definition
├── pyproject.toml
└── README.md
```

---

## License

This project is released under the MIT license found in the `LICENSE` file.