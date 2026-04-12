"""
Baseline Inference Script
=========================
Runs a real LLM agent against all 3 tasks (easy, medium, hard) and reports
average scores across NUM_RUNS episodes per task.

Reproducibility design:
  - TASK_SEEDS pins the starting circuit per run, so the environment
    presents the same problem difficulty across different model comparisons.
  - TEMPERATURE = 0.0 (greedy decoding) makes the LLM deterministic:
    same model + same prompt = same action. This means a single run
    is fully reproducible. The model's route through the circuit will be
    identical every time.
  - NUM_RUNS = 3 averages over multiple episodes to give stable scores
    for the README baseline table.

Why scores still vary slightly across runs even at temperature=0:
  Some APIs (Groq, HF router) do not guarantee bit-exact reproducibility
  at temperature=0 due to batching and hardware differences. The variance
  should be small (<5%). For reporting, use the average.

Required environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier.
    HF_TOKEN       Your Hugging Face / API key (or GROQ_API_KEY for Groq).
    IMAGE_NAME     Docker image name (default: quantum_env).
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

from quantum_openenv_env.server.quantum_openenv_env_environment import GRADERS
from quantum_openenv_env.client import QuantumOpenenvEnv
from quantum_openenv_env.models import QuantumAction

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY") or os.getenv("API_KEY")
IMAGE_NAME   = os.getenv("IMAGE_NAME", "quantum_env")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK    = os.getenv("QUANTUM_BENCHMARK", "quantum_optimization")

MAX_STEPS  = 50
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.10

# ── Reproducibility ────────────────────────────────────────────────────────────
# TEMPERATURE = 0.0: greedy decoding makes the LLM deterministic.
# For a truly non-deterministic model (temperature > 0), increase NUM_RUNS
# and report the average — that is statistically stable even if single runs vary.
TEMPERATURE = 0.0
NUM_RUNS    = 3   # episodes per task; average is reported in summary

ALL_TASKS  = ["easy", "medium", "hard"]
TASK_SEEDS = {
    "easy":   42,
    "medium":  7,
    "hard":   13,
}
# ──────────────────────────────────────────────────────────────────────────────


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI agent tasked with optimizing a multi-qubit quantum circuit.
    You will be given the current circuit as a list of gates with their index,
    name, and target_qubits.

    You have 4 possible actions:
    Action 1: Cancel identical self-inverse gates (H, X, Y, Z, CNOT, SWAP) on
              the same qubits, not blocked by intermediate gates sharing those qubits.
    Action 2: Swap adjacent commuting gates (gates on entirely non-overlapping qubits).
    Action 3: Replace an H-X-H sequence on the same qubit with a Z gate.
    Action 4: Replace a CNOT-SWAP sequence on the same qubits with a CZ gate.

    You MUST output ONLY a valid JSON object with exactly two keys:
      "target_index" (integer) and "action_type" (integer 1-4).
    Example: {"target_index": 2, "action_type": 1}
    Do not output markdown, backticks, or any other text.
    """
).strip()


# ============================================================================
# Logging  (format required by hackathon platform output parser)
# ============================================================================

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ============================================================================
# Prompt helpers
# ============================================================================

def build_user_prompt(
    step: int, circuit: list, last_reward: float, history: List[str]
) -> str:
    circuit_block = (
        "\n".join(
            f"Index {i}: {gate.name} on qubits {gate.target_qubits}"
            for i, gate in enumerate(circuit)
        )
        if circuit else "Empty circuit"
    )
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Current circuit:
        {circuit_block}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Send your next action as a JSON object with "target_index" and "action_type".
        """
    ).strip()


def get_model_action(
    client: OpenAI,
    step: int,
    circuit: list,
    last_reward: float,
    history: List[str],
) -> str:
    user_prompt = build_user_prompt(step, circuit, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "{}"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "{}"


# ============================================================================
# Single episode
# ============================================================================

async def run_episode(
    task_name: str,
    run_number: int,
    seed: int,
    env: QuantumOpenenvEnv,
    client: OpenAI,
) -> float:
    """
    Run one episode and return the score.
    Emits [START] / [END] log lines as required by the platform.
    """
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.01
    success = False

    try:
        result = await env.reset(seed=seed)
        circuit = result.observation.circuit
        last_reward = 0.0
        initial_gate_count = len(circuit)

        actual_task = (result.observation.metadata or {}).get("task", task_name)
        if actual_task not in ALL_TASKS:
            actual_task = task_name

        log_start(task=actual_task, env=BENCHMARK, model=MODEL_NAME)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            message = get_model_action(client, step, circuit, last_reward, history)

            try:
                clean = message.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean)
                target_index = int(parsed["target_index"])
                action_type  = int(parsed.get("action_type", 1))
                error = None
            except Exception as exc:
                error = str(exc)
                target_index = 0
                action_type  = 1

            result = await env.step(
                QuantumAction(target_index=target_index, action_type=action_type)
            )
            reward = result.reward or 0.0
            done   = result.done

            rewards.append(reward)
            steps_taken = step
            circuit     = result.observation.circuit
            last_reward = reward

            log_step(step=step, action=message, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {message!r} -> reward {reward:+.2f}")

            if done:
                break

        if not result.observation.metadata:
            result.observation.metadata = {}
        result.observation.metadata["initial_count"] = initial_gate_count

        grader  = GRADERS.get(actual_task, GRADERS["hard"])
        score   = grader(result.observation)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} run {run_number} error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ============================================================================
# Main: all 3 tasks × NUM_RUNS episodes each
# ============================================================================

async def main() -> None:
    """
    Run all 3 tasks, NUM_RUNS episodes each, and report average scores.

    TEMPERATURE=0.0 makes the LLM greedy/deterministic so scores are stable.
    Average across NUM_RUNS gives a robust baseline for the README table.

    The platform requires [START] task=X ... [END] for each of easy/medium/hard.
    """
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # task → list of scores across runs
    all_scores: dict[str, List[float]] = {t: [] for t in ALL_TASKS}

    for task_name in ALL_TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"  Task : {task_name.upper()}  (seed={TASK_SEEDS[task_name]}, "
              f"runs={NUM_RUNS}, temp={TEMPERATURE})", flush=True)
        print(f"  Model: {MODEL_NAME}", flush=True)
        print(f"{'='*60}", flush=True)

        for run in range(1, NUM_RUNS + 1):
            print(f"\n  --- Run {run}/{NUM_RUNS} ---", flush=True)

            env = await QuantumOpenenvEnv.from_docker_image(
                IMAGE_NAME,
                env_vars={"QUANTUM_TASK": task_name},
            )
            try:
                score = await run_episode(
                    task_name=task_name,
                    run_number=run,
                    seed=TASK_SEEDS[task_name],
                    env=env,
                    client=client,
                )
                all_scores[task_name].append(score)
            finally:
                try:
                    await env.close()
                except Exception as e:
                    print(f"[DEBUG] env.close() error: {e}", flush=True)

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("  BASELINE RESULTS SUMMARY", flush=True)
    print(f"  Model       : {MODEL_NAME}", flush=True)
    print(f"  Temperature : {TEMPERATURE}  (0.0 = greedy/deterministic)", flush=True)
    print(f"  Runs/task   : {NUM_RUNS}", flush=True)
    print(f"  Seeds       : easy={TASK_SEEDS['easy']}  "
          f"medium={TASK_SEEDS['medium']}  hard={TASK_SEEDS['hard']}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Task':<10} {'Avg Score':>10}  {'Min':>6}  {'Max':>6}  Result", flush=True)
    print(f"  {'-'*50}", flush=True)
    for task_name in ALL_TASKS:
        scores  = all_scores[task_name]
        avg     = sum(scores) / len(scores) if scores else 0.0
        mn      = min(scores) if scores else 0.0
        mx      = max(scores) if scores else 0.0
        success = avg >= SUCCESS_SCORE_THRESHOLD
        status  = "PASS ✓" if success else "FAIL ✗"
        print(
            f"  {task_name:<10} {avg:>10.3f}  {mn:>6.3f}  {mx:>6.3f}  {status}",
            flush=True,
        )
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    asyncio.run(main())