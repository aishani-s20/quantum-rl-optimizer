"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

from quantum_openenv_env.server.quantum_openenv_env_environment import GRADERS
from quantum_openenv_env.client import QuantumOpenenvEnv
from quantum_openenv_env.models import QuantumAction

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
IMAGE_NAME = os.getenv("IMAGE_NAME", "quantum_env") 


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("QUANTUM_TASK", "random")
BENCHMARK = os.getenv("QUANTUM_BENCHMARK", "quantum_optimization")
MAX_STEPS = 50 
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1  


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI agent tasked with optimizing a multi-qubit quantum circuit.
    You will be given the current circuit as a list of gates with their index, name, and target_qubits.
    
    You have 4 possible actions you can take at any index.
    Action 1: Cancel identical self-inverse gates (H, X, Y, Z, CNOT, SWAP). They must be on the same qubits and not blocked by intermediate gates sharing those qubits.
    Action 2: Swap adjacent commuting gates (gates that operate on entirely different qubits and do not overlap).
    Action 3: Replace an H-X-H sequence on the same qubit with a Z gate.
    Action 4: Replace a CNOT-SWAP sequence on the same qubits with a CZ gate.

    You MUST output ONLY a valid JSON object with exactly two keys: 'target_index' (integer) and 'action_type' (integer 1-4).
    Example: {"target_index": 2, "action_type": 1}
    Do not output markdown, backticks, or any other text.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, circuit: List[object], last_reward: float, history: List[str]) -> str:
    if circuit:
        circuit_lines = [f"Index {i}: {gate.name} on qubits {gate.target_qubits}" for i, gate in enumerate(circuit)]
        circuit_block = "\n".join(circuit_lines)
    else:
        circuit_block = "Empty circuit"

    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Current circuit:
        {circuit_block}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Send your next action as a JSON object with 'target_index' and 'action_type'.
        """
    ).strip()


def get_model_message(client: OpenAI, step: int, circuit: List[object], last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, circuit, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "hello"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "hello"


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await QuantumOpenenvEnv.from_docker_image(IMAGE_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        result = await env.reset()
        circuit = result.observation.circuit
        last_reward = 0.0
        
        # --- BULLETPROOF FIX START ---
        # 1. Track initial gate count locally so the grader never fails
        initial_gate_count = len(circuit)
        
        # 2. Infer the exact task from the circuit topology if metadata is missing
        actual_task = result.observation.metadata.get("task") if result.observation.metadata else None
        
        if not actual_task or actual_task == "random":
            num_qubits = result.observation.num_qubits
            if num_qubits <= 2:
                actual_task = "easy"
            elif num_qubits <= 4:
                actual_task = "medium"
            else:
                actual_task = "hard"
        # --- BULLETPROOF FIX END ---

        log_start(task=actual_task, env=BENCHMARK, model=MODEL_NAME)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            message = get_model_message(client, step, circuit, last_reward, history)

            try:
                clean_message = message.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean_message)
                target_index = int(parsed["target_index"])
                action_type = int(parsed.get("action_type", 1))
                error = None
            except Exception as exc:
                error = str(exc)
                target_index = 0
                action_type = 1

            result = await env.step(QuantumAction(target_index=target_index, action_type=action_type))

            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step
            circuit = result.observation.circuit
            last_reward = reward

            log_step(step=step, action=message, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {message!r} -> reward {reward:+.2f}")

            if done:
                break

        # Inject the saved initial count back into metadata for the grader
        if not result.observation.metadata:
            result.observation.metadata = {}
        result.observation.metadata["initial_count"] = initial_gate_count

        # Fetch the correct grader safely, falling back to the hard grader if the task name is missing
        grader = GRADERS.get(actual_task, GRADERS["hard"])
        score = grader(result.observation)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())