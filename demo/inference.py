"""
inference.py — Hackathon-compliant evaluation script.

This script runs an LLM agent (via OpenAI-compatible API) through
the Math Word Problem environment and logs results in the required format.

Required environment variables:
    API_BASE_URL  — LLM endpoint (e.g. https://api-inference.huggingface.co/v1)
    MODEL_NAME    — Model identifier (e.g. Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN      — Hugging Face / API key
    ENV_URL       — URL of the running environment (default: http://localhost:8000)

Log format (stdout):
    [START] {"episode": N, "task_level": "...", "problem": "..."}
    [STEP]  {"episode": N, "action": float, "reward": float, "done": bool}
    [END]   {"episode": N, "task_level": "...", "score": float}

Usage:
    python inference.py
"""

import json
import os
import re
import sys

from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "sk-placeholder")

# ── System prompt for the LLM agent ──────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise math problem solver.

When given a word problem:
1. Read it carefully.
2. Identify the unknown quantity.
3. Solve it step by step.
4. Return your answer as JSON ONLY — no extra text, no markdown.

JSON format:
{"answer": <number>, "reasoning": "<one-line explanation>"}

The "answer" must be a plain number (integer or decimal). Do NOT include units."""


# ── Helper: call the LLM ──────────────────────────────────────────────────────
def ask_llm(problem: str) -> tuple[float, str]:
    """
    Send a problem to the LLM and parse its JSON answer.

    Returns:
        (answer: float, reasoning: str)
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ],
        max_tokens=300,
        temperature=0.0,
    )
    content = response.choices[0].message.content.strip()

    # Strip accidental markdown fences
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(content)
        return float(data["answer"]), str(data.get("reasoning", ""))
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback: extract the first number in the response
        numbers = re.findall(r"-?\d+(?:\.\d+)?", content)
        answer = float(numbers[0]) if numbers else 0.0
        return answer, content


# ── Run one episode ───────────────────────────────────────────────────────────
def run_episode(task_level: str, episode_num: int) -> float:
    """
    Run a single episode: reset → observe → act → step → log.

    Args:
        task_level:   "easy", "medium", or "hard"
        episode_num:  Episode index (for log labeling)

    Returns:
        Episode score (reward from 0.0 to 1.0)
    """
    # Import here so the script can be run without the full package installed locally
    try:
        from demo.client import MathEnv
        from demo.models import MathAction
    except ModuleNotFoundError:
        from client import MathEnv
        from models import MathAction

    with MathEnv(base_url=ENV_URL).sync() as env:
        # ── Reset ────────────────────────────────────────────────────────────
        reset_result = env.reset(task_level=task_level)
        problem = reset_result.observation.problem

        print(
            f"[START] "
            + json.dumps(
                {
                    "episode": episode_num,
                    "task_level": task_level,
                    "problem": problem,
                }
            ),
            flush=True,
        )

        # ── LLM decides ──────────────────────────────────────────────────────
        answer, reasoning = ask_llm(problem)

        # ── Step ─────────────────────────────────────────────────────────────
        step_result = env.step(MathAction(answer=answer, reasoning=reasoning))
        obs = step_result.observation

        print(
            f"[STEP]  "
            + json.dumps(
                {
                    "episode": episode_num,
                    "action": answer,
                    "reasoning": reasoning,
                    "reward": step_result.reward,
                    "done": step_result.done,
                    "feedback": obs.feedback,
                }
            ),
            flush=True,
        )

        print(
            f"[END]   "
            + json.dumps(
                {
                    "episode": episode_num,
                    "task_level": task_level,
                    "score": step_result.reward,
                    "is_correct": obs.is_correct,
                }
            ),
            flush=True,
        )

        return step_result.reward


# ── Main: run all 3 task levels ───────────────────────────────────────────────
def main():
    task_levels = ["easy", "medium", "hard"]
    scores: dict[str, float] = {}

    print(
        json.dumps(
            {"type": "INFO", "message": "Starting Math Word Problem evaluation"}
        ),
        flush=True,
    )

    for i, level in enumerate(task_levels, start=1):
        try:
            score = run_episode(task_level=level, episode_num=i)
            scores[level] = score
        except Exception as exc:
            print(
                f"[ERROR] episode={i} task_level={level} error={str(exc)}",
                flush=True,
            )
            scores[level] = 0.0

    avg_score = sum(scores.values()) / len(scores)

    print(
        json.dumps(
            {
                "type": "SUMMARY",
                "scores": scores,
                "average_score": round(avg_score, 4),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()