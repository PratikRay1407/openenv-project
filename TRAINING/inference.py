from __future__ import annotations

import os
import sys
import json
import requests
from typing import Optional

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
API_KEY = os.getenv("HF_TOKEN", os.getenv("OPENAI_API_KEY", "dummy"))


def run_episode(task_type: str, scenario_index: int = 0) -> dict:
    print(f"[START] Episode: task={task_type}, scenario={scenario_index}")

    reset_payload = {
        "task_type": task_type,
        "scenario_index": scenario_index,
        "seed": 42,
    }

    resp = requests.post(f"{API_BASE_URL}/reset", json=reset_payload)
    if resp.status_code != 200:
        print(f"[END] Error resetting environment: {resp.text}")
        return {"task": task_type, "score": 0.0, "error": resp.text}

    observation = resp.json()
    print(f"  Initial observation: {observation['customer_message'][:100]}...")

    total_reward = 0.0
    step_count = 0
    done = False

    while not done:
        step_count += 1
        action = generate_action(observation, task_type)

        step_payload = {
            "message": action["message"],
            "action_type": action["action_type"],
            "confidence": action["confidence"],
        }

        print(f"[STEP] Step {step_count}: action_type={action['action_type']}")
        print(f"  Agent: {action['message'][:100]}...")

        resp = requests.post(f"{API_BASE_URL}/step", json=step_payload)
        if resp.status_code != 200:
            print(f"[END] Error stepping: {resp.text}")
            return {"task": task_type, "score": 0.0, "error": resp.text}

        result = resp.json()
        observation = result["observation"]
        reward = result["reward"]
        done = result["done"]
        info = result["info"]

        total_reward += reward["value"]

        print(f"  Customer: {observation['customer_message'][:100]}...")
        print(f"  Reward: {reward['value']:.3f} | Done: {done}")
        print(f"  Feedback: {reward['feedback']}")

    avg_reward = total_reward / max(1, step_count)
    final_score = info.get("task_result", {}).get("score", avg_reward)

    print(f"[END] Episode complete: steps={step_count}, total_reward={total_reward:.3f}, avg_reward={avg_reward:.3f}, final_score={final_score:.3f}")

    return {
        "task": task_type,
        "score": final_score,
        "steps": step_count,
        "total_reward": total_reward,
        "avg_reward": avg_reward,
    }


def generate_action(observation: dict, task_type: str) -> dict:
    system_prompt = build_system_prompt(observation, task_type)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": observation.get("customer_message", "")},
    ]

    history = observation.get("conversation_history", [])
    for i in range(0, len(history) - 1, 2):
        if i + 1 < len(history):
            messages.insert(1, {"role": "user", "content": history[i]})
            messages.insert(2, {"role": "assistant", "content": history[i + 1]})

    try:
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 300,
        }

        resp = requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=30,
        )

        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        else:
            content = get_rule_based_response(observation, task_type)
    except Exception:
        content = get_rule_based_response(observation, task_type)

    action_type = infer_action_type(content)

    return {
        "message": content,
        "action_type": action_type,
        "confidence": 0.7,
    }


def build_system_prompt(observation: dict, task_type: str) -> str:
    scenario = observation.get("scenario_description", "")

    prompts = {
        "easy": (
            "You are a friendly and helpful customer service representative for TechStore. "
            "Answer customer questions accurately and politely. Be concise and professional. "
            "Only provide information you are certain about."
        ),
        "medium": (
            "You are an experienced customer service representative for TechStore. "
            "A customer has a complaint. Show empathy, acknowledge their problem, "
            "identify the issue, and propose a concrete solution."
        ),
        "hard": (
            "You are a senior customer service representative for TechStore. "
            "A customer is very upset. Priorities: 1) De-escalate, 2) Gather information, "
            "3) Provide resolution path, 4) Escalate to supervisor when appropriate. "
            "Stay calm and professional."
        ),
    }

    base = prompts.get(task_type, prompts["easy"])
    if scenario:
        base += f"\n\nScenario: {scenario}"
    return base


def infer_action_type(message: str) -> str:
    msg = message.lower()
    if any(w in msg for w in ["escalate", "supervisor", "manager", "transfer"]):
        return "escalate"
    elif any(w in msg for w in ["thank you for contacting", "is there anything else", "case closed"]):
        return "close"
    elif "?" in message:
        return "ask_clarify"
    elif any(w in msg for w in ["i understand", "i see", "i hear you"]):
        return "acknowledge"
    return "answer"


def get_rule_based_response(observation: dict, task_type: str) -> str:
    msg = observation.get("customer_message", "").lower()

    if task_type == "easy":
        if any(w in msg for w in ["shipping", "delivery", "ship"]):
            return "We offer Standard (5-7 business days, $5.99), Express (2-3 business days, $12.99), and Overnight (next business day, $24.99) shipping. Free standard shipping on orders over $50!"
        elif any(w in msg for w in ["return", "refund"]):
            return "You can return items within 30 days of delivery in original condition. We provide free return shipping labels and process refunds within 5-7 business days."
        elif any(w in msg for w in ["payment", "pay", "credit"]):
            return "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay. For orders over $100, we also offer Klarna installment payments."
        elif any(w in msg for w in ["track", "tracking"]):
            return "You'll receive a tracking number via email within 24 hours of shipping. Use it on our website or the carrier's site to track your package."
        elif any(w in msg for w in ["warranty", "guarantee"]):
            return "All products include a 1-year manufacturer warranty. We also offer a 2-year extended warranty for $29.99."
        else:
            return "Thank you for your question! I'd be happy to help. Could you provide more details so I can assist you better?"

    elif task_type == "medium":
        if any(w in msg for w in ["defective", "broken", "doesn't work"]):
            return "I'm very sorry to hear about the defective product. I understand how frustrating this must be. I can immediately arrange a replacement with expedited shipping at no cost, or offer a full refund. I'll also provide a prepaid return shipping label. Which option would you prefer?"
        elif any(w in msg for w in ["late", "delayed", "still haven't"]):
            return "I sincerely apologize for the delay. I understand this is frustrating, especially for a time-sensitive order. Let me check the tracking status right away and provide you with an updated delivery estimate. I'll also refund your shipping cost for the inconvenience."
        elif any(w in msg for w in ["charged", "billing", "double"]):
            return "I understand your concern about the billing issue. Let me look into this right away. If a duplicate charge is confirmed, I'll initiate an immediate refund and provide you with a reference number for tracking."
        else:
            return "I understand your frustration and I'm here to help. Could you provide your order number so I can look into this issue for you?"

    else:
        if any(w in msg for w in ["manager", "supervisor"]):
            return "I completely understand your frustration and I sincerely apologize for this experience. I will escalate this to my supervisor right away who will be able to provide additional assistance. In the meantime, I want to assure you that we take this very seriously and will make it right."
        else:
            return "I hear your frustration and I genuinely apologize for this experience. Let me work on resolving this for you immediately. Your satisfaction is our top priority and I want to make sure we address this properly."


def main():
    print("=" * 60)
    print("Customer Service Bot - Baseline Inference")
    print("=" * 60)

    tasks = ["easy", "medium", "hard"]
    results = []

    for task in tasks:
        result = run_episode(task, scenario_index=0)
        results.append(result)
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['task']}: score={r['score']:.3f}, steps={r['steps']}, avg_reward={r['avg_reward']:.3f}")

    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\n  Average Score: {avg_score:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
