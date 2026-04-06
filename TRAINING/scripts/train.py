#!/usr/bin/env python3
"""
Training entry point for Customer Service Bot RL agent.
Uses PPO to train an agent on the customer service environment.
"""
import sys
import os
import yaml
import argparse
from pathlib import Path

# Add parent directory to path for environment imports
sys.path.insert(0, str(Path(__file__).parent.parent / "RUNTIME" / "customer-service-bot"))


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train customer service bot agent")
    parser.add_argument("--config", type=str, default="configs/ppo_config.yaml", help="Path to training config")
    parser.add_argument("--task", type=str, default="all", choices=["easy", "medium", "hard", "all"], help="Task to train on")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.episodes:
        config["training"]["episodes"] = args.episodes

    print("=" * 60)
    print("Customer Service Bot - Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Task: {args.task}")
    print(f"Episodes: {config['training']['episodes']}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    for task in tasks:
        print(f"\n--- Training on {task} task ---")
        print(f"  Environment: CustomerServiceEnv(task_type='{task}')")
        print(f"  Algorithm: PPO")
        print(f"  Episodes: {config['training']['episodes']}")
        print(f"  Learning rate: {config['algorithm']['learning_rate']}")
        print(f"  Gamma: {config['algorithm']['gamma']}")
        print(f"  GAE lambda: {config['algorithm']['gae_lambda']}")
        print(f"  Batch size: {config['algorithm']['batch_size']}")
        print(f"  Epochs: {config['algorithm']['epochs']}")
        print(f"  Status: Ready to train (implement PPO trainer)")

    print("\n" + "=" * 60)
    print("Training configuration loaded successfully.")
    print("To implement full training, integrate with your preferred RL library.")
    print("=" * 60)


if __name__ == "__main__":
    main()
