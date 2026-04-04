    """
    train.py — Train a lightweight LLM on the Math Word Problem environment
            using GRPO (Group Relative Policy Optimization) via TRL.

    This is the simplest possible setup:
    - Model : HuggingFaceTB/SmolLM2-360M-Instruct  (~360M params, fits on CPU/free GPU)
    - Method: GRPO (an RL algorithm well-suited for LLM reasoning tasks)
    - Env   : Your MathEnvironment (running locally on port 8000)

    How GRPO works (in plain English):
    1. For each problem, sample G candidate answers from the model.
    2. Score each answer using your environment's reward function.
    3. Train the model to increase probability of high-reward answers.
    4. Repeat until the model gets consistently good rewards.

    Install dependencies first:
        pip install trl transformers torch openenv-core

    Run the environment server in a separate terminal first:
        uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

    Then run this script:
        python train.py

    Set OPENENV_FAST_TRAIN=0 for a slower, heavier run (more epochs, more generations, full problem set).

    GPU: the default PyPI ``torch`` wheel is often CPU-only. If ``torch.cuda.is_available()`` is False,
    install a CUDA build from https://pytorch.org (pick your CUDA version), e.g.::

        uv pip install torch --index-url https://download.pytorch.org/whl/cu124
    """

    import atexit
    import json
    import os
    import re

    import torch
    from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ── Import your environment client ───────────────────────────────────────────
    try:
        from demo.client import MathEnv
        from demo.models import MathAction
    except ModuleNotFoundError:
        from client import MathEnv
        from models import MathAction

    ENV_URL = "http://localhost:8000"   # where your environment server is running
    MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"  # tiny but capable model

    # Fast demo: small data, few steps, short generations, one env connection. Unset or "0" for a fuller run.
    FAST_TRAIN = os.environ.get("OPENENV_FAST_TRAIN", "1").lower() not in ("0", "false", "no")

    # ── Build a dataset of problems ───────────────────────────────────────────────
    # We hard-code the same problems from the environment so the trainer can
    # construct prompts without needing a live server for dataset preparation.
    # The reward function (below) DOES call the live server.

    ALL_PROBLEMS = [
        # easy
        "Sarah has 12 apples. She gives away 5. How many apples does she have left?",
        "A shop sells 8 red pens and 6 blue pens. How many pens are there in total?",
        "Tom walks 3 km to school and 3 km back home every day. How many km in a day?",
        # medium
        "A train travels at 60 km/h for 2 hours, then at 80 km/h for 1 hour. Total distance?",
        "John earns $120 per day. He works 5 days and spends $200. How much is left?",
        "A rectangle is 15 cm long and 8 cm wide. What is its perimeter in cm?",
        # hard
        "A store marks up products by 40%, then gives a 15% discount. Final price of a $200 item?",
        "Three workers build a wall in 12 days. How many days for 4 workers?",
        "A sum doubles in 8 years at simple interest. What is the annual rate?",
    ]

    SYSTEM_PROMPT = (
        "You are a math solver. Return your answer as JSON only: "
        '{"answer": <number>, "reasoning": "<one line>"}. '
        "No extra text, no markdown."
    )


    def make_prompt(problem: str) -> str:
        """Format problem as a chat prompt for SmolLM2."""
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )


    def parse_answer(text: str) -> float:
        """Extract the numerical answer from the model's JSON output."""
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "answer" in data:
                return float(data["answer"])
            if type(data) in (int, float):
                return float(data)
            if isinstance(data, list) and len(data) == 1 and isinstance(
                data[0], (int, float)
            ):
                return float(data[0])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        return float(numbers[0]) if numbers else 0.0


    # ── Reward function ───────────────────────────────────────────────────────────
    # This is the bridge between your environment and the GRPO trainer.
    # It receives a list of model completions and returns a reward for each one.

    _reward_sync_env = None


    def _get_reward_env():
        """Reuse one WebSocket session for all reward calls (avoids connect overhead per sample)."""
        global _reward_sync_env
        if _reward_sync_env is None:
            client = MathEnv(base_url=ENV_URL).sync()
            client.connect()
            _reward_sync_env = client
        return _reward_sync_env


    def _close_reward_env() -> None:
        global _reward_sync_env
        if _reward_sync_env is not None:
            try:
                _reward_sync_env.close()
            except Exception:
                pass
            _reward_sync_env = None


    atexit.register(_close_reward_env)


    def reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
        """
        For each model completion, send the parsed answer to the math environment
        and return the reward. This is called by GRPOTrainer automatically.

        Args:
            completions: List of model-generated strings (one per sample in the batch)
            prompts    : Corresponding input prompts

        Returns:
            List of float rewards in [0.0, 1.0]
        """
        global _reward_sync_env
        rewards = []
        for completion, prompt in zip(completions, prompts):
            answer = parse_answer(completion)

            # Determine task level from prompt length (rough heuristic)
            # In a production setup you'd store metadata alongside each prompt
            task_level = "easy" if len(prompt) < 200 else ("medium" if len(prompt) < 350 else "hard")

            try:
                env = _get_reward_env()
                env.reset(task_level=task_level)
                step_result = env.step(
                    MathAction(answer=answer, reasoning=completion)
                )
                rewards.append(step_result.reward if step_result.reward is not None else 0.0)
            except Exception:
                _reward_sync_env = None  # reconnect on next call if server dropped the socket
                rewards.append(0.0)

        return rewards


    # ── Build dataset ─────────────────────────────────────────────────────────────

    def build_dataset(tokenizer, problems: list[str]) -> list[dict]:
        """Convert problems into the format GRPOTrainer expects."""
        dataset = []
        for problem in problems:
            prompt = make_prompt(problem)
            dataset.append({"prompt": prompt})
        return dataset


    # ── Main training loop ────────────────────────────────────────────────────────


    def _hardware_kwargs():
        """
        Pick dtype and Trainer flags so CUDA is used when available (bf16 or fp16).
        Falls back to CPU float32 when CUDA is missing or OPENENV_FORCE_CPU=1.
        """
        if os.environ.get("OPENENV_FORCE_CPU", "").lower() in ("1", "true", "yes"):
            return {
                "model_dtype": torch.float32,
                "use_cpu": True,
                "bf16": False,
                "fp16": False,
            }
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            if torch.cuda.is_bf16_supported():
                return {
                    "model_dtype": torch.bfloat16,
                    "use_cpu": False,
                    "bf16": True,
                    "fp16": False,
                }
            return {
                "model_dtype": torch.float16,
                "use_cpu": False,
                "bf16": False,
                "fp16": True,
            }
        return {
            "model_dtype": torch.float32,
            "use_cpu": True,
            "bf16": False,
            "fp16": False,
        }


    def _log_device(hw: dict) -> None:
        if hw["use_cpu"]:
            print(
                "Device: CPU (float32). For NVIDIA GPU training, install a CUDA-enabled "
                "torch wheel from https://pytorch.org — see docstring at top of train.py."
            )
            return
        name = torch.cuda.get_device_name(0)
        dtype = "bfloat16" if hw["bf16"] else "float16"
        print(f"Device: CUDA — {name} ({dtype} + Trainer bf16/fp16 flags)")


    def main():
        hw = _hardware_kwargs()
        print(f"Loading model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=hw["model_dtype"])

        problems = ALL_PROBLEMS[:3] if FAST_TRAIN else ALL_PROBLEMS
        dataset = build_dataset(tokenizer, problems)

        if FAST_TRAIN:
            # Short generations, few GRPO samples, cap steps — good for smoke tests / iteration
            config = GRPOConfig(
                output_dir="./math_env_model",
                max_steps=8,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=1,
                learning_rate=5e-6,
                num_generations=2,
                max_completion_length=48,
                logging_steps=1,
                save_steps=500,
                report_to="none",
                use_cpu=hw["use_cpu"],
                bf16=hw["bf16"],
                fp16=hw["fp16"],
            )
            eval_max_new = 48
        else:
            config = GRPOConfig(
                output_dir="./math_env_model",
                num_train_epochs=3,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=5e-6,
                num_generations=4,
                max_completion_length=150,
                logging_steps=5,
                save_steps=50,
                report_to="none",
                use_cpu=hw["use_cpu"],
                bf16=hw["bf16"],
                fp16=hw["fp16"],
            )
            eval_max_new = 100

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_fn,  # your environment IS the reward function
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,  # TRL renamed tokenizer → processing_class
        )

        mode = "FAST (OPENENV_FAST_TRAIN=1)" if FAST_TRAIN else "full (OPENENV_FAST_TRAIN=0)"
        print(f"Starting GRPO training [{mode}]...")
        _log_device(hw)
        print("(Make sure your environment server is running on port 8000)")
        try:
            trainer.train()
        finally:
            _close_reward_env()

        # Save the fine-tuned model
        trainer.save_model("./math_env_model/final")
        print("Training complete. Model saved to ./math_env_model/final")

        # Quick evaluation
        print("\n── Quick evaluation ──")
        model.eval()
        device = next(model.parameters()).device
        for problem in ALL_PROBLEMS[:3]:
            prompt = make_prompt(problem)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=eval_max_new, do_sample=False
                )
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            answer = parse_answer(response)
            print(f"Problem: {problem[:60]}...")
            print(f"Answer : {answer}")
            print()


    if __name__ == "__main__":
        main()