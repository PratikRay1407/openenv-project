"""
Math Word Problem Environment.

A real-world RL environment where an AI agent solves math word problems
across three difficulty levels. Each episode = one problem to solve.

Reward function gives partial credit:
  - 1.0  → exact answer
  - 0.8  → within 1% of correct answer
  - 0.4  → within 10% of correct answer
  - 0.0  → more than 10% off

Hackathon requirements met:
  ✓ Real-world task (not a toy/game)
  ✓ 3 task levels: easy, medium, hard
  ✓ Rewards are in range [0.0, 1.0]
  ✓ Partial progress signals
  ✓ full step() / reset() / state() API
"""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import MathAction, MathObservation
except ImportError:
    from server.models import MathAction, MathObservation


# ── Problem bank ──────────────────────────────────────────────────────────────
# Each entry: {"problem": str, "answer": float}
# Easy   → single arithmetic operation
# Medium → two operations or multi-step reasoning
# Hard   → percentages, ratios, rates, or algebra
# ─────────────────────────────────────────────────────────────────────────────

TASK_BANK = {
    "easy": [
        {
            "problem": "Sarah has 12 apples. She gives away 5. How many apples does she have left?",
            "answer": 7.0,
        },
        {
            "problem": "A shop sells 8 red pens and 6 blue pens. How many pens are there in total?",
            "answer": 14.0,
        },
        {
            "problem": "Tom walks 3 km to school and 3 km back home every day. How many km does he walk in a day?",
            "answer": 6.0,
        },
        {
            "problem": "There are 24 students in a class. 9 are absent today. How many students are present?",
            "answer": 15.0,
        },
        {
            "problem": "A box has 5 rows of chocolates with 6 in each row. How many chocolates are there in total?",
            "answer": 30.0,
        },
        {
            "problem": "A car travels 280 km on 40 liters of petrol. How many km per liter does it get?",
            "answer": 7.0,
        },
        {
            "problem": "A farmer has 48 eggs and packs them into boxes of 6. How many boxes does he fill?",
            "answer": 8.0,
        },
    ],
    "medium": [
        {
            "problem": (
                "A train travels at 60 km/h for 2 hours, then at 80 km/h for 1 hour. "
                "What is the total distance traveled in km?"
            ),
            "answer": 200.0,
        },
        {
            "problem": (
                "John earns $120 per day. He works for 5 days and then spends $200 on groceries. "
                "How many dollars does he have left?"
            ),
            "answer": 400.0,
        },
        {
            "problem": (
                "A rectangle has a length of 15 cm and a width of 8 cm. "
                "What is its perimeter in cm?"
            ),
            "answer": 46.0,
        },
        {
            "problem": (
                "A water tank can hold 500 liters. It is currently 40% full. "
                "How many more liters are needed to completely fill it?"
            ),
            "answer": 300.0,
        },
        {
            "problem": (
                "Maria reads 25 pages each day. She wants to finish a 325-page book. "
                "How many days will it take her?"
            ),
            "answer": 13.0,
        },
        {
            "problem": (
                "A recipe uses 3 cups of flour to make 12 cookies. "
                "How many cups of flour are needed to make 60 cookies?"
            ),
            "answer": 15.0,
        },
    ],
    "hard": [
        {
            "problem": (
                "A store marks up its products by 40%, then offers a 15% discount. "
                "What is the final price in dollars of an item that originally costs $200?"
            ),
            "answer": 238.0,
        },
        {
            "problem": (
                "Three workers can build a wall in 12 days working together. "
                "How many days will it take 4 workers to build the same wall?"
            ),
            "answer": 9.0,
        },
        {
            "problem": (
                "A sum of money doubles in 8 years at simple interest. "
                "What is the annual interest rate as a percentage?"
            ),
            "answer": 12.5,
        },
        {
            "problem": (
                "A mixture of 40 liters is 25% alcohol. "
                "How many liters of pure alcohol must be added to make it 40% alcohol?"
            ),
            "answer": 10.0,
        },
        {
            "problem": (
                "Train A leaves a station at 9:00 AM traveling at 60 km/h. "
                "Train B leaves the same station at 10:00 AM in the same direction at 90 km/h. "
                "How many km from the station do they meet?"
            ),
            "answer": 180.0,
        },
        {
            "problem": (
                "A shopkeeper buys 100 kg of rice at $0.80/kg and 50 kg at $0.90/kg. "
                "He mixes them and sells at $1.00/kg. What is his total profit in dollars?"
            ),
            "answer": 75.0,
        },
    ],
}


class MathEnvironment(Environment):
    """
    Math Word Problem environment for RL training.

    Each episode presents one problem. The agent calls step() once with
    a numerical answer. The reward reflects answer accuracy.

    Example:
        >>> env = MathEnvironment()
        >>> obs = env.reset()           # get a problem
        >>> print(obs.problem)
        >>>
        >>> obs = env.step(MathAction(answer=7.0, reasoning="12 - 5 = 7"))
        >>> print(obs.reward)           # 1.0 if correct
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task: dict = {}
        self._task_level: str = "easy"

    def reset(self, task_level: str = "easy") -> MathObservation:
        """
        Start a new episode. Pick a random problem for the given difficulty.

        Args:
            task_level: "easy", "medium", or "hard"

        Returns:
            MathObservation with the problem to solve (no answer revealed yet)
        """
        if task_level not in TASK_BANK:
            task_level = "easy"

        self._task_level = task_level
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task = random.choice(TASK_BANK[task_level])

        print(f"Reset is called")
        print(f"Resetting environment with task level: {task_level}")
        return MathObservation(
            problem=self._current_task["problem"],
            task_level=self._task_level,
            done=False,
            reward=0.0,
        )

    def step(self, action: MathAction) -> MathObservation:  # type: ignore[override]
        """
        Evaluate the agent's answer.

        Partial reward signal:
          - 1.0 if exact match
          - 0.8 if within 1% error
          - 0.4 if within 10% error
          - 0.0 otherwise

        Args:
            action: MathAction with the agent's numerical answer

        Returns:
            MathObservation with reward, feedback, and correct answer
        """
        self._state.step_count += 1
        correct = self._current_task["answer"]
        submitted = action.answer

        # Avoid division by zero for problems where answer = 0
        denom = abs(correct) if abs(correct) > 1e-9 else 1.0
        relative_error = abs(submitted - correct) / denom

        if relative_error == 0:
            reward = 1.0
            is_correct = True
            feedback = f"Correct! The answer is {correct}."
        elif relative_error <= 0.01:
            reward = 0.8
            is_correct = False
            feedback = (
                f"Very close! Correct answer is {correct}, you submitted {submitted}."
            )
        elif relative_error <= 0.10:
            reward = 0.4
            is_correct = False
            feedback = (
                f"Partially correct. The answer is {correct}, you submitted {submitted}."
            )
        else:
            reward = 0.0
            is_correct = False
            feedback = (
                f"Incorrect. The correct answer is {correct}, you submitted {submitted}."
            )

        print(f"Step is called")
        print(f"Step result: {reward}, {is_correct}, {feedback}")
        return MathObservation(
            problem=self._current_task["problem"],
            task_level=self._task_level,
            correct_answer=correct,
            is_correct=is_correct,
            feedback=feedback,
            done=True,  # one step per episode
            reward=reward,
            metadata={
                "submitted_answer": submitted,
                "correct_answer": correct,
                "relative_error": relative_error,
                "step": self._state.step_count,
            },
        )

    @property
    def state(self) -> State:
        print(f"State is called")
        print(f"State: {self._state}")
        return self._state