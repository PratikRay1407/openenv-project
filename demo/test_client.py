import asyncio

try:
    from demo.client import MathEnv
    from demo.models import MathAction
except ModuleNotFoundError:
    from client import MathEnv
    from models import MathAction


async def main():
    async with MathEnv(base_url="http://localhost:8000") as env:
        reset_result = await env.reset(task_level="easy")
        print(reset_result.observation.problem)

        step_result = await env.step(
            MathAction(answer=7.0, reasoning="quick sanity check")
        )
        print(step_result.observation.feedback, step_result.reward, step_result.done)


asyncio.run(main())
