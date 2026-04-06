from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from server.environment import CustomerServiceEnv
from models import Action, ActionType, Observation, Reward, State


env: Optional[CustomerServiceEnv] = None


class ResetRequest(BaseModel):
    task_type: Optional[str] = None
    scenario_index: int = 0
    seed: Optional[int] = None


class StepRequest(BaseModel):
    message: str
    action_type: ActionType
    confidence: float = 0.5


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global env
    env = CustomerServiceEnv()
    yield


app = FastAPI(title="Customer Service Bot Environment", lifespan=lifespan)


@app.post("/reset")
async def reset(request: ResetRequest) -> Observation:
    global env
    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    if request.seed is not None:
        env = CustomerServiceEnv(seed=request.seed)

    observation = env.reset(
        task_type=request.task_type,
        scenario_index=request.scenario_index,
    )
    return observation


@app.post("/step")
async def step(request: StepRequest) -> StepResponse:
    global env
    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    action = Action(
        message=request.message,
        action_type=request.action_type,
        confidence=request.confidence,
    )

    observation, reward, done, info = env.step(action)

    return StepResponse(
        observation=observation,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state")
async def get_state() -> State:
    global env
    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    return env.state()


@app.get("/health")
async def health():
    return {"status": "ok"}
