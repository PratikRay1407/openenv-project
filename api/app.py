from fastapi import FastAPI
from env.environment import OpenEnv
from env.tasks import tasks
from pydantic import BaseModel
from env.grader import grade_easy, grade_medium, grade_hard

app = FastAPI()
env = OpenEnv()

@app.get("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: dict):
    return env.step(action)

@app.get("/state")
def state():
    return env.state()

@app.get("/tasks")
def get_tasks():
    return tasks


class GraderInput(BaseModel):
    action: dict
    level: str


@app.post("/grader")
def grader(input: GraderInput):
    action = input.action
    level = input.level

    if level == "easy":
        return {"score": grade_easy(action)}
    elif level == "medium":
        return {"score": grade_medium(action)}
    else:
        return {"score": grade_hard(action)}