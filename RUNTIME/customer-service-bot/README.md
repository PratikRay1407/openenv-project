# Customer Service Bot - OpenEnv Environment

A real-world customer service training environment where AI agents learn to handle customer inquiries professionally across three difficulty levels.

## Overview

This environment simulates realistic customer service interactions for training AI agents. Agents must respond to customer messages with appropriate empathy, accuracy, and professionalism. The environment provides dense reward signals, multi-dimensional grading, and realistic customer behavior simulation.

## Tasks

### Easy: FAQ Response
- **Scenario**: Customers ask common questions about shipping, returns, payments, and policies
- **Objective**: Provide accurate, polite, concise answers
- **Grading**: Correctness (40%), Professionalism (20%), Conciseness (20%), No hallucination (20%)
- **Max turns**: 5

### Medium: Complaint Resolution
- **Scenario**: Customers have problems (defective products, late deliveries, billing errors)
- **Objective**: Acknowledge issue, show empathy, propose solution, follow up
- **Grading**: Empathy (25%), Problem identification (25%), Solution appropriateness (25%), Professionalism (25%)
- **Max turns**: 7

### Hard: Multi-Turn Escalation
- **Scenario**: Complex issues requiring de-escalation and potential supervisor handoff
- **Objective**: Navigate conversation, de-escalate anger, provide resolution or proper handoff
- **Grading**: De-escalation (20%), Information gathering (20%), Resolution path (20%), Satisfaction trajectory (20%), Protocol compliance (20%)
- **Max turns**: 10

## Observation Space

```python
class Observation(BaseModel):
    customer_message: str        # The customer's latest message
    sentiment: float             # Current sentiment (-1.0 to 1.0)
    intent: str                  # Detected intent category
    urgency: int                 # Urgency level (1-5)
    conversation_history: list   # Full conversation transcript
    turn_count: int              # Current turn number
    task_type: str               # Task difficulty (easy/medium/hard)
    scenario_description: str    # Brief scenario context
```

## Action Space

```python
class Action(BaseModel):
    message: str                 # Response message to customer
    action_type: ActionType      # answer | acknowledge | ask_clarify | escalate | close
    confidence: float            # Agent confidence (0.0-1.0)
```

## Reward Design

Rewards are dense and provided at every step:
- **Progress reward** (40%): Based on task-specific grading rubric
- **Sentiment delta** (30%): Improvement in customer mood
- **Action bonus** (variable): Appropriate action type selection
- **Verbosity penalty**: Penalizes overly long responses (>500 chars)

Episode bonuses:
- +0.2 for successful resolution
- Customer mood trajectory affects final scoring

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python tests/test_environment.py

# Start server
python start_server.py

# Or with uvicorn directly
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Reset environment
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_type": "easy", "scenario_index": 0, "seed": 42}'

# Take a step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello! How can I help?", "action_type": "answer", "confidence": 0.8}'

# Get current state
curl http://localhost:8000/state
```

### Baseline Inference

```bash
# From project root
python inference.py
```

## Docker

```bash
# Build
docker build -t customer-service-bot .

# Run
docker run -p 8000:8000 customer-service-bot
```

## Project Structure

```
customer-service-bot/
├── server/
│   ├── app.py              # FastAPI server
│   └── environment.py      # OpenEnv interface
├── src/
│   ├── tasks/              # Task implementations
│   ├── graders/            # Scoring logic
│   ├── customers/          # Customer simulation
│   ├── knowledge_base/     # Company policies and FAQs
│   └── utils/              # Scoring utilities
├── models.py               # Pydantic models
├── client.py               # OpenAI client wrapper
├── openenv.yaml            # Environment metadata
├── requirements.txt        # Dependencies
└── tests/                  # Test suite
```

## Scoring

All graders produce scores in the range [0.0, 1.0] with multi-dimensional breakdowns:
- **Easy**: 4 criteria (correctness, professionalism, conciseness, hallucination)
- **Medium**: 4 criteria (empathy, problem identification, solution, professionalism)
- **Hard**: 5 criteria (de-escalation, information gathering, resolution path, satisfaction trajectory, protocol compliance)

Scoring is deterministic and reproducible with the same seed.

## Running and Testing

### Prerequisites

This project uses a Python virtual environment (`.venv`). You must activate it before running anything.

### Option A: WSL / Linux / macOS

```bash
# From project root
cd RUNTIME/customer-service-bot

# Activate virtual environment
source .venv/bin/activate

# Install dependencies (first time only)
pip install -r requirements.txt

# Run tests
python tests/test_environment.py

# Start the server
uvicorn server.app:app --host 127.0.0.1 --port 8000
```

### Option B: Windows PowerShell

```powershell
# From project root (PS C:\NPersonal\Projects\openenv-project>)
cd RUNTIME\customer-service-bot

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies (first time only)
pip install -r requirements.txt

# Run tests
python tests\test_environment.py

# Start the server
uvicorn server.app:app --host 127.0.0.1 --port 8000
```

If activation fails on Windows due to execution policy, run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Server Output

Once started, you should see:
```
INFO:     Started server process [xxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Test the API (open a new terminal)

```bash
# Health check
curl http://127.0.0.1:8000/health

# Reset environment (easy task)
curl -X POST http://127.0.0.1:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_type": "easy", "scenario_index": 0, "seed": 42}'

# Take a step
curl -X POST http://127.0.0.1:8000/step \
  -H "Content-Type: application/json" \
  -d '{"message": "We offer Standard (5-7 business days, $5.99), Express (2-3 days, $12.99), and Overnight (next day, $24.99). Free shipping on orders over $50!", "action_type": "answer", "confidence": 0.8}'

# Get current state
curl http://127.0.0.1:8000/state
```

On Windows PowerShell, use `Invoke-RestMethod` instead:
```powershell
# Health check
Invoke-RestMethod -Uri http://127.0.0.1:8000/health

# Reset
Invoke-RestMethod -Uri http://127.0.0.1:8000/reset -Method POST -ContentType "application/json" -Body '{"task_type":"easy","scenario_index":0,"seed":42}'

# Step
Invoke-RestMethod -Uri http://127.0.0.1:8000/step -Method POST -ContentType "application/json" -Body '{"message":"We offer Standard 5-7 business days","action_type":"answer","confidence":0.8}'

# State
Invoke-RestMethod -Uri http://127.0.0.1:8000/state
```

### Expected Responses

**`/health`** →
```json
{"status": "ok"}
```

**`/reset`** → Observation object:
```json
{
  "customer_message": "Hi! I'm thinking of placing an order. How long does shipping usually take?",
  "sentiment": 0.2,
  "intent": "shipping_inquiry",
  "urgency": 2,
  "conversation_history": ["Hi! I'm thinking of placing an order..."],
  "turn_count": 0,
  "task_type": "easy",
  "scenario_description": "Customer wants to know shipping timeframes"
}
```

**`/step`** → StepResponse:
```json
{
  "observation": {
    "customer_message": "Thank you, that's helpful!",
    "sentiment": 0.6,
    "turn_count": 1,
    "task_type": "easy"
  },
  "reward": {
    "value": 0.332,
    "breakdown": {"progress": 0.265, "sentiment_delta": 0.03, "action_bonus": 0.0, "verbosity_penalty": 0.0},
    "feedback": "Accurate information provided"
  },
  "done": false,
  "info": {
    "turn_count": 1,
    "customer_mood": 7.5,
    "is_resolved": false,
    "is_escalated": false,
    "task_result": {"score": 0.795, "breakdown": {"correctness": 0.68, "professionalism": 0.5, "conciseness": 1.0, "hallucination": 1.0}}
  }
}
```

**`/state`** → State object:
```json
{
  "customer_mood": 7.5,
  "satisfaction_trajectory": [7.0, 7.5],
  "resolution_status": "ongoing",
  "turn_count": 1,
  "task_type": "easy",
  "conversation_history": ["Hi! I'm thinking...", "Thank you, that's helpful!"],
  "episode_done": false
}
```

### Run the Test Suite

```bash
# Make sure .venv is activated first
cd RUNTIME/customer-service-bot
python tests/test_environment.py
```

Expected output:
```
============================================================
CUSTOMER SERVICE BOT - TEST SUITE
============================================================
...
RESULTS: 8 passed, 0 failed, 8 total
============================================================
```

This runs 8 tests locally without needing the server running:
- Reset all task types (easy, medium, hard)
- Easy task full episode
- Medium task full episode
- Hard task full episode
- State management
- Reward range validation (all rewards in [0.0, 1.0])
- Deterministic scoring (same seed = same results)
- Episode boundaries (episodes end correctly)

