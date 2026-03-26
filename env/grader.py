def grade_easy(action):
    return 1.0 if action.get("action_type") else 0.0


def grade_medium(action):
    return 0.5 if action.get("action_type") else 0.0


def grade_hard(action):
    score = 0
    if action.get("action_type"):
        score += 0.3
    if action.get("payload"):
        score += 0.4
    return score