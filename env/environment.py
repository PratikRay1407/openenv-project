class OpenEnv:
    def __init__(self):
        self.state_data = {}

    def reset(self):
        self.state_data = {"message": "env started"}
        return self.state_data

    def step(self, action: dict):
        return self.state_data, 0.1, False, {}

    def state(self):
        return self.state_data