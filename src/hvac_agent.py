import numpy as np

class QLearningHVACAgent:
    def __init__(self, actions=3, alpha=0.15, gamma=0.95, epsilon=0.25, seed=0):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.q = {}

    def _values(self, state):
        if state not in self.q:
            self.q[state] = np.zeros(self.actions, dtype=float)
        return self.q[state]

    def act(self, state, explore=True):
        values = self._values(state)
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.actions))
        return int(np.argmax(values))

    def update(self, state, action, reward, next_state, done):
        values = self._values(state)
        target = reward if done else reward + self.gamma * np.max(self._values(next_state))
        values[action] += self.alpha * (target - values[action])

    def train(self, env, episodes=10):
        scores = []
        for _ in range(episodes):
            state = env.reset()
            total = 0.0
            done = False
            while not done:
                action = self.act(state, explore=True)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state, done)
                total += reward
                state = next_state
            scores.append(total)
        return scores
