import numpy as np

class BatteryEnvironment:
    def __init__(self, capacity=10.0, seed=1):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.charge_rate = 2.0
        self.efficiency = 0.92
        self.reset()

    def reset(self):
        self.soc = self.capacity * 0.5
        self.t = 0
        return self.state()

    def state(self):
        soc_bin = int(np.clip(self.soc / self.capacity * 5, 0, 4))
        return (soc_bin, self.t % 24)

    def price(self):
        return 0.08 + 0.08 * (0.5 + 0.5 * np.sin((self.t - 8) * np.pi / 12))

    def step(self, action, net_load, pv):
        action = int(action)
        grid_before = max(0.0, net_load - pv)
        grid_after = grid_before

        if action == 1:
            available = max(0.0, self.capacity - self.soc)
            charge = min(self.charge_rate, available)
            self.soc += charge * self.efficiency
            grid_after += max(0.0, charge - max(0.0, pv - net_load))

        elif action == 2:
            discharge = min(self.charge_rate, self.soc)
            self.soc -= discharge
            grid_after = max(0.0, grid_before - discharge)

        cost = grid_after * self.price()
        soc_penalty = 0.01 if (
            self.soc < 0.1 * self.capacity or
            self.soc > 0.95 * self.capacity
        ) else 0.0

        reward = -(cost + soc_penalty)
        self.t += 1
        done = self.t >= 24

        return self.state(), reward, done, {
            "soc": self.soc,
            "grid_energy": grid_after,
            "cost": cost,
        }

class QLearningBatteryAgent:
    def __init__(self, alpha=0.15, gamma=0.95, epsilon=0.2, seed=2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.q = {}

    def vals(self, state):
        if state not in self.q:
            self.q[state] = np.zeros(3)
        return self.q[state]

    def act(self, state, explore=True):
        values = self.vals(state)
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(3))
        return int(np.argmax(values))

    def update(self, state, action, reward, next_state, done):
        values = self.vals(state)
        target = reward if done else reward + self.gamma * np.max(self.vals(next_state))
        values[action] += self.alpha * (target - values[action])
