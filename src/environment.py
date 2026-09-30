import numpy as np

class BuildingEnvironment:
    def __init__(self, building_id=0, seed=0, horizon=24):
        self.building_id = building_id
        self.rng = np.random.default_rng(seed)
        self.horizon = horizon
        self.target = 24.0 + 0.25 * building_id
        self.thermal_gain = 0.10 + 0.015 * building_id
        self.hvac_power = np.array([0.0, 0.8, 1.5])
        self.reset()

    def reset(self):
        self.t = 0
        self.temperature = float(self.rng.normal(27.0, 0.5))
        self.occupancy = 0
        return self.state()

    def outside_temperature(self):
        return 30 + 4 * np.sin((self.t - 7) * np.pi / 12)

    def price(self):
        return 0.08 + 0.08 * (0.5 + 0.5 * np.sin((self.t - 8) * np.pi / 12))

    def pv(self):
        daylight = max(0.0, np.sin((self.t - 6) * np.pi / 12))
        return 2.2 * daylight

    def state(self):
        temp_bin = int(np.clip(round(self.temperature) - 20, 0, 12))
        occ = int(7 <= self.t <= 20)
        price_bin = int(self.price() > 0.12)
        return (temp_bin, occ, self.t % 24, price_bin)

    def step(self, action):
        action = int(np.clip(action, 0, 2))
        self.occupancy = int(7 <= self.t <= 20)

        outside = self.outside_temperature()
        power = float(self.hvac_power[action])
        cooling = [0.0, 0.45, 0.85][action]
        noise = self.rng.normal(0, 0.06)

        self.temperature += (
            self.thermal_gain * (outside - self.temperature)
            - cooling + noise
        )

        comfort_error = abs(self.temperature - self.target)
        comfort_penalty = comfort_error * (2.0 if self.occupancy else 0.4)
        energy_cost = power * self.price()
        reward = -(energy_cost + comfort_penalty)

        self.t += 1
        done = self.t >= self.horizon

        info = {
            "temperature": self.temperature,
            "hvac_energy": power,
            "price": self.price(),
            "pv": self.pv(),
            "comfort_violation": float(
                self.occupancy and comfort_error > 1.5
            ),
        }
        return self.state(), reward, done, info
