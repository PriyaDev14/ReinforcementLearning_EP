from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .environment import BuildingEnvironment
from .hvac_agent import QLearningHVACAgent
from .federated import federated_average, copy_model
from .battery_agent import BatteryEnvironment, QLearningBatteryAgent
from .metrics import summarize

class EnergySystemSimulator:
    def __init__(self, num_buildings=3, federated_rounds=15,
                 local_episodes=12, seed=42):
        self.num_buildings = num_buildings
        self.federated_rounds = federated_rounds
        self.local_episodes = local_episodes
        self.seed = seed

        self.agents = [
            QLearningHVACAgent(seed=seed + i)
            for i in range(num_buildings)
        ]
        self.envs = [
            BuildingEnvironment(i, seed=seed + 100 + i)
            for i in range(num_buildings)
        ]
        self.battery_agent = QLearningBatteryAgent(seed=seed + 500)
        self.battery_env = BatteryEnvironment(seed=seed + 600)

    def federated_train(self):
        for rnd in range(self.federated_rounds):
            local_models = []
            for i, (agent, env) in enumerate(zip(self.agents, self.envs)):
                agent.epsilon = max(
                    0.03, 0.25 * (1 - rnd / max(1, self.federated_rounds))
                )
                agent.train(env, episodes=self.local_episodes)
                local_models.append(copy_model(agent.q))

            global_model = federated_average(local_models)
            for agent in self.agents:
                agent.q = copy_model(global_model)

    def train_battery(self, episodes=80):
        rng = np.random.default_rng(self.seed)
        for _ in range(episodes):
            state = self.battery_env.reset()
            done = False
            while not done:
                hour = self.battery_env.t
                base_load = 3.0 + 1.2 * int(7 <= hour <= 21)
                pv = max(0.0, 3.5 * np.sin((hour - 6) * np.pi / 12))
                net_load = base_load + rng.normal(0, 0.15)

                action = self.battery_agent.act(state)
                next_state, reward, done, _ = self.battery_env.step(
                    action, net_load, pv
                )
                self.battery_agent.update(
                    state, action, reward, next_state, done
                )
                state = next_state

    def evaluate(self):
        rows = []
        self.battery_env.reset()

        for hour in range(24):
            total_hvac = 0.0
            total_pv = 0.0
            comfort = 0.0
            temperatures = []

            for agent, env in zip(self.agents, self.envs):
                env.t = hour
                state = env.state()
                action = agent.act(state, explore=False)
                _, _, _, info = env.step(action)

                total_hvac += info["hvac_energy"]
                total_pv += info["pv"]
                comfort += info["comfort_violation"]
                temperatures.append(info["temperature"])

            state = self.battery_env.state()
            net_load = total_hvac + 1.5
            pv = total_pv / max(1, self.num_buildings)
            action = self.battery_agent.act(state, explore=False)
            _, _, _, binfo = self.battery_env.step(action, net_load, pv)

            rows.append({
                "hour": hour,
                "temperature": float(np.mean(temperatures)),
                "hvac_energy": total_hvac,
                "pv": pv,
                "pv_used": min(pv, net_load),
                "grid_energy": binfo["grid_energy"],
                "cost": binfo["cost"],
                "comfort_violation": comfort,
                "soc": binfo["soc"],
            })

        return pd.DataFrame(rows)

    def run(self):
        self.federated_train()
        self.train_battery()
        history = self.evaluate()
        return {"history": history, "summary": summarize(history)}

    def plot(self, output: Path):
        history = pd.read_csv("results/simulation_history.csv")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history["hour"], history["grid_energy"], label="Grid energy")
        ax.plot(history["hour"], history["hvac_energy"], label="HVAC energy")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Energy")
        ax.set_title("Simulated Energy Consumption")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        plt.close(fig)
