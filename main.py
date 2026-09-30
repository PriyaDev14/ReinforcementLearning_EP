import argparse
from pathlib import Path
from src.simulator import EnergySystemSimulator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--buildings", type=int, default=3)
    args = parser.parse_args()

    sim = EnergySystemSimulator(
        num_buildings=args.buildings,
        federated_rounds=args.rounds,
        local_episodes=args.episodes,
        seed=42,
    )
    results = sim.run()

    out = Path("results")
    out.mkdir(exist_ok=True)
    results["history"].to_csv(out / "simulation_history.csv", index=False)
    sim.plot(out / "energy_results.png")

    print("\n=== SIMULATED RESULTS ===")
    for k, v in results["summary"].items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
