def summarize(history):
    return {
        "total_cost": float(history["cost"].sum()),
        "grid_energy": float(history["grid_energy"].sum()),
        "hvac_energy": float(history["hvac_energy"].sum()),
        "comfort_violations": float(history["comfort_violation"].sum()),
        "pv_energy": float(history["pv"].sum()),
        "pv_used": float(history["pv_used"].sum()),
        "mean_temperature": float(history["temperature"].mean()),
    }
