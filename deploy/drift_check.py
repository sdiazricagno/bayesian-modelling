"""Input PSI and 90% predictive-interval coverage for the shipping-cost model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from train import simulate

ROOT = Path(__file__).resolve().parent


def psi(reference: np.ndarray, new: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    reference_share = np.histogram(reference, edges)[0] / len(reference)
    new_share = np.histogram(new, edges)[0] / len(new)
    reference_share = np.clip(reference_share, 1e-6, None)
    new_share = np.clip(new_share, 1e-6, None)
    return float(np.sum((new_share - reference_share) * np.log(new_share / reference_share)))


def coverage(frame: pd.DataFrame, draws: np.lib.npyio.NpzFile, meta: dict, rng: np.random.Generator) -> float:
    carrier_index = {name: i for i, name in enumerate(meta["carriers"])}
    zone_index = {name: i for i, name in enumerate(meta["zones"])}
    carrier = frame["carrier"].map(carrier_index).to_numpy()
    zone = frame["zone"].map(zone_index).to_numpy()
    mean = (
        draws["a"][carrier].T
        + draws["zone_eff"][zone].T
        + draws["b_w"][:, None] * np.log(frame["weight_kg"].to_numpy())
        + draws["b_d"][:, None] * np.log(frame["distance_km"].to_numpy() / 250)
    )
    predictive = np.exp(rng.normal(mean, draws["sigma"][:, None]))
    low, high = np.quantile(predictive, [0.05, 0.95], axis=0)
    actual = frame["cost_usd"].to_numpy()
    return float(np.mean((actual >= low) & (actual <= high)))


def _status(score: float) -> str:
    if score > 0.25:
        return "ALERT"
    if score > 0.1:
        return "watch"
    return "ok"


def evaluate() -> list[dict]:
    draws = np.load(ROOT / "posterior_draws.npz")
    meta = json.loads((ROOT / "model_meta.json").read_text(encoding="utf-8"))
    train_path = ROOT / "train.csv"
    reference = (
        pd.read_csv(train_path) if train_path.exists() else simulate(4000, np.random.default_rng(2026))
    )
    reference_log_weight = np.log(reference["weight_kg"].to_numpy())
    rng = np.random.default_rng(5)
    rows = []
    scenarios = [
        ("week_ok", simulate(800, rng)),
        ("week_heavier_parcels", simulate(800, rng, weight_scale=1.8)),
    ]
    for name, week in scenarios:
        score = psi(reference_log_weight, np.log(week["weight_kg"].to_numpy()))
        row = {
            "name": name,
            "psi_log_weight": score,
            "status": _status(score),
            "coverage": coverage(week, draws, meta, rng),
        }
        rows.append(row)
    shifted = simulate(800, rng)
    carrier_d = shifted["carrier"] == "carrier_d"
    shifted.loc[carrier_d, "cost_usd"] = shifted.loc[carrier_d, "cost_usd"] * 1.15
    score = psi(reference_log_weight, np.log(shifted["weight_kg"].to_numpy()))
    rows.append(
        {
            "name": "carrier_d_tariff",
            "psi_log_weight": score,
            "status": _status(score),
            "coverage": coverage(shifted, draws, meta, rng),
            "coverage_carrier_d": coverage(shifted.loc[carrier_d], draws, meta, rng),
        }
    )
    return rows


def main() -> list[dict]:
    rows = evaluate()
    for row in rows:
        if row["name"] == "carrier_d_tariff":
            print(
                f"carrier_d +15% tariff: PSI(log weight)={row['psi_log_weight']:.3f}; "
                f"coverage all={row['coverage']:.3f}; coverage carrier_d={row['coverage_carrier_d']:.3f}"
            )
        else:
            print(
                f"{row['name']}: PSI(log weight)={row['psi_log_weight']:.3f} -> {row['status']}; "
                f"90% interval coverage on actual costs={row['coverage']:.3f}"
            )
    return rows


if __name__ == "__main__":
    main()
