"""Train a hierarchical Bayesian shipping-cost model and export a serving artifact."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

ROOT = Path(__file__).resolve().parent
CARRIERS = ["carrier_a", "carrier_b", "carrier_c", "carrier_d"]
ZONES = ["metro", "regional", "remote"]
DRAWS = 1000
TUNE = 1000
CHAINS = 4
SERVING_DRAWS = 1000


def simulate(n: int, rng: np.random.Generator, weight_scale: float = 1.0) -> pd.DataFrame:
    """Simulated shipments: log cost linear in carrier, zone, log weight and log distance."""
    carrier = rng.integers(0, 4, n)
    zone = rng.integers(0, 3, n)
    weight = rng.lognormal(0.3, 0.8, n) * weight_scale
    distance = rng.lognormal(5.5, 0.7, n)
    carrier_intercept = np.array([1.6, 1.4, 1.2, 2.1])
    zone_effect = np.array([0.0, 0.25, 0.6])
    mean_log_cost = (
        carrier_intercept[carrier] + zone_effect[zone] + 0.45 * np.log(weight) + 0.20 * np.log(distance / 250)
    )
    cost = np.exp(rng.normal(mean_log_cost, 0.25))
    return pd.DataFrame(
        {
            "carrier": np.array(CARRIERS)[carrier],
            "zone": np.array(ZONES)[zone],
            "weight_kg": weight.round(3),
            "distance_km": distance.round(1),
            "cost_usd": cost.round(2),
        }
    )


def _serving_array(data_array) -> np.ndarray:
    """Move the sample axis last so group indexing matches the API (carrier, draw)."""
    values = np.asarray(data_array)
    dims = getattr(data_array, "dims", ())
    if "sample" in dims:
        values = np.moveaxis(values, dims.index("sample"), -1)
    return values


def fit_and_export(shipments: pd.DataFrame, out_dir: Path = ROOT) -> None:
    carrier_index = shipments["carrier"].map({name: i for i, name in enumerate(CARRIERS)}).to_numpy()
    zone_index = shipments["zone"].map({name: i for i, name in enumerate(ZONES)}).to_numpy()
    log_weight = np.log(shipments["weight_kg"].to_numpy())
    log_distance = np.log(shipments["distance_km"].to_numpy() / 250)
    log_cost = np.log(shipments["cost_usd"].to_numpy())

    with pm.Model(coords={"carrier": CARRIERS, "zone": ZONES}):
        mu_a = pm.Normal("mu_a", 1.5, 1.0)
        sd_a = pm.HalfNormal("sd_a", 0.5)
        intercept = pm.Normal("a", mu_a, sd_a, dims="carrier")
        zone_raw = pm.Normal("z_raw", 0, 0.5, dims="zone")
        # Metro is the baseline, so zone effects are differences from zone 0.
        zone_effect = pm.Deterministic("zone_eff", zone_raw - zone_raw[0], dims="zone")
        weight_slope = pm.Normal("b_w", 0.5, 0.3)
        distance_slope = pm.Normal("b_d", 0.2, 0.3)
        sigma = pm.HalfNormal("sigma", 0.5)
        mean = (
            intercept[carrier_index]
            + zone_effect[zone_index]
            + weight_slope * log_weight
            + distance_slope * log_distance
        )
        pm.Normal("log_cost", mean, sigma, observed=log_cost)
        idata = pm.sample(
            DRAWS,
            tune=TUNE,
            chains=CHAINS,
            cores=2,
            target_accept=0.9,
            random_seed=1,
            progressbar=False,
        )
        pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=1, progressbar=False)

    summary = az.summary(idata, var_names=["a", "zone_eff", "b_w", "b_d", "sigma"])
    print(summary.to_string())
    n_divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
    print("divergences:", n_divergences)
    replicated = np.asarray(idata["posterior_predictive"]["log_cost"]).reshape(-1, len(shipments))
    low, high = np.quantile(replicated, [0.05, 0.95], axis=0)
    coverage = float(np.mean((log_cost >= low) & (log_cost <= high)))
    print("PPC 90% interval coverage (train):", round(coverage, 3))

    free = summary.drop(index=[idx for idx in summary.index if str(idx).endswith("[metro]")], errors="ignore")
    max_rhat = float(free["r_hat"].astype(float).max())
    min_ess = float(free["ess_bulk"].astype(float).min())
    if n_divergences != 0 or max_rhat >= 1.01 or min_ess < 400:
        raise RuntimeError(
            f"sampling gates failed: divergences={n_divergences} rhat={max_rhat} ess={min_ess}"
        )

    posterior = az.extract(idata, num_samples=SERVING_DRAWS, random_seed=0)
    arrays = {name: _serving_array(posterior[name]) for name in ["a", "zone_eff", "b_w", "b_d", "sigma"]}
    np.savez(out_dir / "posterior_draws.npz", **arrays)

    csv_path = out_dir / "train.csv"
    shipments.to_csv(csv_path, index=False)
    data_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()[:12]
    meta = {
        "model_name": "shipping_cost_hier_lognormal",
        "model_version": "1.0.0",
        "train_data_sha": data_hash,
        "carriers": CARRIERS,
        "zones": ZONES,
        "n_draws": SERVING_DRAWS,
        "max_rhat": max_rhat,
        "min_ess_bulk": int(min_ess),
        "train_ref": {"log_weight_quantiles": np.quantile(log_weight, np.linspace(0, 1, 11)).tolist()},
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    public = {key: meta[key] for key in list(meta) if key != "train_ref"}
    print(json.dumps(public))
    print("serving shapes:", {key: value.shape for key, value in arrays.items()})

    # Full InferenceData (posterior plus replicated costs) is the audit file.
    # It is about 100 MB, so this study repo records the size and does not keep it.
    nc_path = Path(tempfile.gettempdir()) / "shipping_idata.nc"
    idata.to_netcdf(nc_path)
    print(f"audit netcdf bytes (deleted, not committed): {nc_path.stat().st_size}")
    nc_path.unlink()


def main() -> None:
    shipments = simulate(4000, np.random.default_rng(2026))
    fit_and_export(shipments)


if __name__ == "__main__":
    main()
