"""FastAPI service: posterior predictive shipping-cost quotes with 90% intervals."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
META = json.loads((ROOT / "model_meta.json").read_text(encoding="utf-8"))
DRAWS = np.load(ROOT / "posterior_draws.npz")
CARRIER_INDEX = {name: i for i, name in enumerate(META["carriers"])}
ZONE_INDEX = {name: i for i, name in enumerate(META["zones"])}
RNG = np.random.default_rng(0)


class QuoteRequest(BaseModel):
    carrier: Literal["carrier_a", "carrier_b", "carrier_c", "carrier_d"]
    zone: Literal["metro", "regional", "remote"]
    weight_kg: float = Field(gt=0, le=70)
    distance_km: float = Field(gt=0, le=5000)
    quoted_price_usd: float | None = Field(default=None, gt=0)


class QuoteResponse(BaseModel):
    model_name: str
    model_version: str
    expected_cost_usd: float
    median_cost_usd: float
    ci90_low_usd: float
    ci90_high_usd: float
    prob_cost_exceeds_quote: float | None
    out_of_training_range: bool
    latency_ms: float


app = FastAPI(title="shipping-cost-model", version=META["model_version"])


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_version": META["model_version"],
        "train_data_sha": META["train_data_sha"],
    }


@app.post("/v1/shipping-cost/quote", response_model=QuoteResponse)
def quote(request: QuoteRequest) -> QuoteResponse:
    started = time.perf_counter()
    mean = (
        DRAWS["a"][CARRIER_INDEX[request.carrier]]
        + DRAWS["zone_eff"][ZONE_INDEX[request.zone]]
        + DRAWS["b_w"] * np.log(request.weight_kg)
        + DRAWS["b_d"] * np.log(request.distance_km / 250)
    )
    # Parameter uncertainty (the draw) plus observation noise.
    predictive = np.exp(RNG.normal(mean, DRAWS["sigma"]))
    quantiles = META["train_ref"]["log_weight_quantiles"]
    log_weight = float(np.log(request.weight_kg))
    exceed = None
    if request.quoted_price_usd is not None:
        exceed = round(float(np.mean(predictive > request.quoted_price_usd)), 3)
    return QuoteResponse(
        model_name=META["model_name"],
        model_version=META["model_version"],
        expected_cost_usd=round(float(predictive.mean()), 2),
        median_cost_usd=round(float(np.median(predictive)), 2),
        ci90_low_usd=round(float(np.quantile(predictive, 0.05)), 2),
        ci90_high_usd=round(float(np.quantile(predictive, 0.95)), 2),
        prob_cost_exceeds_quote=exceed,
        out_of_training_range=not (quantiles[0] <= log_weight <= quantiles[-1]),
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )
