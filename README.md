# Bayesian modelling

Interview-prep notebooks for a senior data scientist role in statistical modelling: Bayesian methods for e-commerce pricing, demand, segmentation, and shipping cost. Every dataset is simulated. Seeds are fixed.

## Study order

| # | Notebook | What it trains |
|---|---|---|
| 01 | [Bayesian price elasticity](notebooks/01_bayesian_price_elasticity.ipynb) | Priors, likelihood, posterior, prior and posterior predictive checks, credible intervals, a pricing decision |
| 02 | [Markov chains and HMMs](notebooks/02_markov_chains_and_hmms.ipynb) | Lifecycle transitions, time to churn, Baum-Welch, BIC, Viterbi, forward-backward |
| 03 | [Metropolis-Hastings](notebooks/03_metropolis_hastings.ipynb) | A sampler from scratch, step size, a covariance-adapted proposal, R-hat and ESS, a NUTS cross-check |
| 04 | [GMM and EM](notebooks/04_gmm_and_em.ipynb) | EM for a 1D mixture versus scikit-learn, then BIC/AIC segmentation |
| 05 | [Beta-Binomial segment conversion](notebooks/05_bayesian_segment_conversion.ipynb) | Closed-form segmentation. The case where MCMC is the wrong tool |
| 06 | [Train and deploy](notebooks/06_train_and_deploy_shipping_cost.ipynb) | Hierarchical shipping-cost model, thinned posterior, FastAPI contract |
| 07 | [Lifecycle, drift, retraining](notebooks/07_model_lifecycle_drift.ipynb) | PSI, interval coverage, when to retrain, how to say the uncertainty |

A gap-focused pass can start at 02 and 03, then 04, then 01 and 05, then 06 and 07. Each notebook ends with a short "how to explain this to a CTO" cell.

## Setup

Python 3.12. These notebooks were executed with 3.12.3. The reference run of the same examples used 3.13.5 with the library pins below; 3.12 is the closest interpreter on this machine and the pins are unchanged.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name bayesian-modelling
```

`requirements.txt` pins PyMC 6.3.2, ArviZ 1.3.0, hmmlearn 0.3.3, scikit-learn 1.9.1, FastAPI 0.162.0, pydantic 2.14.1, NumPy 2.5.4, and the libraries imported next to them. `deploy/requirements.txt` is the serving image only: no PyMC.

ArviZ 1 prints an 89% equal-tailed interval as `eti89_lb` / `eti89_ub`. Older ArviZ prints `hdi_3%` / `hdi_97%`.

## Run the notebooks

From `notebooks/`, open them in order. Notebook 07 reads the artifact written by notebook 06.

```bash
python -m nbconvert --execute --inplace notebooks/*.ipynb
```

Sampling is the published length (four chains, on the order of 1,000 draws) because these models finish in a few minutes on a laptop. Do not turn the progress bar back on if you are storing outputs.

## Run the shipping-cost service

```bash
cd deploy
python train.py    # writes train.csv, posterior_draws.npz, model_meta.json
uvicorn app:app --port 8000
```

`GET /health` and `POST /v1/shipping-cost/quote`. The handler loads the thinned draws once and does not run MCMC. `python drift_check.py` prints PSI and 90% interval coverage for three simulated weeks.

`train.py` also writes an ArviZ netcdf under the system temp directory, prints its size, and deletes it. A full InferenceData with replicated costs is about 100 MB. Regenerate it locally if you need the audit file. Do not commit it. `*.nc` is gitignored.

The `Dockerfile` is the serving image (FastAPI, uvicorn, NumPy, pydantic). It was not built here because Docker is not installed in the execution environment.

## Sources

- [PyMC documentation](https://www.pymc.io/projects/docs/en/stable/) and the [example gallery](https://www.pymc.io/projects/examples/en/latest/gallery.html)
- [Prior and posterior predictive checks](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/posterior_predictive.html)
- [ArviZ](https://python.arviz.org/en/stable/)
- [Vehtari et al., rank-normalised R-hat](https://arxiv.org/abs/1903.08008)
- [hmmlearn](https://hmmlearn.readthedocs.io/en/latest/)
- [scikit-learn Gaussian mixtures](https://scikit-learn.org/stable/modules/mixture.html)
- [FastAPI](https://fastapi.tiangolo.com/) and [Pydantic](https://docs.pydantic.dev/latest/)
