# Bayesian modelling

Portfolio of Bayesian and probabilistic models by Santiago Diaz Ricagno, Lead Data Scientist. Seven notebooks take a pricing, lifecycle, segmentation, or shipping decision from a simulated dataset to a posterior, a check that the fit is trustworthy, and a quantity a business can act on. The last two export a hierarchical shipping-cost model as a thinned posterior and serve it from FastAPI, then watch that service for input drift and miscalibration. Seeds are fixed and every dataset is simulated, so the repository can be rerun from a laptop.

## Notebooks

| Notebook | Technique | Business question |
|---|---|---|
| [01. Price elasticity](notebooks/01_bayesian_price_elasticity.ipynb) | Hierarchical log-log Poisson demand in PyMC (NUTS), partial pooling, prior and posterior predictive checks | If we raise this SKU's price by 10%, what happens to revenue, and how sure are we? |
| [02. Markov chains and HMMs](notebooks/02_markov_chains_and_hmms.ipynb) | Observed Markov chain with a Dirichlet posterior, then a Gaussian HMM (Baum-Welch, BIC, Viterbi, forward-backward) | What is this customer's lifecycle state, their chance of buying next week, and the expected time to churn? |
| [03. Metropolis-Hastings](notebooks/03_metropolis_hastings.ipynb) | Random-walk Metropolis-Hastings from scratch, a covariance-adapted proposal, R-hat and ESS, cross-checked with NUTS | When a pricing or cost model has no closed form, has the sampler actually converged? |
| [04. GMM and EM](notebooks/04_gmm_and_em.ipynb) | Expectation-maximisation for a one-dimensional Gaussian mixture, checked against scikit-learn, then BIC/AIC on two features | Which shoppers form soft segments on order frequency and basket value, and who is too ambiguous to hard-assign? |
| [05. Segment conversion](notebooks/05_bayesian_segment_conversion.ipynb) | Conjugate Beta-Binomial update | For two segments or treatments, what is the probability B beats A, and what is the expected lift? |
| [06. Train and deploy](notebooks/06_train_and_deploy_shipping_cost.ipynb) | Hierarchical log-normal shipping-cost model, thinned posterior, FastAPI request and response contract | What should checkout expect the shipment to cost, with a 90% interval and the probability it exceeds the quoted price? |
| [07. Drift and retraining](notebooks/07_model_lifecycle_drift.ipynb) | Population stability index on inputs, plus coverage of the 90% predictive interval | Has the parcel mix shifted, or have tariffs moved while the inputs look the same, so the quote model should be retrained? |

Notebooks 06 and 07 read `deploy/` as a sibling of `notebooks/`. Run them from the repository root, or from inside `notebooks/`. Both layouts resolve the same directory.

## How to run

Python 3.12. The notebooks in this repository were executed with 3.12.3. The same examples were also run with 3.13.5 against the pins in `requirements.txt`.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name bayesian-modelling
python -m nbconvert --execute --inplace notebooks/*.ipynb
```

The kernel name matches the notebooks. Execute in filename order. Notebook 07 uses the artifact notebook 06 writes (`deploy/posterior_draws.npz`, `deploy/model_meta.json`, `deploy/train.csv`). Those files are already in the repository, so 07 can run on its own. Sampling uses the published length, four chains of about 1,000 draws, which finishes in a few minutes on a laptop. Leave the progress bar off when you want to keep the stored outputs stable.

ArviZ 1 prints an 89% equal-tailed interval as `eti89_lb` / `eti89_ub`. Older ArviZ prints `hdi_3%` / `hdi_97%`.

### Shipping-cost service

```bash
cd deploy
python train.py          # writes train.csv, posterior_draws.npz, model_meta.json
uvicorn app:app --port 8000
```

`GET /health` and `POST /v1/shipping-cost/quote`. The process loads the thinned draws once at startup. `python drift_check.py` prints the population stability index and 90% interval coverage for three simulated weeks.

`train.py` writes an ArviZ netCDF under the system temp directory, prints its size, and deletes it. A full InferenceData with replicated costs is about 100 MB. Regenerate it locally when you need the audit file. `*.nc` is gitignored.

`deploy/Dockerfile` is the serving image (FastAPI, uvicorn, NumPy, pydantic, and the saved draws). `deploy/requirements.txt` is that image only.

## Tech stack

| Piece | Role |
|---|---|
| Python 3.12, NumPy, SciPy, pandas | Simulation, array maths, posterior predictive draws |
| PyMC 6.3.2, ArviZ 1.3.0 | Hierarchical models, NUTS, R-hat, ESS, posterior predictive checks |
| hmmlearn 0.3.3 | Hidden Markov model fit and decode |
| scikit-learn 1.9.1 | Gaussian-mixture reference for the EM implementation |
| Matplotlib | Diagnostic plots stored in the notebooks |
| FastAPI 0.162.0, uvicorn 0.54.0, pydantic 2.14.1 | Shipping-cost service and the request contract |
| Jupyter, nbconvert | Notebooks shipped with executed outputs |

Library versions are pinned in `requirements.txt`.

## Sources

- [PyMC documentation](https://www.pymc.io/projects/docs/en/stable/) and the [example gallery](https://www.pymc.io/projects/examples/en/latest/gallery.html)
- [Prior and posterior predictive checks](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/posterior_predictive.html)
- [ArviZ](https://python.arviz.org/en/stable/)
- [Vehtari et al., rank-normalised R-hat](https://arxiv.org/abs/1903.08008)
- [hmmlearn](https://hmmlearn.readthedocs.io/en/latest/)
- [scikit-learn Gaussian mixtures](https://scikit-learn.org/stable/modules/mixture.html)
- [FastAPI](https://fastapi.tiangolo.com/) and [Pydantic](https://docs.pydantic.dev/latest/)
