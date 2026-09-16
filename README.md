# PVI Landscapes on Tractable Exact Posteriors

Possibilistic Variational Inference (PVI) evaluated on four one-dimensional problems
with known exact posteriors g*_max: fits candidate possibility families (Gaussian,
Laplace, Beta), verifies the LCBO (s = -1) / UCBO (s = +1) sandwich decomposition
via the max-relative entropy D_max, checks grid resolution and domain truncation artefacts,
and evaluates order-p divergence softenings (D_p) to confirm candidate misspecification.


## Structure

```
xp_lands.ipynb   -- main notebook (fits, sandwich bounds, order-p divergence)
src/
  cases.py       -- tractable 1D test cases with known exact posteriors g*_max
  families.py    -- candidate possibility families (Gaussian, Laplace, Beta, asymmetric Gaussian)
  cbo.py         -- LCBO, UCBO, D_max, and the sandwich bound check
results/         -- summary_metrics.csv (cross-case LCBO/UCBO/D_max table)
figs/            -- fig_laplace_posterior_logscale.png (the only figure used in report.tex)
requirements.txt
```


## Setup

The full pipeline can be run on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## Usage

Open `xp_lands.ipynb` and run all cells sequentially.
For each of the 4 cases, the notebook fits candidate families to g*_max and checks the
sandwich decomposition log Z_max = LCBO(g) + D_max(g||g*_max) = UCBO(g) - D_max(g*_max||g).
A cross-case summary is written to `results/summary_metrics.csv`, and a section
relaxes D_max into the order-p divergence D_p, softened via log-sum-exp and recovering
D_max exactly as p -> infty. The last cells probe the finite-implementation effects
discussed in `report.tex`: domain truncation via domain-widening for cases 1 and 4,
grid resolution via the N-convergence check for case 3, float32 vs float64 precision
for case 2, and the grid domain/boundary summary for case 1.


## Cases and Targets

| # | Case | Exact posterior g* | Geometry / Challenge |
|---|---|---|---|
| 1 | Gaussian (known variance) | Gaussian possibility | Smooth quadratic energy, symmetric exact recovery |
| 2 | Bernoulli-Beta | Beta possibility | Compact support [0, 1], asymmetric boundary behavior |
| 3 | Laplace likelihood | Laplace possibility | Non-smooth L1 cusp at median, sharp Hessian discontinuity |
| 4 | Bimodal mixture | Mixture possibility | Non-convex multimodal landscape, mode selection under CBO |


## Evaluation Metrics

- **Sandwich bounds:** LCBO(g) <= log Z_max <= UCBO(g).
- **Max-relative entropy:** D_max(g \|\| g*_max) (under-estimation, LCBO residual) and D_max(g*_max \|\| g) (over-estimation, UCBO residual).
- **Order-p divergence:** D_p(g \|\| f), a log-sum-exp softening of D_max that recovers it exactly as p -> infty.


## Summary of Results

Fitted-candidate sandwich decomposition (matches Table 5, Section "Landscapes on Tractable
Posteriors" of `report.tex`):

| Case | Family | log Z_max | LCBO | UCBO | D_max(g\|\|g*) | D_max(g*\|\|g) |
|---|---|---|---|---|---|---|
| Gaussian (sanity) | Gaussian | -0.1961 | -0.7632 | 0.3020 | 0.5671 | 0.4980 |
| Bernoulli-Beta | Gaussian | -15.1582 | -152.4862 | -15.1556 | 137.3280 | 0.0026 |
| Bernoulli-Beta | Beta | -15.1582 | -15.1582 | -15.1582 | 3.05e-05 | 2.29e-05 |
| Laplace | Gaussian | -17.0948 | -17.4412 | 632.9875 | 0.3464 | 650.0823 |
| Laplace | Laplace | -17.0948 | -110.9888 | -16.6765 | 93.8941 | 0.4183 |
| Bimodal | Gaussian (mu=2) | -9.4045 | -43.4301 | 6.3974 | 34.0256 | 15.8019 |