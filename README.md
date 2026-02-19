# Trading Volume Forecasting with a Residual GRU

## Summary

This project explores one-day-ahead trading volume forecasting using a time series dataset with 320 latent features. The goal was to test whether increasingly expressive GRU-based models could outperform a simple persistence rule under a leakage-safe rolling-origin backtest.

Across multiple model variants, the persistence approach turned out to be a very strong baseline. This led to a reformulation of the task as a residual forecasting problem, where the model predicts only the deviation from persistence.

## Key results

Average across rolling-origin folds:

- Persistence (P): RMSE = 65.6M
- Residual GRU (Res): RMSE = 63.0M
  → **~4% improvement over persistence**

**Main findings:**

- Direct GRU models (A, B, C) did not outperform persistence.
- The residual formulation produced a consistent improvement across folds.
- Multi-seed checks confirmed that the gain is stable.

## Modeling journey

The project followed an iterative modeling process:

1. **Baseline (P)**
2. **Model A**: Uses only past target values
3. **Model B**: Adds the most recent covariates
4. **Model C**: Uses full covariate history

Despite increasing complexity, none of the GRU models outperformed the persistence rule. Autocorrelation analysis showed strong short-term persistence, indicating that the main challenge was not model capacity but problem formulation. This motivated a residual forecasting approach.

## Final residual model

Instead of predicting the next level directly, the model predicts the residual:

r[t+1] = y[t+1] − y[t], y_hat[t+1] = y[t] + r_hat[t+1]

A GRU-based network is trained to estimate the residual from recent target history:
r_hat[t+1] = g(y[t−w+1 : t])

This formulation:

- Preserves the strong persistence structure.
- Lets the neural network focus only on short-term deviations.
- Produces a consistent improvement over the baseline.

## Repository structure

```text
.
├── src/
│   ├── config.py        # seeds and experiment configuration
│   ├── data.py          # data loading and sequence construction
│   ├── models.py        # GRU architectures
│   ├── training.py      # training utilities
│   ├── backtest.py      # rolling-origin evaluation
│   ├── evaluation.py    # metrics
│   ├── plots.py         # visualization helpers
│   └── __init__.py
├── 01_eda.ipynb       # exploratory analysis (ACF, PACF, distributions)
├── 02_main.ipynb      # main experiments and results
├── 03_appendix.ipynb  # multi-seed robustness checks
├── Time_Series.csv    # dataset
├── requirements.txt
└── README.md
```

## How to run

1. Install dependencies:
   ```bash
    pip install -r requirements.txt
   ```
2. Run the notebooks in order:
   - 01_eda.ipynb
   - 02_main.ipynb
   - 03_appendix.ipynb (optional robustness checks)

   All experiments are fully reproducible with fixed random seeds.
