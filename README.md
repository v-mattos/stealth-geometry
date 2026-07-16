# Geometric Characterization of Stealthy Attacks in Data-Driven Robust Control

This repository contains the implementation accompanying our extension of

> Bhowmik, S., Bopardikar, S. D., and Hespanha, J. P., *Data-Driven Robust Control under Input-Output Stealthy Attacks*, IEEE Control Systems Letters (L-CSS), 2025.

The repository reproduces the original game-theoretic DeePC formulation and introduces a geometric characterization of the stealthy attack set, showing how attack authority depends on the disturbance geometry.

## Files

- `deepc.py` – DeePC primitives and data generation.
- `tracking.py` – Nominal robust tracking.
- `game.py` – Attacker and defender optimization problems.
- `stealth_geometry.py` – Geometric characterization.
- `stealth_experiment.py` – Numerical validation.
- `stealth_figures.py` – Figure generation.
- `sweep_budget.py` – Disturbance-budget analysis.
