import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.stats import norm  # normal distribution

## ---------------------------------------------------------
## 2.1 The model
## ---------------------------------------------------------


def simulate_income_model(
    seed=1234,
    education_diff=True,
    shocks=True,
    depreciation=True,
    unemployment=True
):

    # Parameters

    N = 50_000

    # Education parameters

    p_e = np.array([0.40, 0.35, 0.25])
    S_e = np.array([1, 3, 5])
    h0_e = np.array([1.00, 1.20, 1.55])
    Delta_e = np.array([0.010, 0.020, 0.030])

    # Human capital parameters

    delta = 0.06
    sigma_psi = 0.10

    # Labor market parameters

    job_finding = 0.60
    job_separation = 0.05

    # Income parameters

    y_SU = 0.45
    rho = 0.60
    y_floor = 0.35

    # Ages

    ages = np.arange(18, 66)
    T = len(ages)

    # Random number generator

    rng = np.random.default_rng(seed)

    # Education

    education_draw = rng.choice(
        3,
        size=N,
        p=p_e
    )

    if education_diff:
        education = education_draw
    else:
        # Everyone gets medium education
        education = np.ones(N, dtype=int)

    education_years = S_e[education]

    human_capital = h0_e[education].copy()

    growth = Delta_e[education]

    # Arrays for simulation

    income = np.zeros((N, T))

    employed_history = np.zeros((N, T), dtype=bool)

    labor_force_history = np.zeros((N, T), dtype=bool)

    employed = np.zeros(N, dtype=bool)

    last_job_income = np.full(N, np.nan)

    # Simulate life cycle

    for t, age in enumerate(ages):

        years_since_18 = age - 18

        # Education or labor market

        in_education = years_since_18 < education_years

        in_labor_market = ~in_education

        # Employment status

        if unemployment:

            working = in_labor_market & employed

            unemployed = in_labor_market & ~employed

        else:

            # Everyone works immediately after education
            working = in_labor_market

            unemployed = np.zeros(N, dtype=bool)

        # Save status

        labor_force_history[:, t] = in_labor_market

        employed_history[:, t] = working

        # Income during education

        income[in_education, t] = y_SU

        # Income while employed

        income[working, t] = human_capital[working]

        last_job_income[working] = income[working, t]

        # Income while unemployed

        had_job_before = unemployed & ~np.isnan(last_job_income)

        income[had_job_before, t] = (
            rho * last_job_income[had_job_before]
        )

        # Benefit floor if never employed

        never_had_job = unemployed & np.isnan(last_job_income)

        income[never_had_job, t] = y_floor

        # Stop after age 65

        if t == T - 1:
            continue

        # Human capital shock

        psi_draw = rng.lognormal(
            mean=-0.5 * sigma_psi**2,
            sigma=sigma_psi,
            size=N
        )

        if shocks:
            psi = psi_draw
        else:
            psi = np.ones(N)

        # Human capital while employed

        human_capital[working] = (
            human_capital[working]
            * (1 + growth[working])
            * psi[working]
        )

        # Human capital while unemployed

        if depreciation:
            depreciation_rate = delta
        else:
            depreciation_rate = 0.0

        human_capital[unemployed] = (
            human_capital[unemployed]
            * (1 - depreciation_rate)
            * psi[unemployed]
        )

        # Labor market transitions

        draw = rng.random(N)

        if unemployment:

            find_job = unemployed & (draw < job_finding)

            lose_job = working & (draw < job_separation)

            employed[find_job] = True

            employed[lose_job] = False

    # Return results

    return {
        "ages": ages,
        "income": income,
        "education": education,
        "employed": employed_history,
        "labor_force": labor_force_history
    }
