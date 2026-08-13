import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.stats import norm  # normal distribution

## ---------------------------------------------------------
## 2.1 The model
## ---------------------------------------------------------


def simulate_income_model(seed=1234):

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

    # 0 = short
    # 1 = medium
    # 2 = long
    education = rng.choice(
        3,
        size=N,
        p=p_e
    )

    # Number of years in education
    education_years = S_e[education]

    # Initial human capital
    human_capital = h0_e[education].copy()

    # Education-specific human capital growth
    growth = Delta_e[education]

    # Arrays for simulation

    # Income for each person at each age
    income = np.zeros((N, T))

    # Employment status for each person at each age
    employed_history = np.zeros((N, T), dtype=bool)

    # Whether each person is in the labor market
    labor_force_history = np.zeros((N, T), dtype=bool)

    # Everyone starts outside employment
    employed = np.zeros(N, dtype=bool)

    # Last income from a job
    # nan means the person has never had a job
    last_job_income = np.full(N, np.nan)

    # Simulate life cycle

    for t, age in enumerate(ages):

        years_since_18 = age - 18

        # Education or labor market

        in_education = years_since_18 < education_years
        in_labor_market = ~in_education

        # Employment status

        working = in_labor_market & employed
        unemployed = in_labor_market & ~employed

        # Save status

        labor_force_history[:, t] = in_labor_market
        employed_history[:, t] = working

        # Income during education

        income[in_education, t] = y_SU

        # Income while employed

        income[working, t] = human_capital[working]

        # Save most recent job income

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

        psi = rng.lognormal(
            mean=-0.5 * sigma_psi**2,
            sigma=sigma_psi,
            size=N
        )

        # Human capital while employed

        human_capital[working] = (
            human_capital[working]
            * (1 + growth[working])
            * psi[working]
        )

        # Human capital while unemployed

        human_capital[unemployed] = (
            human_capital[unemployed]
            * (1 - delta)
            * psi[unemployed]
        )

        # Human capital does not change during education

        # Labor market transitions

        draw = rng.random(N)

        # Unemployed people can find a job
        find_job = unemployed & (draw < job_finding)

        # Employed people can lose their job
        lose_job = working & (draw < job_separation)

        # Update employment status for next year
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


