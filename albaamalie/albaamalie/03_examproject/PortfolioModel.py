""" a portfolio with a risky and a safe asset (Problem 3)

Starting point for the exam. The methods raising NotImplementedError are the ones
you should write yourself.

"""

from types import SimpleNamespace
import numpy as np

class PortfolioModelClass:
    """ a portfolio of a risky and a safe asset with a rebalancing rule """

    def __init__(self,**kwargs):
        """ set the default parameters, then overwrite with any keyword arguments """

        par = self.par = SimpleNamespace()

        # a. returns
        par.mu = 0.05 # mean log return on the risky asset
        par.sigma = 0.20 # standard deviation of the log return on the risky asset
        par.r = 0.01 # log return on the safe asset

        # b. the rebalancing rule
        par.theta_star = 0.50 # target share of wealth in the risky asset
        par.Delta = 0.10 # width of the no-trade band
        par.tau = 0.01 # proportional transaction cost

        # c. preferences
        par.gamma = 3.0 # relative risk aversion

        # d. simulation settings
        par.W0 = 1.0 # initial wealth
        par.T = 40 # number of periods
        par.N = 50_000 # number of simulated portfolios
        par.seed = 2026 # seed for the random number generator

        # e. overwrite with keyword arguments, e.g. PortfolioModelClass(Delta=0.0)
        for key,value in kwargs.items(): setattr(par,key,value)

        # f. empty container for simulation results
        self.sim = SimpleNamespace()

    def __str__(self):
        """ called when using print """

        par = self.par

        text = 'Portfolio model with:\n'
        text += f'  mu    = {par.mu:.4f}, sigma = {par.sigma:.4f}, r = {par.r:.4f}\n'
        text += f'  theta_star = {par.theta_star:.4f}, Delta = {par.Delta:.4f}, tau = {par.tau:.4f}\n'
        text += f'  gamma = {par.gamma:.4f} (relative risk aversion)\n'
        text += f'  W0 = {par.W0:.2f}, T = {par.T}, N = {par.N:,}, seed = {par.seed}'

        return text

    def draw_returns(self):
        """ draw the gross return on the risky asset in all periods and all portfolios

        Returns:

            (ndarray): gross returns with shape (N,T)

        """

        par = self.par

        rng = np.random.default_rng(par.seed)
        eps = rng.normal(size=(par.N,par.T))

        return np.exp(par.mu + par.sigma*eps)

    def u(self,W):
        """ CRRA utility of wealth """

        par = self.par

        return W**(1-par.gamma)/(1-par.gamma)

    # the share of wealth in the risky asset after trading, and the amount traded
    def trade(self,theta):
        par = self.par

        distance = np.abs(theta - par.theta_star)
        traded = distance > par.Delta
        theta_post = np.where(traded, par.theta_star, theta)
        amount_traded = np.abs(theta_post - theta)

        return theta_post, amount_traded, traded

    # simulate all N portfolios forward T periods
    def simulate(self,R=None):
        par = self.par

        if R is None:
            R = self.draw_returns()

        R_safe = np.exp(par.r)
        W = np.empty((par.N, par.T + 1))
        theta = np.empty((par.N, par.T + 1))
        traded = np.zeros((par.N, par.T), dtype=bool)

        W[:, 0] = par.W0
        theta[:, 0] = par.theta_star

        for t in range(par.T):
            theta_post, amount_traded, traded[:, t] = self.trade(theta[:, t])
            W_post = W[:, t] * (1 - par.tau * amount_traded)
            risky_wealth = theta_post * W_post * R[:, t]
            safe_wealth = (1 - theta_post) * W_post * R_safe
            W[:, t + 1] = risky_wealth + safe_wealth
            theta[:, t + 1] = risky_wealth / W[:, t + 1]

        self.sim.W = W
        self.sim.theta = theta
        self.sim.traded = traded
        self.sim.R = R

        return self.sim

    # the numbers to report for a rule, including expected utility
    def summary(self):
        par = self.par
        W = self.sim.W
        theta = self.sim.theta
        WT = W[:, -1]

        return {
            'Avg trades': self.sim.traded.sum(axis=1).mean(),
            'Avg distance': np.abs(theta[:, :-1] - par.theta_star).mean(),
            'Mean WT': WT.mean(),
            'Median WT': np.median(WT),
            'P10 WT': np.percentile(WT, 10),
            'Expected utility': self.u(WT).mean()
        }
