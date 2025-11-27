"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
from gurobipy import GRB
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    # Use auto_adjust=False to match the original assignment template logic
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust=False)

    # Handle multi-level columns if yfinance returns them
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            Bdf[asset] = raw['Adj Close'][asset]
        except KeyError:
            Bdf[asset] = raw['Adj Close']
    else:
        Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation
"""

class MyPortfolio:
    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Universe excludes the benchmark
        universe = [col for col in self.price.columns if col != self.exclude]
        n_assets = len(universe)

        # Initialize weights
        self.portfolio_weights = pd.DataFrame(
            0.0, index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        # --- STRATEGY: Concentrated Momentum (Top 4) + Min Variance ---

        # Configuration
        # 126 days (6 months) is optimal for capturing Sector trends
        robust_lookback = 126
        rebalance_freq = 21  # Monthly
        n_top = 4            # Focus on the Top 4 winners (High Conviction)

        # 1. Warm-up Period: Equal Weights
        # Crucial for 2012-2013 performance. Avoids cash drag.
        if n_assets > 0:
            self.portfolio_weights.loc[self.price.index[:robust_lookback], universe] = 1.0 / n_assets

        # 2. Rolling Optimization
        for i in range(robust_lookback, len(self.price), rebalance_freq):
            current_date = self.price.index[i]

            # Data Window
            window_returns = self.returns[universe].iloc[i - robust_lookback : i]

            # A. Filter for Valid Assets (Remove assets with 0 variance, e.g., XLC pre-2018)
            # If std is 0, the optimizer might dump 100% there, which is a data artifact.
            valid_assets = [a for a in universe if window_returns[a].std() > 1e-5]
            if not valid_assets:
                continue

            # B. Calculate Momentum (Cumulative Return)
            momentum = (1 + window_returns[valid_assets]).prod() - 1

            # Check Regime: Are we in a broad Bull or Bear market?
            # If fewer than 2 assets are positive, play Defense (GMV on all valid).
            # Otherwise, play Offense (Top 4).
            positive_momentum_assets = momentum[momentum > 0].index.tolist()

            if len(positive_momentum_assets) >= 2:
                # --- OFFENSE: Min-Var on Top 4 Winners ---
                # Select Top 4 from the valid assets
                selected_assets = momentum.nlargest(n_top).index.tolist()
            else:
                # --- DEFENSE: GMV on All Valid Assets ---
                selected_assets = valid_assets

            n_sel = len(selected_assets)

            # C. Optimization
            # Calculate covariance for selected assets
            subset_returns = window_returns[selected_assets]
            Sigma = subset_returns.cov().values
            Sigma += 1e-6 * np.eye(n_sel) # Regularization

            try:
                m = gp.Model("MomMinVar")
                m.setParam("OutputFlag", 0)

                # Weights
                w = m.addVars(n_sel, lb=0.0, ub=1.0, name="w")

                # Fully Invested
                m.addConstr(gp.quicksum(w[j] for j in range(n_sel)) == 1, "Budget")

                # Minimize Variance
                p_var = gp.QuadExpr()
                for r in range(n_sel):
                    for c in range(n_sel):
                        p_var += w[r] * w[c] * Sigma[r][c]

                m.setObjective(p_var, GRB.MINIMIZE)

                m.optimize()

                if m.status == GRB.OPTIMAL:
                    opt_w = [w[j].X for j in range(n_sel)]
                    self.portfolio_weights.loc[current_date, selected_assets] = opt_w
                else:
                    # Fallback
                    self.portfolio_weights.loc[current_date, selected_assets] = 1.0 / n_sel

            except Exception:
                self.portfolio_weights.loc[current_date, selected_assets] = 1.0 / n_sel

        """
        TODO: Complete Task 4 Above
        """

        # 3. Fill Weights
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    from grader_2 import AssignmentJudge

    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )
    parser.add_argument("--score", action="append", help="Score for assignment")
    parser.add_argument("--allocation", action="append", help="Allocation for asset")
    parser.add_argument("--performance", action="append", help="Performance for portfolio")
    parser.add_argument("--report", action="append", help="Report for evaluation metric")
    parser.add_argument("--cumulative", action="append", help="Cumulative product result")

    args = parser.parse_args()
    judge = AssignmentJudge()
    judge.run_grading(args)