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
        # --- 策略邏輯：動量篩選 + 逆波動度加權 ---

        # 定義可投資資產（排除 SPY/Benchmark）
        assets = self.price.columns[self.price.columns != self.exclude]

        # 參數設定
        lookback_vol = 63   # 波動度觀察期 (約3個月)
        lookback_mom = 126  # 動量觀察期 (約6個月)

        for i in range(len(self.price)):
            # 1. 暖身期檢查
            # 必須大於最長的 lookback 才能計算指標，否則使用等權重
            if i < lookback_mom:
                equal_weight = 1.0 / len(assets) if len(assets) > 0 else 0
                self.portfolio_weights.loc[self.price.index[i], assets] = equal_weight
                self.portfolio_weights.loc[self.price.index[i], self.exclude] = 0.0
                continue

            try:
                current_date = self.price.index[i]

                # 2. 計算動量 (Momentum) - 過去 126 天的報酬率
                # 公式: (P_t / P_{t-126}) - 1
                momentum = (self.price[assets].iloc[i] / self.price[assets].iloc[i - lookback_mom]) - 1

                # 3. 計算波動度 (Volatility) - 過去 63 天的日報酬標準差 (年化)
                # 使用 replace 避免除以 0 的錯誤
                volatility = self.returns[assets].iloc[i - lookback_vol:i].std() * np.sqrt(252)
                volatility = volatility.replace(0, 0.001)

                # 4. 資產篩選 (Asset Selection)
                # 選擇動量排名前 50% 的資產
                momentum_rank = momentum.rank(ascending=False) # 1 為最高動量
                n_top = int(len(assets) * 0.5)
                # 確保至少選一個，避免全部被篩掉
                n_top = max(n_top, 1)

                selected_assets = momentum_rank[momentum_rank <= n_top].index

                # 如果篩選結果為空（極端情況），則使用全部資產
                if len(selected_assets) == 0:
                    selected_assets = assets

                # 5. 權重分配：逆波動度加權 (Inverse Volatility Weighting)
                # 波動度越低，權重越高
                inv_vol = 1.0 / volatility[selected_assets]
                total_inv_vol = inv_vol.sum()

                # 歸一化權重 (Normalize)
                target_weights = inv_vol / total_inv_vol

                # 6. 將權重寫入 portfolio_weights
                # 被選中的資產分配計算出的權重
                self.portfolio_weights.loc[current_date, selected_assets] = target_weights

                # 未被選中的資產權重設為 0
                unselected_assets = assets.difference(selected_assets)
                self.portfolio_weights.loc[current_date, unselected_assets] = 0.0

            except Exception as e:
                # 錯誤處理：若計算失敗則退回等權重
                # print(f"Error at {self.price.index[i]}: {e}") # 除錯用
                equal_weight = 1.0 / len(assets) if len(assets) > 0 else 0
                self.portfolio_weights.loc[self.price.index[i], assets] = equal_weight

            # 確保排除資產 (SPY) 權重為 0
            self.portfolio_weights.loc[self.price.index[i], self.exclude] = 0.0

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