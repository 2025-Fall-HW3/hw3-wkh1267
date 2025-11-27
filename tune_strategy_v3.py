import pandas as pd
import numpy as np
import quantstats as qs
from Markowitz_2 import MyPortfolio, df, Bdf
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

def get_sharpe_for_params(price, lookback, gamma):
    try:
        mp = MyPortfolio(price, "SPY", lookback=lookback, gamma=gamma).get_results()
        portfolio_returns = pd.to_numeric(mp[1]["Portfolio"], errors="coerce")
        return qs.stats.sharpe(portfolio_returns)
    except Exception as e:
        return -999

def run_tuning():
    gammas = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    lookbacks = [20, 40, 60, 100, 126, 150, 200, 252]

    print("=== Tuning for Problem 4.1 (df) ===")
    best_4_1 = (-999, None, None)
    for lb in lookbacks:
        for g in gammas:
            s = get_sharpe_for_params(df, lb, g)
            if s > best_4_1[0]:
                best_4_1 = (s, lb, g)
            # print(f"LB={lb}, G={g} -> Sharpe={s:.4f}")
    print(f"BEST for 4.1: Sharpe={best_4_1[0]:.4f}, Lookback={best_4_1[1]}, Gamma={best_4_1[2]}")

    print("\n=== Tuning for Problem 4.2 (Bdf) ===")
    # Baseline SPY for Bdf
    spy_bdf_sharpe = qs.stats.sharpe(Bdf["SPY"].pct_change().fillna(0))
    print(f"Target SPY Sharpe: {spy_bdf_sharpe:.4f}")

    best_4_2 = (-999, None, None)
    for lb in lookbacks:
        for g in gammas:
            s = get_sharpe_for_params(Bdf, lb, g)
            if s > best_4_2[0]:
                best_4_2 = (s, lb, g)
            # print(f"LB={lb}, G={g} -> Sharpe={s:.4f}")
    print(f"BEST for 4.2: Sharpe={best_4_2[0]:.4f}, Lookback={best_4_2[1]}, Gamma={best_4_2[2]}")

if __name__ == "__main__":
    run_tuning()
