import pandas as pd
import numpy as np
import quantstats as qs
from Markowitz_2 import df, Bdf
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

def get_sharpe(returns):
    return qs.stats.sharpe(returns)

def test_eqw(price, exclude):
    assets = price.columns[price.columns != exclude]
    returns = price.pct_change().fillna(0)
    n = len(assets)
    # Equal weights
    portfolio_returns = returns[assets].mean(axis=1)
    return get_sharpe(portfolio_returns)

def test_rp(price, exclude, lookback=50):
    assets = price.columns[price.columns != exclude]
    returns = price.pct_change().fillna(0)

    # Rolling volatility
    # Note: Using the fix from Problem 2 (iloc[1:]) for correctness,
    # though for Sharpe it might not matter much if the first day is 0.
    vol = returns[assets].iloc[1:].rolling(window=lookback).std()

    # Realign to original index
    vol = vol.reindex(returns.index)

    vol = vol.shift(1) # Shift to use past data

    # Inverse vol
    inv_vol = 1.0 / vol
    inv_vol = inv_vol.replace([np.inf, -np.inf], 0).fillna(0)

    sum_inv_vol = inv_vol.sum(axis=1)
    weights = inv_vol.div(sum_inv_vol, axis=0).fillna(0)

    portfolio_returns = (returns[assets] * weights).sum(axis=1)
    return get_sharpe(portfolio_returns)

def run_tests():
    print("--- Baselines ---")
    spy_df = df["SPY"].pct_change().fillna(0)
    spy_bdf = Bdf["SPY"].pct_change().fillna(0)

    print(f"SPY Sharpe (df - 2019+): {get_sharpe(spy_df):.4f}")
    print(f"SPY Sharpe (Bdf - 2012+): {get_sharpe(spy_bdf):.4f}")

    print("\n--- Equal Weight ---")
    eqw_df = test_eqw(df, "SPY")
    eqw_bdf = test_eqw(Bdf, "SPY")
    print(f"EQW Sharpe (df): {eqw_df:.4f}")
    print(f"EQW Sharpe (Bdf): {eqw_bdf:.4f}")

    print("\n--- Risk Parity ---")
    for lb in [252]:
        rp_df = test_rp(df, "SPY", lb)
        rp_bdf = test_rp(Bdf, "SPY", lb)
        print(f"RP (lb={lb}) Sharpe (df): {rp_df:.4f}")
        print(f"RP (lb={lb}) Sharpe (Bdf): {rp_bdf:.4f}")

if __name__ == "__main__":
    run_tests()
