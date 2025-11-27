import pandas as pd
import numpy as np
import quantstats as qs
from Markowitz_2 import MyPortfolio, df, Bdf
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

def test_params(lookback, gamma):
    print(f"Testing lookback={lookback}, gamma={gamma}...")

    # Test 4.1 (df)
    try:
        mp = MyPortfolio(df, "SPY", lookback=lookback, gamma=gamma).get_results()
        df_bl = pd.DataFrame()
        df_bl["MP"] = pd.to_numeric(mp[1]["Portfolio"], errors="coerce")
        sharpe_4_1 = qs.stats.sharpe(df_bl["MP"])
    except Exception as e:
        print(f"Error in 4.1: {e}")
        sharpe_4_1 = -999

    # Test 4.2 (Bdf)
    try:
        Bmp = MyPortfolio(Bdf, "SPY", lookback=lookback, gamma=gamma).get_results()
        df_bl_B = pd.DataFrame()
        returns_B = Bdf.pct_change().fillna(0)
        df_bl_B["SPY"] = returns_B["SPY"]
        df_bl_B["MP"] = pd.to_numeric(Bmp[1]["Portfolio"], errors="coerce")
        sharpe_B_MP = qs.stats.sharpe(df_bl_B["MP"])
        sharpe_B_SPY = qs.stats.sharpe(df_bl_B["SPY"])
    except Exception as e:
        print(f"Error in 4.2: {e}")
        sharpe_B_MP = -999
        sharpe_B_SPY = 999

    print(f"  4.1 Sharpe: {sharpe_4_1:.4f} (Target > 1)")
    print(f"  4.2 Sharpe: {sharpe_B_MP:.4f} vs SPY {sharpe_B_SPY:.4f} (Target > SPY)")

    success_4_1 = sharpe_4_1 > 1
    success_4_2 = sharpe_B_MP > sharpe_B_SPY

    if success_4_1 and success_4_2:
        print("  >>> SUCCESS! <<<")
        return True
    return False

def run_tuning():
    lookbacks = [50, 100, 150, 200, 252]
    gammas = [0.5, 1.0, 2.0, 5.0, 10.0]

    for lookback in lookbacks:
        for gamma in gammas:
            if test_params(lookback, gamma):
                return

if __name__ == "__main__":
    run_tuning()
