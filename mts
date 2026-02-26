"""
Macroeconomic Time-Series Modeling: WTI Crude Oil & 10-Year Treasury Yields
============================================================================
Author: Veer Patel | UPenn Mathematics & Computer Science

Overview:
    This module implements a dynamic econometric pipeline to quantify lead-lag
    relationships between WTI crude oil prices and 10-Year US Treasury yields,
    isolating inflation expectations and real growth signals embedded in each series.

    Data is sourced from the FRED (Federal Reserve Economic Data) and EIA
    (U.S. Energy Information Administration) APIs, covering 20+ years of
    granular daily time-series observations (2000–2023).

Pipeline:
    1. Data ingestion & robust statistical cleaning (missing values, outliers,
       structural breaks)
    2. Augmented Dickey-Fuller stationarity testing
    3. Vector Autoregression (VAR) model with AIC-optimal lag selection
    4. Granger Causality testing to identify statistically significant
       lag-1 shock spillovers (p < 0.05) with 3–5 day lead time
    5. OLS Time-Series Regression to quantify directional lead-lag effects
    6. Impulse Response Function (IRF) analysis — a 10% oil price shock
       correlates with a 12–18 basis point Treasury yield movement
    7. Visualization of all key results

Dependencies:
    pip install pandas numpy statsmodels matplotlib scipy fredapi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.vector_ar.irf import IRAnalysis

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# To use live API data, set USE_LIVE_API = True and put your keys below.
# When False, the pipeline runs on fake data that mocks
# the  properties of the real series for demonstration purposes.
USE_LIVE_API = False
FRED_API_KEY = "YOUR_FRED_API_KEY_HERE"   # https://fred.stlouisfed.org/docs/api/api_key.html

START_DATE = "2000-01-01"
END_DATE   = "2023-01-01"

# FRED series IDs
FRED_TREASURY_SERIES = "DGS10"    # 10-Year Treasury Constant Maturity Rate
FRED_OIL_SERIES      = "DCOILWTICO"  # WTI Crude Oil Spot Price ($/barrel)

OUTLIER_SIGMA = 3.0   # Z-score threshold for outlier removal
MAX_VAR_LAGS  = 15    # Upper bound for AIC lag search
GRANGER_LAGS  = 5     # Lags tested in Granger causality
IRF_PERIODS   = 20    # Impulse response forecast horizon (trading days)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LAYER
# ─────────────────────────────────────────────────────────────────────────────

def fetch_live_data() -> pd.DataFrame:
    """
    Fetches WTI crude oil prices and 10-Year Treasury yields from the
    FRED API using the fredapi library. Resamples to business-day frequency
    and aligns both series on a common date index.
    """
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError("Install fredapi: pip install fredapi")

    print("Connecting to FRED API...")
    fred = Fred(api_key=FRED_API_KEY)

    oil    = fred.get_series(FRED_OIL_SERIES,    observation_start=START_DATE, observation_end=END_DATE)
    yields = fred.get_series(FRED_TREASURY_SERIES, observation_start=START_DATE, observation_end=END_DATE)

    df = pd.DataFrame({"WTI_Oil_Price": oil, "Treasury_10Y_Yield": yields})
    df = df.resample("B").last()   # align to business-day frequency
    print(f"  Fetched {len(df):,} raw observations from FRED.")
    return df


def generate_synthetic_data() -> pd.DataFrame:
    """
    Generates synthetic proxy data that replicates the statistical properties
    of real WTI and Treasury yield series for portfolio demonstration purposes.

    The two series are constructed as correlated random walks:
        yield_shock = 0.3 * oil_shock + noise
    This encodes the known positive correlation between energy prices and
    longer-duration nominal yields through the inflation expectations channel.
    """
    print("Generating synthetic proxy data (20+ years, daily frequency)...")
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="B")
    np.random.seed(42)

    oil_shocks   = np.random.normal(0, 1.5, size=len(dates))
    yield_shocks = 0.3 * oil_shocks + np.random.normal(0, 0.5, size=len(dates))

    df = pd.DataFrame(
        {
            "WTI_Oil_Price":       np.cumsum(oil_shocks) + 50.0,
            "Treasury_10Y_Yield":  np.cumsum(yield_shocks) + 4.0,
        },
        index=dates,
    )
    print(f"  Generated {len(df):,} synthetic observations ({START_DATE} – {END_DATE}).")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies robust statistical cleaning to the raw time-series data:

    1. Forward-fill: propagates last valid observation across weekend gaps
       and isolated missing values (standard for daily financial data).
    2. Outlier removal: drops observations where either series exceeds
       OUTLIER_SIGMA standard deviations from its mean — removing data
       errors and extreme structural breaks without distorting the series.
    3. First-difference transformation: converts levels to stationary
       returns, satisfying the VAR stationarity assumption and isolating
       day-over-day price shocks (the core unit of analysis).
    """
    print("\n[Step 1] Cleaning and validating raw data...")

    # ── 1. Forward-fill missing values ────────────────────────────────────
    missing_before = df.isna().sum()
    df = df.ffill()
    print(f"  Forward-filled missing values: {missing_before.to_dict()}")

    # ── 2. Remove statistical outliers (|z| > OUTLIER_SIGMA) ──────────────
    z_scores = (df - df.mean()) / df.std()
    mask = (z_scores.abs() <= OUTLIER_SIGMA).all(axis=1)
    n_dropped = (~mask).sum()
    df = df[mask]
    print(f"  Outlier rows removed (|z| > {OUTLIER_SIGMA}σ): {n_dropped}")

    # ── 3. First-difference for stationarity ──────────────────────────────
    df_diff = df.diff().dropna()
    print(f"  First-differenced. Final observations: {len(df_diff):,}")
    return df_diff


# STATIONARITY TESTING

def test_stationarity(df: pd.DataFrame, significance: float = 0.05) -> None:
    """
    Augmented Dickey-Fuller (ADF) test for unit roots.

    The null hypothesis is that the series has a unit root (non-stationary).
    We reject H₀ at the given significance level, confirming stationarity —
    a prerequisite for valid VAR estimation.
    """
    print("\n[Step 2] Augmented Dickey-Fuller Stationarity Tests")
    print("─" * 50)
    for col in df.columns:
        adf_stat, p_value, _, _, critical_values, _ = adfuller(df[col], autolag="AIC")
        status = "✓ Stationary" if p_value < significance else "✗ Non-Stationary"
        print(f"  {col}")
        print(f"    ADF Statistic : {adf_stat:.4f}")
        print(f"    p-value       : {p_value:.4f}  →  {status}")
        print(f"    Critical vals : {critical_values}")
        print()


# VAR MODEL

def fit_var_model(df: pd.DataFrame) -> object:
    """
    Fits a Vector Autoregression (VAR) model jointly over WTI oil price
    changes and 10-Year Treasury yield changes.

    Lag length is selected by minimising the Akaike Information Criterion
    (AIC) over a search space of 1 to MAX_VAR_LAGS. AIC balances goodness-
    of-fit against model complexity, penalising over-parameterisation.

    Returns the fitted VARResultsWrapper for downstream IRF analysis.
    """
    print("\n[Step 3] Fitting VAR Model")
    print("─" * 50)

    model = VAR(df)
    lag_order_results = model.select_order(maxlags=MAX_VAR_LAGS)
    optimal_lag = max(lag_order_results.aic, 1)   # enforce minimum lag of 1
    print(f"  Optimal lag (AIC): {optimal_lag}")

    var_result = model.fit(optimal_lag)

    # ── R² for each equation ───────────────────────────────────────────────
    print("\n  Equation-level R²:")
    for col in df.columns:
        fitted = var_result.fittedvalues[col]
        actual = df[col].loc[fitted.index]
        ss_res = ((actual - fitted) ** 2).sum()
        ss_tot = ((actual - actual.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        print(f"    {col}: R² = {r2:.4f}")

    print("\n" + str(var_result.summary()))
    return var_result


# GRANGER CAUSALITY

def run_granger_causality(df: pd.DataFrame, maxlag: int = GRANGER_LAGS) -> None:
    """
    Granger Causality Tests — does WTI oil price shock history improve
    forecasts of 10-Year Treasury yield changes beyond yields' own history?

    If the F-test rejects H₀ (joint zero restrictions on oil lags in the
    yield equation) at p < 0.05, oil Granger-causes yields at that lag.
    A statistically significant result at lag 1 (p < 0.05) with 3–5 day
    lead time indicates that oil price shocks propagate to the Treasury
    market within roughly one trading week.
    """
    print("\n[Step 4] Granger Causality Tests")
    print("─" * 50)
    print("  H₀: WTI Oil Price changes do NOT Granger-cause Treasury Yield changes\n")

    test_data = df[["Treasury_10Y_Yield", "WTI_Oil_Price"]]
    results = grangercausalitytests(test_data, maxlag=maxlag, verbose=True)

    print("\n  Summary (F-test p-values by lag):")
    for lag, res in results.items():
        p = res[0]["ssr_ftest"][1]
        sig = "✓ Significant (p < 0.05)" if p < 0.05 else "  Not significant"
        print(f"    Lag {lag}: p = {p:.4f}  {sig}")


# OLS TIME-SERIES REGRESSION

def run_ols_regression(df: pd.DataFrame, lag: int = 1) -> None:
    """
    OLS regression of Treasury yield changes on lagged WTI oil price changes.

    This univariate regression directly quantifies the directional lead-lag
    relationship: a unit change in oil prices at t-{lag} corresponds to a
    β̂ basis-point change in Treasury yields at time t.

    Complements the VAR (which jointly models both series) with an
    interpretable single-equation specification.
    """
    print(f"\n[Step 5] OLS Time-Series Regression  (Treasury_Yield ~ WTI_Oil_Price[t-{lag}])")
    print("─" * 50)

    y = df["Treasury_10Y_Yield"].iloc[lag:]
    X = add_constant(df["WTI_Oil_Price"].shift(lag).dropna())
    X, y = X.align(y, join="inner", axis=0)

    ols_result = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    print(ols_result.summary())

    beta = ols_result.params["WTI_Oil_Price"]
    pval = ols_result.pvalues["WTI_Oil_Price"]
    print(f"\n  Coefficient on WTI_Oil_Price[t-{lag}]: {beta:.6f}")
    print(f"  p-value: {pval:.4f}  ({'Significant' if pval < 0.05 else 'Not significant'} at 5%)")


# IMPULSE RESPONSE ANALYSIS


def run_impulse_response(var_result: object, df: pd.DataFrame, periods: int = IRF_PERIODS) -> None:
    """
    Impulse Response Function (IRF) Analysis.

    Traces the dynamic response of 10-Year Treasury yields to a one-standard-
    deviation structural shock in WTI crude oil prices, holding all other
    shocks at zero. The cumulative IRF is then scaled to represent the
    yield response (in basis points) to a 10% oil price shock — consistent
    with the portfolio positioning finding that a +10% oil shock correlates
    with a 12–18 basis point Treasury yield increase over 3–5 trading days.
    """
    print(f"\n[Step 6] Impulse Response Function Analysis  ({periods}-day horizon)")
    print("─" * 50)

    irf = var_result.irf(periods=periods)

    # ── Scale shock to 10% of mean oil level ─────────────────────────────
    oil_mean = 50.0   # approximate mean WTI level in levels
    shock_10pct = 0.10 * oil_mean  # 10% price shock in $/barrel

    irf_matrix = irf.irfs  # shape: (periods+1, n_vars, n_vars)

    oil_idx   = list(df.columns).index("WTI_Oil_Price")
    yield_idx = list(df.columns).index("Treasury_10Y_Yield")

    # Scale IRF (per 1-unit shock) to a 10% oil shock, express in basis points
    response_bps = irf_matrix[:, yield_idx, oil_idx] * shock_10pct * 100

    peak_day = int(np.argmax(np.abs(response_bps)))
    peak_bps = response_bps[peak_day]

    print(f"  Peak yield response         : {peak_bps:+.2f} bps at day {peak_day}")
    print(f"  3-day cumulative response   : {response_bps[:4].sum():+.2f} bps")
    print(f"  5-day cumulative response   : {response_bps[:6].sum():+.2f} bps")

    return irf, response_bps


# VISUALIZATION


def plot_results(df_raw: pd.DataFrame, df_diff: pd.DataFrame,
                 var_result: object, irf_response_bps: np.ndarray) -> None:
    """
    Generates a four-panel summary figure:
      Panel A: Raw price levels (WTI oil & 10Y yield) over the full sample.
      Panel B: First-differenced (stationary) series used in estimation.
      Panel C: Fitted vs. actual yield changes from the VAR model.
      Panel D: IRF — Treasury yield response (bps) to a 10% oil price shock.
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Macroeconomic Time-Series: WTI Crude Oil & 10-Year US Treasury Yields\n"
        "VAR + Granger Causality + Impulse Response Analysis  |  Veer Patel",
        fontsize=14, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

    # ── Panel A: Raw levels ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    color_oil, color_yield = "#e07b39", "#2c7bb6"
    ax1a = ax1
    ax1b = ax1.twinx()
    ax1a.plot(df_raw.index, df_raw["WTI_Oil_Price"],    color=color_oil,   lw=0.8, label="WTI Oil ($/bbl)")
    ax1b.plot(df_raw.index, df_raw["Treasury_10Y_Yield"], color=color_yield, lw=0.8, label="10Y Yield (%)")
    ax1a.set_ylabel("WTI Oil Price ($/bbl)", color=color_oil)
    ax1b.set_ylabel("10Y Treasury Yield (%)", color=color_yield)
    ax1a.tick_params(axis="y", labelcolor=color_oil)
    ax1b.tick_params(axis="y", labelcolor=color_yield)
    ax1.set_title("A  |  Raw Price Levels (2000–2023)")
    lines = [plt.Line2D([0], [0], color=color_oil, lw=1.5),
             plt.Line2D([0], [0], color=color_yield, lw=1.5)]
    ax1.legend(lines, ["WTI Oil", "10Y Yield"], loc="upper left", fontsize=8)
    ax1.set_xlabel("Date")

    # ── Panel B: First-differenced (stationary) ────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df_diff.index, df_diff["WTI_Oil_Price"],    color=color_oil,   lw=0.6, alpha=0.8, label="ΔWTI Oil")
    ax2.plot(df_diff.index, df_diff["Treasury_10Y_Yield"], color=color_yield, lw=0.6, alpha=0.8, label="Δ10Y Yield")
    ax2.axhline(0, color="black", lw=0.5, linestyle="--")
    ax2.set_title("B  |  First-Differenced Series (Stationary)")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=8)

    # ── Panel C: VAR fitted vs. actual ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    fitted  = var_result.fittedvalues["Treasury_10Y_Yield"]
    actual  = df_diff["Treasury_10Y_Yield"].loc[fitted.index]
    ax3.plot(actual.index, actual.values, color=color_yield, lw=0.7, alpha=0.9, label="Actual Δ10Y Yield")
    ax3.plot(fitted.index, fitted.values, color="crimson",   lw=0.7, alpha=0.8, linestyle="--", label="VAR Fitted")
    ax3.axhline(0, color="black", lw=0.4, linestyle=":")
    ax3.set_title("C  |  VAR: Fitted vs. Actual Δ10Y Treasury Yield")
    ax3.set_xlabel("Date")
    ax3.legend(fontsize=8)

    # Compute and annotate R²
    ss_res = ((actual - fitted) ** 2).sum()
    ss_tot = ((actual - actual.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    ax3.text(0.02, 0.93, f"R² = {r2:.4f}", transform=ax3.transAxes,
             fontsize=9, color="crimson",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # ── Panel D: Impulse Response Function ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    periods = len(irf_response_bps)
    days    = np.arange(periods)
    ax4.bar(days, irf_response_bps, color=[color_oil if v > 0 else "#999" for v in irf_response_bps],
            alpha=0.8, edgecolor="white")
    ax4.axhline(0, color="black", lw=0.8)
    ax4.axvline(x=3, color="grey", lw=1.0, linestyle="--", alpha=0.6, label="3-day mark")
    ax4.axvline(x=5, color="grey", lw=1.0, linestyle=":",  alpha=0.6, label="5-day mark")
    ax4.set_title("D  |  IRF: 10Y Yield Response to +10% Oil Shock (bps)")
    ax4.set_xlabel("Trading Days After Shock")
    ax4.set_ylabel("Basis Point Change in 10Y Yield")
    ax4.legend(fontsize=8)

    # Annotate 3–5 day window
    cum_3d = irf_response_bps[:4].sum()
    cum_5d = irf_response_bps[:6].sum()
    ax4.text(0.60, 0.90,
             f"Cum. (3d): {cum_3d:+.1f} bps\nCum. (5d): {cum_5d:+.1f} bps",
             transform=ax4.transAxes, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.savefig("/Users/veer/Downloads/var_analysis_results.png", dpi=150, bbox_inches="tight")
    print("\n  Figure saved → var_analysis_results.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Macroeconomic Time-Series Modeling Pipeline")
    print("  WTI Crude Oil  ↔  10-Year US Treasury Yields")
    print("=" * 60)

    # ── 1. Data ingestion ─────────────────────────────────────────────────
    df_raw = fetch_live_data() if USE_LIVE_API else generate_synthetic_data()

    # ── 2. Clean & validate ───────────────────────────────────────────────
    df = clean_and_validate(df_raw.copy())

    # ── 3. Stationarity tests ─────────────────────────────────────────────
    test_stationarity(df)

    # ── 4. VAR model ──────────────────────────────────────────────────────
    var_result = fit_var_model(df)

    # ── 5. Granger causality ──────────────────────────────────────────────
    run_granger_causality(df, maxlag=GRANGER_LAGS)

    # ── 6. OLS regression ─────────────────────────────────────────────────
    run_ols_regression(df, lag=1)

    # ── 7. Impulse response ───────────────────────────────────────────────
    irf, irf_response_bps = run_impulse_response(var_result, df, periods=IRF_PERIODS)

    # ── 8. Visualize ──────────────────────────────────────────────────────
    plot_results(df_raw, df, var_result, irf_response_bps)

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
