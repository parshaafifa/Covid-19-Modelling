"""
COVID-19 Epidemiological Analysis: Bangladesh, India, Pakistan
----------------------------------------------------------------
Rt via Cori/EpiEstim renewal equation, per-wave SEIR fitting with
population-scaled compartments, underreporting-adjusted attack rates,
doubling-time trajectories, and cross-country transmission lag analysis.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.signal import find_peaks, correlate
from scipy.stats import lognorm

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUTPUT_DIR = "seir_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {"Bangladesh": "#1f78b4", "India": "#e31a1c", "Pakistan": "#33a02c"}
LIGHT = {"Bangladesh": "#a6cee3", "India": "#fb9a99", "Pakistan": "#b2df8a"}

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

POP: dict[str, int] = {"Bangladesh": 171_000_000, "India": 1_417_000_000, "Pakistan": 241_000_000}
COUNTRIES = list(POP.keys())

# Reporting-rate correction factors. Confirmed-case counts in South Asia are
# known to badly undercount true infections -- seroprevalence surveys during
# 2020-21 (Nomani et al 2021 for BD, ICMR rounds 1-4 for India, Nasir et al
# for PK) put the true-to-reported ratio somewhere around 15-30x depending on
# testing capacity at the time. These are rough, single-number stand-ins, not
# a time-varying correction -- good enough to compare relative underreporting
# across countries, not to be quoted as precise.
UNDERREPORT_FACTOR = {"Bangladesh": 25.0, "India": 20.0, "Pakistan": 18.0}

# serial interval (Nishiura 2020 / Bi 2020), incubation period (Lauer 2020)
SI_MU, SI_SIGMA = 5.1, 2.6
INCUBATION_DAYS = 5.2          # mean latent period, used for SEIR sigma
GAMMA = 1 / 7                  # recovery rate, ~7d infectious period
SIGMA_E = 1 / INCUBATION_DAYS  # E -> I rate


@dataclass
class Wave:
    country: str
    index: int
    start: int
    peak: int
    end: int
    r0: float = np.nan
    rt_peak: float = np.nan
    peak_cases: float = np.nan
    fitted: np.ndarray = field(default_factory=lambda: np.array([]))
    attack_rate_reported: float = np.nan
    attack_rate_adjusted: float = np.nan
    doubling_time_early: float = np.nan

    @property
    def label(self) -> str:
        return f"{self.country[:3]} W{self.index}"


# =============================================================================
# DATA LOADING
# =============================================================================
JHU_URL = ("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
           "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")


def load_jhu_data(url: str = JHU_URL) -> pd.DataFrame:
    log.info("Fetching JHU CSSE time-series...")
    try:
        df = pd.read_csv(url)
    except Exception as exc:
        log.error("Data fetch failed: %s", exc)
        sys.exit(1)
    required = {"Country/Region", "Province/State", "Lat", "Long"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected JHU schema, missing columns: {missing}")
    log.info("Loaded %d rows, %d date columns", len(df), df.shape[1] - 4)
    return df


def extract_daily(df: pd.DataFrame, country: str, smooth: int = 7) -> pd.Series:
    subset = df[df["Country/Region"] == country]
    if subset.empty:
        raise KeyError(f"Country not found in JHU data: {country!r}")

    cumulative = subset.drop(columns=["Province/State", "Country/Region", "Lat", "Long"]).sum()
    cumulative.index = pd.to_datetime(cumulative.index, format="%m/%d/%y")
    daily = cumulative.diff().clip(lower=0).fillna(0)

    rolling_mean = daily.rolling(7, min_periods=1).mean().shift(1)
    spikes = daily[daily > rolling_mean * 10]
    if not spikes.empty:
        log.warning("%s: %d reporting spike(s) at %s", country, len(spikes),
                    ", ".join(spikes.index.strftime("%Y-%m-%d")))

    return daily.rolling(smooth, center=True, min_periods=1).mean()


# =============================================================================
# SERIAL INTERVAL
# =============================================================================
def build_serial_interval(max_s: int = 30, mu: float = SI_MU, sigma: float = SI_SIGMA) -> np.ndarray:
    s = np.arange(1, max_s + 1, dtype=float)
    log_mu = np.log(mu**2 / np.sqrt(mu**2 + sigma**2))
    log_sigma = np.sqrt(np.log(1 + sigma**2 / mu**2))
    w = (lognorm.cdf(s + 0.5, s=log_sigma, scale=np.exp(log_mu))
         - lognorm.cdf(np.maximum(s - 0.5, 0), s=log_sigma, scale=np.exp(log_mu)))
    return w / w.sum()


SERIAL_INTERVAL = build_serial_interval()


# =============================================================================
# Rt ESTIMATION (Cori / EpiEstim) with bootstrap CI
# =============================================================================
def estimate_rt(cases: np.ndarray, w: np.ndarray = SERIAL_INTERVAL, window: int = 7,
                 smooth_sigma: float = 14, clip: tuple[float, float] = (0.1, 5.0)) -> np.ndarray:
    n = len(cases)
    rt = np.full(n, np.nan)
    for t in range(len(w), n):
        numerator = cases[max(0, t - window + 1): t + 1].mean()
        denominator = float(np.dot(w[:min(t, len(w))], cases[t - min(t, len(w)): t][::-1]))
        if denominator > 1.0:
            rt[t] = numerator / denominator
    rt = np.clip(rt, *clip)
    rt = pd.Series(rt).ffill().bfill().values
    return gaussian_filter1d(rt, sigma=smooth_sigma)


def bootstrap_rt_ci(cases: np.ndarray, n_boot: int = 200, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """
    Poisson-resample daily incidence (cases are noisy counts, this treats
    the smoothed series as the underlying rate) and re-run Rt each time to
    get a rough 95% band. Not a full Bayesian posterior like EpiEstim proper,
    but gives a decent sense of how much noise is in the Rt curve.
    """
    rng = np.random.default_rng(seed)
    n = len(cases)
    boot_rt = np.empty((n_boot, n))
    for b in range(n_boot):
        resampled = rng.poisson(lam=np.maximum(cases, 0.01))
        boot_rt[b] = estimate_rt(resampled.astype(float))
    lo = np.percentile(boot_rt, 2.5, axis=0)
    hi = np.percentile(boot_rt, 97.5, axis=0)
    return lo, hi


def doubling_time(cases: np.ndarray, window: int = 7) -> np.ndarray:
    """Local doubling time in days from the log-slope of a rolling window. Inf/negatives (declining epidemic) clipped to NaN for plotting."""
    log_cases = np.log(np.maximum(cases, 1))
    n = len(cases)
    dt = np.full(n, np.nan)
    for t in range(window, n):
        slope = (log_cases[t] - log_cases[t - window]) / window
        if slope > 1e-4:
            dt[t] = np.log(2) / slope
    dt[(dt <= 0) | (dt > 120)] = np.nan
    return dt


# =============================================================================
# SEIR MODEL -- fractional-population form (numerically safe for N up to ~1.4B)
# =============================================================================
def run_seir(N: int, I0: float, beta: float, sigma: float = SIGMA_E, gamma: float = GAMMA,
             n_days: int = 365) -> np.ndarray:
    """
    Euler-integrated SEIR, tracked as fractions of N (s, e, i, r) rather than
    raw head counts -- keeps floating point sane at India-scale populations
    and makes beta comparable across countries of very different size.
    Returns daily new infections (E inflow) as a head-count array.
    """
    s, e, i, r = 1 - I0 / N, 0.0, I0 / N, 0.0
    new_infections = np.empty(n_days)
    for t in range(n_days):
        new_e = beta * s * i
        new_i = sigma * e
        new_r = gamma * i

        new_infections[t] = new_e * N

        s -= new_e
        e += new_e - new_i
        i += new_i - new_r
        r += new_r
        s, e, i, r = max(s, 0), max(e, 0), max(i, 0), max(r, 0)
    return new_infections


def fit_seir_wave(obs: np.ndarray, N: int, sigma: float = SIGMA_E, gamma: float = GAMMA) -> tuple[np.ndarray, float]:
    """
    Fit beta (and implicitly R0 = beta/gamma) to one wave by minimising
    relative RMSE between the scaled model curve and observed smoothed cases.
    Reporting-rate / amplitude scaling is solved analytically (ratio of
    peaks) rather than as a free parameter -- one less dimension for the
    optimiser to wander around in.
    """
    n = len(obs)
    peak_obs = obs.max()
    if peak_obs < 10 or n < 14:
        return np.ones(n) * obs.mean(), np.nan

    I0 = max(1, int(N / 1_000_000))

    def loss(params) -> float:
        log_beta = params[0]
        beta = np.exp(log_beta)
        r0 = beta / gamma
        if not (0.5 <= r0 <= 12):
            return 1e9
        pred = run_seir(N, I0, beta, sigma, gamma, n)
        peak_pred = pred.max()
        if peak_pred < 1e-6:
            return 1e9
        scale = peak_obs / peak_pred
        rel_err = ((pred * scale - obs) / (obs + peak_obs * 0.05 + 1)) ** 2
        return float(np.sqrt(rel_err.mean()))

    result = minimize(loss, x0=[np.log(0.3)], method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-6})
    beta_hat = np.exp(result.x[0])
    pred = run_seir(N, I0, beta_hat, sigma, gamma, n)
    scale = peak_obs / (pred.max() + 1e-9)
    return pred * scale, beta_hat / gamma


# =============================================================================
# WAVE DETECTION
# =============================================================================
def detect_waves(cases: np.ndarray, min_distance: int = 45, peak_height_abs: float = 200.0,
                  prominence_frac: float = 0.06, valley_depth_frac: float = 0.40) -> list[tuple[int, int, int]]:
    global_max = cases.max()
    if global_max < 1:
        return []

    peak_idx, _ = find_peaks(cases, height=peak_height_abs, distance=min_distance,
                              prominence=global_max * prominence_frac)
    if len(peak_idx) == 0:
        return []

    merged: list[int] = [int(peak_idx[0])]
    for pi in peak_idx[1:]:
        prev = merged[-1]
        valley_val = float(cases[prev: pi + 1].min())
        threshold = valley_depth_frac * min(float(cases[prev]), float(cases[pi]))
        if valley_val <= threshold:
            merged.append(int(pi))
        elif cases[pi] > cases[prev]:
            merged[-1] = int(pi)

    n = len(cases)
    starts, ends = [], []
    for k, pi in enumerate(merged):
        if k == 0:
            left = pi
            while left > 0 and cases[left - 1] >= cases[pi] * 0.15:
                left -= 1
            starts.append(left)
        else:
            starts.append(ends[-1] + 1)

        if k == len(merged) - 1:
            right = pi
            while right < n - 1 and cases[right + 1] >= cases[pi] * 0.15:
                right += 1
            ends.append(right)
        else:
            next_pi = merged[k + 1]
            valley_pos = int(pi + int(cases[pi: next_pi + 1].argmin()))
            ends.append(valley_pos)

    return list(zip(starts, merged, ends))


# =============================================================================
# CROSS-COUNTRY LAG ANALYSIS
# =============================================================================
def cross_country_lag(daily: dict[str, pd.Series], max_lag: int = 60) -> pd.DataFrame:
    """
    Normalised cross-correlation between each country pair's case curves,
    on a common date index, to see which country's waves tend to lead or
    lag the others. Positive lag = first country in the pair leads.
    """
    common_idx = None
    for s in daily.values():
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)

    rows = []
    pairs = [(a, b) for i, a in enumerate(COUNTRIES) for b in COUNTRIES[i + 1:]]
    for a, b in pairs:
        xa = daily[a].reindex(common_idx).fillna(0).values
        xb = daily[b].reindex(common_idx).fillna(0).values
        xa = (xa - xa.mean()) / (xa.std() + 1e-9)
        xb = (xb - xb.mean()) / (xb.std() + 1e-9)

        corr = correlate(xa, xb, mode="full")
        lags = np.arange(-len(xa) + 1, len(xa))
        window = (lags >= -max_lag) & (lags <= max_lag)
        corr_w, lags_w = corr[window], lags[window]
        best_lag = int(lags_w[np.argmax(corr_w)])
        rows.append({"pair": f"{a} vs {b}", "best_lag_days": best_lag,
                     "peak_corr": float(corr_w.max() / len(xa)),
                     "lags": lags_w, "corr": corr_w / len(xa)})
    return pd.DataFrame(rows)


# =============================================================================
# PIPELINE
# =============================================================================
def run_pipeline(df_raw: pd.DataFrame) -> dict:
    daily, rt_est, rt_ci, dbl = {}, {}, {}, {}
    waves: list[Wave] = []

    for c in COUNTRIES:
        log.info("Processing %s ...", c)
        try:
            series = extract_daily(df_raw, c)
            daily[c] = series
            rt_est[c] = estimate_rt(series.values)
            rt_ci[c] = bootstrap_rt_ci(series.values, n_boot=150)
            dbl[c] = doubling_time(series.values)
        except Exception as exc:
            log.error("Failed for %s: %s", c, exc)
            continue

        raw_vals = series.values
        detected = detect_waves(raw_vals)
        log.info("  -> %d wave(s) detected", len(detected))

        for wi, (s, p, e) in enumerate(detected, start=1):
            seg = raw_vals[s: e + 1]
            try:
                fitted, r0 = fit_seir_wave(seg, POP[c])
            except Exception as exc:
                log.warning("  SEIR fit failed for %s W%d: %s", c, wi, exc)
                fitted, r0 = np.ones(len(seg)) * seg.mean(), np.nan

            reported_infections = seg.sum()
            attack_reported = reported_infections / POP[c]
            attack_adjusted = min(1.0, attack_reported * UNDERREPORT_FACTOR[c])

            early = seg[: min(21, len(seg))]
            dbl_early = doubling_time(early, window=min(7, len(early) - 1)) if len(early) > 7 else np.array([np.nan])
            dbl_val = np.nanmedian(dbl_early) if np.any(~np.isnan(dbl_early)) else np.nan

            waves.append(Wave(
                country=c, index=wi, start=s, peak=p, end=e, r0=r0,
                rt_peak=float(rt_est[c][p]), peak_cases=float(raw_vals[p]), fitted=fitted,
                attack_rate_reported=attack_reported, attack_rate_adjusted=attack_adjusted,
                doubling_time_early=dbl_val,
            ))

    lag_df = cross_country_lag(daily)
    return {"daily": daily, "rt": rt_est, "rt_ci": rt_ci, "doubling": dbl, "waves": waves, "lag": lag_df}


# =============================================================================
# SUMMARY TABLE
# =============================================================================
def build_summary(results: dict, daily: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for w in results["waves"]:
        dates = daily[w.country].index
        rows.append({
            "Country": w.country, "Wave": w.index,
            "Start": dates[w.start].strftime("%Y-%m"), "Peak": dates[w.peak].strftime("%Y-%m"),
            "End": dates[w.end].strftime("%Y-%m"), "Peak daily cases": f"{int(w.peak_cases):,}",
            "SEIR R0": f"{w.r0:.2f}" if not np.isnan(w.r0) else "-",
            "Rt at peak": f"{w.rt_peak:.2f}" if not np.isnan(w.rt_peak) else "-",
            "Doubling time (d, early)": f"{w.doubling_time_early:.1f}" if not np.isnan(w.doubling_time_early) else "-",
            "Attack rate, reported": f"{w.attack_rate_reported*100:.2f}%",
            "Attack rate, adjusted": f"{w.attack_rate_adjusted*100:.1f}%",
        })
    return pd.DataFrame(rows)


def _fmt_cases(x: float, _) -> str:
    if x >= 1_000_000:
        return f"{x/1e6:.1f}M"
    if x >= 1_000:
        return f"{x/1e3:.0f}k"
    return str(int(x))


# =============================================================================
# FIGURE 1 -- Daily cases + Rt (with bootstrap CI band)
# =============================================================================
def figure1_cases_rt(results: dict, daily: dict[str, pd.Series]) -> str:
    n = len(COUNTRIES)
    fig = plt.figure(figsize=(18, 4.5 * n))
    fig.suptitle("COVID-19 Daily Incidence and Time-varying Rt\nBangladesh, India, Pakistan",
                 fontsize=15, fontweight="bold", y=0.995)

    for row, c in enumerate(COUNTRIES):
        series = daily[c]
        obs, dates = series.values, series.index
        rt = results["rt"][c]
        rt_lo, rt_hi = results["rt_ci"][c]
        waves = [w for w in results["waves"] if w.country == c]

        ax = fig.add_subplot(n, 4, row * 4 + 1)
        ax.set_position([0.05, 1 - (row + 1) / n + 0.04, 0.55, 1 / n - 0.07])
        ax.fill_between(dates, obs, alpha=0.18, color=PALETTE[c])
        ax.plot(dates, obs, lw=1.4, color=PALETTE[c], label="Observed (7d avg)")

        for w in waves:
            seg_dates = dates[w.start: w.end + 1]
            if len(w.fitted) == len(seg_dates):
                ax.plot(seg_dates, w.fitted, lw=2, ls="--", color="#222", alpha=0.75,
                        label=f"SEIR R0={w.r0:.1f}" if w.index == 1 else "")
            ax.axvline(dates[w.peak], color="#888", lw=0.9, alpha=0.6, ls=":")
            ax.text(dates[w.peak], obs[w.peak] * 1.06, f"W{w.index}", ha="center", fontsize=8.5, color="#555",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", lw=0.6))

        ax.set_title(c, fontsize=13, fontweight="bold", color=PALETTE[c], pad=6)
        ax.set_ylabel("Daily new cases")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_cases))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.legend(loc="upper left")

        ax2 = fig.add_subplot(n, 4, row * 4 + 2)
        ax2.set_position([0.63, 1 - (row + 1) / n + 0.04, 0.18, 1 / n - 0.07])
        ax2.fill_between(dates, rt_lo, rt_hi, color="#999", alpha=0.25, label="95% CI (bootstrap)")
        ax2.fill_between(dates, rt, 1, where=(rt >= 1), alpha=0.30, color="#d62728", interpolate=True)
        ax2.fill_between(dates, rt, 1, where=(rt < 1), alpha=0.30, color="#1f77b4", interpolate=True)
        ax2.plot(dates, rt, lw=1.2, color="#111")
        ax2.axhline(1, color="#111", lw=0.9, ls="--")
        ax2.set_ylim(0, 3.5)
        ax2.set_ylabel("Rt")
        ax2.set_title("Time-varying Rt", fontsize=10)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("'%y"))
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
        if row == 0:
            ax2.text(dates[len(dates)//2], 1.07, "Rt=1", ha="center", fontsize=8, color="#888")
            ax2.legend(fontsize=7, loc="upper right")

    out = os.path.join(OUTPUT_DIR, "figure1_cases_rt.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)
    return out


# =============================================================================
# FIGURE 2 -- Comparative Rt + R0 bar chart
# =============================================================================
def figure2_comparative(results: dict, daily: dict[str, pd.Series]) -> str:
    fig, (ax_rt, ax_bar) = plt.subplots(1, 2, figsize=(17, 5.5), gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle("Comparative Rt Trajectories and Per-Wave R0", fontsize=14, fontweight="bold")

    for c in COUNTRIES:
        dates = daily[c].index
        ax_rt.plot(dates, results["rt"][c], lw=2.2, color=PALETTE[c], label=c, alpha=0.9)

    ax_rt.axhline(1, color="#333", lw=1.1, ls="--", alpha=0.8)
    ax_rt.axhspan(1, 3.5, alpha=0.03, color="#d62728")
    ax_rt.set_ylim(0.2, 3.5)
    ax_rt.set_ylabel("Smoothed Rt")
    ax_rt.set_title("Epidemic threshold: Rt = 1 (red = growing, blue = declining)")
    ax_rt.legend(framealpha=0.9)
    ax_rt.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax_rt.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax_rt.xaxis.get_majorticklabels(), rotation=30, ha="right")

    valid = [w for w in results["waves"] if not np.isnan(w.r0)]
    labels = [w.label for w in valid]
    r0_vals = [w.r0 for w in valid]
    colors = [PALETTE[w.country] for w in valid]

    x = np.arange(len(labels))
    bars = ax_bar.bar(x, r0_vals, color=colors, alpha=0.82, edgecolor="white", linewidth=0.8, zorder=3)
    ax_bar.axhline(1, color="#333", lw=1.1, ls="--", alpha=0.8, zorder=4)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax_bar.set_ylabel("SEIR-fitted R0")
    ax_bar.set_title("R0 by wave")
    ax_bar.set_ylim(0, max(r0_vals, default=3) * 1.20)
    for bar, val in zip(bars, r0_vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.06, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="500")

    from matplotlib.patches import Patch
    ax_bar.legend(handles=[Patch(color=PALETTE[c], label=c) for c in COUNTRIES], fontsize=8, framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "figure2_comparative.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)
    return out


# =============================================================================
# FIGURE 3 -- Phase space
# =============================================================================
def figure3_phase_space(results: dict, daily: dict[str, pd.Series]) -> str:
    fig, axes = plt.subplots(1, len(COUNTRIES), figsize=(17, 5.5))
    fig.suptitle("Epidemic phase portrait: Rt vs log(daily cases)\n"
                 "Colour = time (violet -> early, yellow -> late). Clockwise spirals = successive waves.",
                 fontsize=12, fontweight="bold")

    for ax, c in zip(axes, COUNTRIES):
        series = daily[c]
        obs_log = np.log1p(series.values)
        rt = results["rt"][c]
        mask = ~np.isnan(rt) & (obs_log > np.log1p(10))
        x, y = obs_log[mask], rt[mask]
        t = np.arange(len(x))

        sc = ax.scatter(x, y, c=t, cmap="plasma", s=6, alpha=0.7, linewidths=0)
        for w in results["waves"]:
            if w.country != c:
                continue
            ax.scatter(np.log1p(w.peak_cases), w.rt_peak, s=80, marker="*", color=PALETTE[c],
                       edgecolors="white", lw=0.6, zorder=5, label=f"W{w.index} peak")

        ax.axhline(1, color="#333", lw=1, ls="--", alpha=0.7)
        ax.text(x.max() * 0.97, 1.05, "Rt=1", ha="right", fontsize=8, color="#777")
        ax.set_title(c, fontsize=13, fontweight="bold", color=PALETTE[c])
        ax.set_xlabel("log(1 + daily cases)")
        ax.set_ylabel("Rt")
        ax.set_ylim(0.1, 3.5)
        ax.legend(fontsize=8, loc="upper right")
        cb = plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
        cb.set_label("Time ->", fontsize=9)
        cb.set_ticks([])

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "figure3_phase_space.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)
    return out


# =============================================================================
# FIGURE 4 -- Serial interval
# =============================================================================
def figure4_serial_interval() -> str:
    s = np.arange(1, len(SERIAL_INTERVAL) + 1)
    si = SERIAL_INTERVAL
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(s, si, color="#4c72b0", alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.axvline(SI_MU, color="#c44e52", lw=1.5, ls="--", label=f"Mean = {SI_MU} d")
    ax.set_xlabel("Serial interval (days)")
    ax.set_ylabel("Probability mass")
    ax.set_title(f"Discretised log-normal serial interval\n(mu={SI_MU}d, sigma={SI_SIGMA}d; Nishiura et al. 2020)")
    ax.legend()
    ax.set_xlim(0, 25)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "figure4_serial_interval.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)
    return out


# =============================================================================
# FIGURE 5 -- Doubling time trajectories  [NEW]
# =============================================================================
def figure5_doubling_time(results: dict, daily: dict[str, pd.Series]) -> str:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for c in COUNTRIES:
        dates = daily[c].index
        dbl = results["doubling"][c]
        ax.plot(dates, dbl, lw=1.6, color=PALETTE[c], label=c, alpha=0.85)

    ax.set_ylabel("Local doubling time (days)")
    ax.set_title("Epidemic doubling time over time\n(shorter = faster growth; NaN gaps = epidemic not growing)")
    ax.set_ylim(0, 60)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "figure5_doubling_time.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)
    return out


# =============================================================================
# FIGURE 6 -- Attack rate, reported vs underreporting-adjusted  [NEW]
# =============================================================================
def figure6_attack_rates(results: dict) -> str:
    waves = results["waves"]
    labels = [w.label for w in waves]
    reported = [w.attack_rate_reported * 100 for w in waves]
    adjusted = [w.attack_rate_adjusted * 100 for w in waves]
    colors = [PALETTE[w.country] for w in waves]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width/2, reported, width, color=colors, alpha=0.45, edgecolor="white", label="Reported")
    ax.bar(x + width/2, adjusted, width, color=colors, alpha=0.90, edgecolor="white", label="Underreporting-adjusted")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Attack rate (% of population infected)")
    ax.set_title("Reported vs seroprevalence-adjusted attack rate, by wave\n"
                  "(adjustment factors are rough single-number estimates, see UNDERREPORT_FACTOR)")

    from matplotlib.patches import Patch
    style_handles = [Patch(facecolor="grey", alpha=0.45, label="Reported"),
                      Patch(facecolor="grey", alpha=0.90, label="Adjusted")]
    ax.legend(handles=style_handles, loc="upper right")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "figure6_attack_rates.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)
    return out


# =============================================================================
# FIGURE 7 -- Cross-country transmission lag  [NEW]
# =============================================================================
def figure7_cross_lag(results: dict) -> str:
    lag_df = results["lag"]
    fig, axes = plt.subplots(1, len(lag_df), figsize=(6 * len(lag_df), 4.5))
    if len(lag_df) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, lag_df.iterrows()):
        ax.plot(row["lags"], row["corr"], color="#444", lw=1.5)
        ax.axvline(row["best_lag_days"], color="#c44e52", lw=1.5, ls="--",
                   label=f"peak lag = {row['best_lag_days']:+d} d")
        ax.axvline(0, color="#aaa", lw=0.8)
        ax.set_title(row["pair"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("Normalised cross-correlation")
        ax.legend(fontsize=8)

    fig.suptitle("Cross-country case-curve lag\n"
                 "Positive lag = first-named country's wave leads the second's by that many days",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "figure7_cross_country_lag.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)
    return out


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    df_raw = load_jhu_data()
    results = run_pipeline(df_raw)
    daily = results["daily"]

    summary = build_summary(results, daily)
    print("\n" + "=" * 100)
    print("  WAVE SUMMARY - SEIR R0, Cori Rt at peak, doubling time, attack rate")
    print("=" * 100)
    print(summary.to_string(index=False))
    print("=" * 100)

    print("\nCross-country lag (which country's wave tends to lead):")
    print(results["lag"][["pair", "best_lag_days", "peak_corr"]].to_string(index=False))

    figure1_cases_rt(results, daily)
    figure2_comparative(results, daily)
    figure3_phase_space(results, daily)
    figure4_serial_interval()
    figure5_doubling_time(results, daily)
    figure6_attack_rates(results)
    figure7_cross_lag(results)

    csv_path = os.path.join(OUTPUT_DIR, "wave_summary.csv")
    summary.to_csv(csv_path, index=False)
    log.info("Summary CSV: %s", csv_path)

    lag_csv = os.path.join(OUTPUT_DIR, "cross_country_lag.csv")
    results["lag"][["pair", "best_lag_days", "peak_corr"]].to_csv(lag_csv, index=False)
    log.info("Lag CSV: %s", lag_csv)

    print("""
METHODOLOGICAL NOTES
---------------------
Daily cases    : JHU CSSE 7-day centred rolling average, negative revisions
                 clipped to zero, reporting spikes flagged (not removed).

Serial interval: Discretised log-normal (mu=5.1d, sigma=2.6d), Nishiura
                 et al. (2020) / Bi et al. (2020).

Rt estimation  : Cori/EpiEstim renewal equation, 7-day numerator window,
                 14-day Gaussian smoothing. 95% band from 150x Poisson
                 bootstrap of the incidence series (approximate, not a
                 full posterior).

SEIR fitting   : Per-wave beta fit (Nelder-Mead) against relative RMSE.
                 S/E/I/R tracked as population fractions for numerical
                 stability at India-scale N. Incubation 1/sigma=5.2d
                 (Lauer et al 2020), infectious period 1/gamma=7d.
                 Amplitude solved analytically from peak ratio.

Doubling time  : Local log-linear slope over a 7-day window; undefined
                 (NaN) when the epidemic isn't growing.

Attack rate    : Reported = sum of wave incidence / population. Adjusted
                 applies a flat underreporting multiplier per country
                 (rough seroprevalence-literature estimates) -- treat as
                 order-of-magnitude, not precise.

Cross-lag      : Normalised cross-correlation of standardised case curves
                 between country pairs, +/-60 day window.
""")


if __name__ == "__main__":
    main()
