"""
COVID-19 Epidemiological Analysis: Bangladesh · India · Pakistan
================================================================
Methodology  : Discrete renewal-equation Rt (Cori/EpiEstim), per-wave SIR fitting
Visualisation: Publication-quality figures with seaborn + matplotlib
Robustness   : Full error handling, data-quality checks, logging, type hints
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
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
from scipy.stats import lognorm

warnings.filterwarnings("ignore")

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = "seir_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── design tokens ─────────────────────────────────────────────────────────────
PALETTE = {
    "Bangladesh": "#1f78b4",
    "India":      "#e31a1c",
    "Pakistan":   "#33a02c",
}
LIGHT = {
    "Bangladesh": "#a6cee3",
    "India":      "#fb9a99",
    "Pakistan":   "#b2df8a",
}

# Publication-quality rcParams
plt.rcParams.update({
    "figure.dpi":        150,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#fafafa",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#e0e0e0",
    "grid.linewidth":    0.6,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

POP: dict[str, int] = {
    "Bangladesh":  171_000_000,
    "India":     1_417_000_000,
    "Pakistan":    241_000_000,
}
COUNTRIES = list(POP.keys())

# ── serial interval parameters ────────────────────────────────────────────────
# Nishiura et al. (2020) & Bi et al. (2020) meta-analytic estimates for SARS-CoV-2
SI_MU    = 5.1   # days
SI_SIGMA = 2.6   # days
GAMMA    = 1 / 7  # removal rate (infectious period ~7d)


# =============================================================================
# DATACLASSES
# =============================================================================
@dataclass
class Wave:
    country:   str
    index:     int
    start:     int
    peak:      int
    end:       int
    r0:        float = np.nan
    rt_peak:   float = np.nan
    peak_cases: float = np.nan
    fitted:    np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def label(self) -> str:
        return f"{self.country[:3]} W{self.index}"


# =============================================================================
# 1.  DATA LOADING & VALIDATION
# =============================================================================
JHU_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
    "time_series_covid19_confirmed_global.csv"
)

def load_jhu_data(url: str = JHU_URL) -> pd.DataFrame:
    log.info("Fetching JHU CSSE time-series …")
    try:
        df = pd.read_csv(url)
    except Exception as exc:
        log.error("Data fetch failed: %s", exc)
        sys.exit(1)

    required = {"Country/Region", "Province/State", "Lat", "Long"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected JHU schema; missing columns: {missing}")
    log.info("Loaded %d rows × %d date columns", len(df), df.shape[1] - 4)
    return df


def extract_daily(df: pd.DataFrame, country: str, smooth: int = 7) -> pd.Series:
    """
    Aggregate provinces → national cumulative → daily incidence → smooth.
    Negative corrections (reporting artefacts) are clipped to zero.
    """
    subset = df[df["Country/Region"] == country]
    if subset.empty:
        raise KeyError(f"Country not found in JHU data: {country!r}")

    cumulative = (
        subset
        .drop(columns=["Province/State", "Country/Region", "Lat", "Long"])
        .sum()
    )
    cumulative.index = pd.to_datetime(cumulative.index, format="%m/%d/%y")
    daily = cumulative.diff().clip(lower=0).fillna(0)

    # Flag suspicious spikes (>10× previous 7d mean) — log warning only
    rolling_mean = daily.rolling(7, min_periods=1).mean().shift(1)
    spikes = daily[daily > rolling_mean * 10]
    if not spikes.empty:
        log.warning(
            "%s: %d potential reporting spike(s) detected (dates: %s)",
            country,
            len(spikes),
            ", ".join(spikes.index.strftime("%Y-%m-%d")),
        )

    return daily.rolling(smooth, center=True, min_periods=1).mean()


# =============================================================================
# 2.  SERIAL INTERVAL  (discretised log-normal)
# =============================================================================
def build_serial_interval(max_s: int = 30, mu: float = SI_MU, sigma: float = SI_SIGMA) -> np.ndarray:
    """
    Discretise a log-normal serial interval distribution.
    Uses exact log-normal parameters derived from mean + SD.
    Reference: Nishiura et al. Int J Infect Dis, 2020.
    """
    s = np.arange(1, max_s + 1, dtype=float)
    # Convert (mean, SD) → (μ_log, σ_log)
    log_mu    = np.log(mu**2 / np.sqrt(mu**2 + sigma**2))
    log_sigma = np.sqrt(np.log(1 + sigma**2 / mu**2))
    # Probability mass: P(s-0.5 < X ≤ s+0.5)
    w = lognorm.cdf(s + 0.5, s=log_sigma, scale=np.exp(log_mu)) \
      - lognorm.cdf(np.maximum(s - 0.5, 0), s=log_sigma, scale=np.exp(log_mu))
    return w / w.sum()   # normalise to sum = 1


SERIAL_INTERVAL = build_serial_interval()


# =============================================================================
# 3.  Rt ESTIMATION  (Cori / EpiEstim renewal equation)
# =============================================================================
def estimate_rt(
    cases: np.ndarray,
    w: np.ndarray = SERIAL_INTERVAL,
    window: int = 7,
    smooth_sigma: float = 14,
    clip: tuple[float, float] = (0.1, 5.0),
) -> np.ndarray:
    """
    Instantaneous Rt via discrete renewal equation:
        Rt(t) ≈ mean[I(t-k+1..t)] / Σ_s w(s)·I(t-s)

    Parameters
    ----------
    cases        : 1-D array of smoothed daily incidence
    w            : discretised serial-interval weights
    window       : numerator averaging window (days)
    smooth_sigma : Gaussian smoothing bandwidth (days)
    clip         : (min, max) clip for Rt values
    """
    n  = len(cases)
    rt = np.full(n, np.nan)

    for t in range(len(w), n):
        numerator   = cases[max(0, t - window + 1) : t + 1].mean()
        denominator = float(np.dot(w[: min(t, len(w))], cases[t - min(t, len(w)) : t][::-1]))
        if denominator > 1.0:
            rt[t] = numerator / denominator

    rt = np.clip(rt, *clip)
    rt = pd.Series(rt).ffill().bfill().values
    return gaussian_filter1d(rt, sigma=smooth_sigma)


# =============================================================================
# 4.  SIR MODEL  (vectorised Euler integration)
# =============================================================================
def run_sir(
    N: int,
    I0: float,
    beta: float,
    gamma: float = GAMMA,
    n_days: Optional[int] = None,
) -> np.ndarray:
    """Return array of daily new infections from a simple SIR model."""
    if n_days is None:
        n_days = 365
    S, I = float(N - I0), float(I0)
    new_infections = []
    for _ in range(n_days):
        foi = beta * I / N          # force of infection
        dI  = foi * S - gamma * I
        new_infections.append(foi * S)
        S  -= foi * S
        I  += dI
        I   = max(I, 0.0)
        S   = max(S, 0.0)
    return np.array(new_infections)


def fit_sir_wave(obs: np.ndarray, N: int, gamma: float = GAMMA) -> tuple[np.ndarray, float]:
    """
    Fit a SIR model to a single epidemic wave using relative-RMSE loss.
    Amplitude (reporting rate) is solved analytically to avoid an extra
    optimisation dimension.

    Returns
    -------
    fitted : smoothed predicted curve, rescaled to observed amplitude
    R0     : beta / gamma
    """
    n        = len(obs)
    peak_obs = obs.max()
    if peak_obs < 10 or n < 14:
        return np.ones(n) * obs.mean(), np.nan

    I0 = max(1, int(N / 1_000_000))

    def loss(log_beta: float) -> float:
        beta = np.exp(log_beta)
        R0   = beta / gamma
        if not (0.5 <= R0 <= 10):
            return 1e9
        pred = run_sir(N, I0, beta, gamma, n)
        peak_pred = pred.max()
        if peak_pred < 1e-6:
            return 1e9
        scale = peak_obs / peak_pred
        rel_err = ((pred * scale - obs) / (obs + peak_obs * 0.05 + 1)) ** 2
        return float(np.sqrt(rel_err.mean()))

    result  = minimize_scalar(loss, bounds=(-3, 2), method="bounded",
                              options={"xatol": 1e-4})
    beta_hat = np.exp(result.x)
    pred     = run_sir(N, I0, beta_hat, gamma, n)
    scale    = peak_obs / (pred.max() + 1e-9)
    return pred * scale, beta_hat / gamma


# =============================================================================
# 5.  WAVE DETECTION
# =============================================================================
def detect_waves(
    cases: np.ndarray,
    min_distance: int = 45,
    peak_height_abs: float = 200.0,
    prominence_frac: float = 0.06,
    valley_depth_frac: float = 0.40,
) -> list[tuple[int, int, int]]:
    """
    Two-stage wave detection using scipy.signal.find_peaks + valley confirmation.

    Stage 1 — Peak finding
        find_peaks with:
        • height      >= peak_height_abs (absolute floor, ignores noise)
        • distance    >= min_distance days between peaks
        • prominence  >= prominence_frac × global_max
          (prevents shoulders from being counted as separate waves)

    Stage 2 — Valley merging
        For each consecutive pair of candidate peaks, compute the valley
        minimum between them.  If valley > valley_depth_frac × min(peaks),
        the two surges are not truly separated → keep the higher peak only.

    Segment boundaries
        Start/end of each wave = valley midpoint between consecutive peaks
        (non-overlapping by construction).  For the first/last wave the
        boundary is the point where cases fall to 15 % of the local peak.

    This approach correctly recovers small early waves even when later waves
    are an order of magnitude higher (Bangladesh 2020 vs Delta/Omicron), and
    avoids the S-depletion resonance trap of global-threshold methods.
    """
    global_max = cases.max()
    if global_max < 1:
        return []

    # ── Stage 1: candidate peaks ──
    peak_idx, _ = find_peaks(
        cases,
        height=peak_height_abs,
        distance=min_distance,
        prominence=global_max * prominence_frac,
    )
    if len(peak_idx) == 0:
        return []

    # ── Stage 2: valley-based merge ──
    merged: list[int] = [int(peak_idx[0])]
    for pi in peak_idx[1:]:
        prev       = merged[-1]
        valley_val = float(cases[prev : pi + 1].min())
        threshold  = valley_depth_frac * min(float(cases[prev]), float(cases[pi]))
        if valley_val <= threshold:
            merged.append(int(pi))          # clear valley → distinct wave
        else:
            if cases[pi] > cases[prev]:     # no clear valley → keep higher peak
                merged[-1] = int(pi)

    # ── Build non-overlapping segments ──
    n      = len(cases)
    starts = []
    ends   = []

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
            next_pi    = merged[k + 1]
            valley_pos = int(pi + int(cases[pi : next_pi + 1].argmin()))
            ends.append(valley_pos)

    return list(zip(starts, merged, ends))


# =============================================================================
# 6.  PIPELINE
# =============================================================================
def run_pipeline(df_raw: pd.DataFrame) -> dict:
    """
    Run full analysis pipeline for all countries.

    Returns dict with keys: 'daily', 'rt', 'waves'
    """
    daily  = {}
    rt_est = {}
    waves: list[Wave] = []

    for c in COUNTRIES:
        log.info("Processing %s …", c)
        try:
            series     = extract_daily(df_raw, c)
            daily[c]   = series
            rt_est[c]  = estimate_rt(series.values)
        except Exception as exc:
            log.error("Failed for %s: %s", c, exc)
            continue

        raw_vals = series.values
        detected = detect_waves(raw_vals)
        log.info("  → %d wave(s) detected", len(detected))

        for wi, (s, p, e) in enumerate(detected, start=1):
            seg = raw_vals[s : e + 1]
            try:
                fitted, r0 = fit_sir_wave(seg, POP[c])
            except Exception as exc:
                log.warning("  SIR fit failed for %s W%d: %s", c, wi, exc)
                fitted, r0 = np.ones(len(seg)) * seg.mean(), np.nan

            waves.append(Wave(
                country    = c,
                index      = wi,
                start      = s,
                peak       = p,
                end        = e,
                r0         = r0,
                rt_peak    = float(rt_est[c][p]),
                peak_cases = float(raw_vals[p]),
                fitted     = fitted,
            ))

    return {"daily": daily, "rt": rt_est, "waves": waves}


# =============================================================================
# 7.  SUMMARY TABLE
# =============================================================================
def build_summary(results: dict, daily: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for w in results["waves"]:
        dates = daily[w.country].index
        rows.append({
            "Country":           w.country,
            "Wave":              w.index,
            "Start":             dates[w.start].strftime("%Y-%m"),
            "Peak":              dates[w.peak].strftime("%Y-%m"),
            "End":               dates[w.end].strftime("%Y-%m"),
            "Peak daily cases":  f"{int(w.peak_cases):,}",
            "SIR R₀":            f"{w.r0:.2f}" if not np.isnan(w.r0) else "—",
            "Rₜ at peak":        f"{w.rt_peak:.2f}" if not np.isnan(w.rt_peak) else "—",
        })
    return pd.DataFrame(rows)


# =============================================================================
# 8.  FIGURE 1  — Daily cases + Rt per country
# =============================================================================
def _fmt_cases(x: float, _) -> str:
    if x >= 1_000_000: return f"{x/1e6:.1f}M"
    if x >= 1_000:     return f"{x/1e3:.0f}k"
    return str(int(x))


def figure1_cases_rt(results: dict, daily: dict[str, pd.Series]) -> str:
    n   = len(COUNTRIES)
    fig = plt.figure(figsize=(18, 4.5 * n))
    fig.suptitle(
        "COVID-19 Daily Incidence and Time-varying Rₜ\nBangladesh · India · Pakistan",
        fontsize=15, fontweight="bold", y=0.995,
    )

    for row, c in enumerate(COUNTRIES):
        series = daily[c]
        obs    = series.values
        dates  = series.index
        rt     = results["rt"][c]
        waves  = [w for w in results["waves"] if w.country == c]

        # ── Cases panel (left 75 %) ──
        ax = fig.add_subplot(n, 4, row * 4 + 1)
        ax.set_position([0.05, 1 - (row + 1) / n + 0.04,
                          0.55, 1 / n - 0.07])

        ax.fill_between(dates, obs, alpha=0.18, color=PALETTE[c])
        ax.plot(dates, obs, lw=1.4, color=PALETTE[c], label="Observed (7d avg)")

        for w in waves:
            seg_dates = dates[w.start : w.end + 1]
            if len(w.fitted) == len(seg_dates):
                ax.plot(seg_dates, w.fitted, lw=2, ls="--", color="#222",
                        alpha=0.75,
                        label=f"SIR R₀={w.r0:.1f}" if w.index == 1 else "")
            ax.axvline(dates[w.peak], color="#888", lw=0.9, alpha=0.6, ls=":")
            ypos = obs[w.peak] * 1.06
            ax.text(dates[w.peak], ypos, f"W{w.index}",
                    ha="center", fontsize=8.5, color="#555",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", lw=0.6))

        ax.set_title(c, fontsize=13, fontweight="bold",
                     color=PALETTE[c], pad=6)
        ax.set_ylabel("Daily new cases")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_cases))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.legend(loc="upper left")

        # ── Rt panel (right 20 %) ──
        ax2 = fig.add_subplot(n, 4, row * 4 + 2)
        ax2.set_position([0.63, 1 - (row + 1) / n + 0.04,
                           0.18, 1 / n - 0.07])

        ax2.fill_between(dates, rt, 1,
                         where=(rt >= 1), alpha=0.30, color="#d62728", interpolate=True)
        ax2.fill_between(dates, rt, 1,
                         where=(rt <  1), alpha=0.30, color="#1f77b4", interpolate=True)
        ax2.plot(dates, rt, lw=1.2, color="#111")
        ax2.axhline(1, color="#111", lw=0.9, ls="--")
        ax2.set_ylim(0, 3.5)
        ax2.set_ylabel("Rₜ")
        ax2.set_title("Time-varying Rₜ", fontsize=10)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("'%y"))
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

        # threshold annotation on first row only
        if row == 0:
            ax2.text(dates[len(dates)//2], 1.07, "Rₜ=1",
                     ha="center", fontsize=8, color="#888")

    out = os.path.join(OUTPUT_DIR, "figure1_cases_rt.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    log.info("Saved: %s", out)
    return out


# =============================================================================
# 9.  FIGURE 2  — Comparative Rt + R0 bar chart
# =============================================================================
def figure2_comparative(results: dict, daily: dict[str, pd.Series]) -> str:
    fig, (ax_rt, ax_bar) = plt.subplots(
        1, 2, figsize=(17, 5.5),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    fig.suptitle("Comparative Rₜ Trajectories and Per-Wave R₀",
                 fontsize=14, fontweight="bold")

    # ── Rt lines ──
    for c in COUNTRIES:
        dates = daily[c].index
        rt    = results["rt"][c]
        ax_rt.plot(dates, rt, lw=2.2, color=PALETTE[c], label=c, alpha=0.9)

    ax_rt.axhline(1, color="#333", lw=1.1, ls="--", alpha=0.8)
    ax_rt.axhspan(1, 3.5, alpha=0.03, color="#d62728")
    ax_rt.set_ylim(0.2, 3.5)
    ax_rt.set_ylabel("Smoothed Rₜ")
    ax_rt.set_title("Epidemic threshold: Rₜ = 1  (red = growing, blue = declining)")
    ax_rt.legend(framealpha=0.9)
    ax_rt.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax_rt.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax_rt.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── R0 bars ──
    waves   = results["waves"]
    valid   = [w for w in waves if not np.isnan(w.r0)]
    labels  = [w.label for w in valid]
    r0_vals = [w.r0 for w in valid]
    colors  = [PALETTE[w.country] for w in valid]

    x    = np.arange(len(labels))
    bars = ax_bar.bar(x, r0_vals, color=colors, alpha=0.82, edgecolor="white", linewidth=0.8,
                      zorder=3)
    ax_bar.axhline(1, color="#333", lw=1.1, ls="--", alpha=0.8, zorder=4)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax_bar.set_ylabel("SIR-fitted R₀")
    ax_bar.set_title("R₀ by wave")
    ax_bar.set_ylim(0, max(r0_vals, default=3) * 1.20)

    for bar, val in zip(bars, r0_vals):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.06,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="500",
        )

    # Legend patches for country colours
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=PALETTE[c], label=c) for c in COUNTRIES]
    ax_bar.legend(handles=legend_handles, fontsize=8, framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "figure2_comparative.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    log.info("Saved: %s", out)
    return out


# =============================================================================
# 10.  FIGURE 3  — Phase-space: Rt vs log(cases)
# =============================================================================
def figure3_phase_space(results: dict, daily: dict[str, pd.Series]) -> str:
    fig, axes = plt.subplots(1, len(COUNTRIES), figsize=(17, 5.5))
    fig.suptitle(
        "Epidemic phase portrait: Rₜ vs log(daily cases)\n"
        "Colour = time (violet → early, yellow → late). "
        "Clockwise spirals indicate successive waves.",
        fontsize=12, fontweight="bold",
    )

    for ax, c in zip(axes, COUNTRIES):
        series = daily[c]
        obs_log = np.log1p(series.values)
        rt      = results["rt"][c]
        mask    = ~np.isnan(rt) & (obs_log > np.log1p(10))
        x, y    = obs_log[mask], rt[mask]
        n       = len(x)
        t       = np.arange(n)

        sc = ax.scatter(x, y, c=t, cmap="plasma", s=6, alpha=0.7, linewidths=0)

        # Mark wave peaks
        for w in results["waves"]:
            if w.country != c:
                continue
            peak_log = np.log1p(w.peak_cases)
            ax.scatter(peak_log, w.rt_peak, s=80, marker="*",
                       color=PALETTE[c], edgecolors="white", lw=0.6, zorder=5,
                       label=f"W{w.index} peak")

        ax.axhline(1, color="#333", lw=1, ls="--", alpha=0.7)
        ax.text(x.max() * 0.97, 1.05, "Rₜ=1",
                ha="right", fontsize=8, color="#777")
        ax.set_title(c, fontsize=13, fontweight="bold", color=PALETTE[c])
        ax.set_xlabel("log(1 + daily cases)")
        ax.set_ylabel("Rₜ")
        ax.set_ylim(0.1, 3.5)
        ax.legend(fontsize=8, loc="upper right")
        cb = plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
        cb.set_label("Time →", fontsize=9)
        cb.set_ticks([])

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "figure3_phase_space.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    log.info("Saved: %s", out)
    return out


# =============================================================================
# 11.  FIGURE 4  — Serial interval visualisation  [NEW]
# =============================================================================
def figure4_serial_interval() -> str:
    """
    Visualise the discretised log-normal serial interval distribution.
    Makes the methodological assumption explicit and citable.
    """
    s  = np.arange(1, len(SERIAL_INTERVAL) + 1)
    si = SERIAL_INTERVAL

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(s, si, color="#4c72b0", alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.axvline(SI_MU, color="#c44e52", lw=1.5, ls="--", label=f"Mean = {SI_MU} d")
    ax.set_xlabel("Serial interval (days)")
    ax.set_ylabel("Probability mass")
    ax.set_title(f"Discretised log-normal serial interval\n(μ={SI_MU} d, σ={SI_SIGMA} d;  Nishiura et al. 2020)")
    ax.legend()
    ax.set_xlim(0, 25)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "figure4_serial_interval.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    log.info("Saved: %s", out)
    return out


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    df_raw  = load_jhu_data()
    results = run_pipeline(df_raw)
    daily   = results["daily"]

    summary = build_summary(results, daily)
    print("\n" + "═" * 76)
    print("  WAVE SUMMARY  —  SIR R₀ and Cori Rₜ at peak")
    print("═" * 76)
    print(summary.to_string(index=False))
    print("═" * 76)

    figure1_cases_rt(results, daily)
    figure2_comparative(results, daily)
    figure3_phase_space(results, daily)
    figure4_serial_interval()

    csv_path = os.path.join(OUTPUT_DIR, "wave_summary.csv")
    summary.to_csv(csv_path, index=False)
    log.info("Summary CSV: %s", csv_path)

    print("""
METHODOLOGICAL NOTES
────────────────────
Daily cases  : JHU CSSE 7-day centred rolling average; negative corrections
               (data revisions) clipped to zero; reporting spikes flagged.

Serial interval: Discretised log-normal (μ=5.1 d, σ=2.6 d) using exact
               log-normal parameterisation from Nishiura et al. (2020) and
               Bi et al. (2020). Probability mass computed via CDF differences
               (more accurate than PDF point evaluation).

Rₜ estimation : Discrete renewal equation (Cori/EpiEstim):
               Rₜ(t) = mean[I(t-k+1..t)] / Σ_s w(s)·I(t-s)
               Numerator window = 7 d; posterior smoothed with 14-d Gaussian
               kernel.  Red shading = Rₜ>1 (epidemic growing).

SIR fitting   : Per-wave optimisation of β (bounded Brent method) minimising
               relative-RMSE.  Amplitude (reporting fraction) solved
               analytically to avoid S-depletion resonance.  γ = 1/7 d⁻¹.
               New infections tracked as force-of-infection × S (not γI),
               giving a sharper early-epidemic signal.

Phase portrait: Rₜ vs log(1+cases); clockwise spirals indicate wave cycles.
               Wave peaks marked with ★.
""")


if __name__ == "__main__":
    main()