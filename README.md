# COVID-19 Epidemiological Analysis: Bangladesh, India, Pakistan

## Overview


## Data
- Source: [JHU CSSE COVID-19 time series](https://github.com/CSSEGISandData/COVID-19)
- Countries: Bangladesh, India, Pakistan
- Series: daily confirmed cases, 7-day centred rolling average

## Methods
- **Serial interval**: discretised log-normal (μ = 5.1 d, σ = 2.6 d) — Nishiura et al. (2020), Bi et al. (2020)
- **Rt estimation**: Cori/EpiEstim renewal equation, 7-day numerator window, 14-day Gaussian smoothing, with a 95% band from 150× Poisson bootstrap
- **Wave detection**: two-stage peak-finding + valley merge on the smoothed incidence curve
- **SEIR fitting**: per-wave β fit (Nelder–Mead) against relative RMSE, S/E/I/R tracked as population fractions; incubation 1/σ = 5.2 d (Lauer et al. 2020), infectious period 1/γ = 7 d
- **Doubling time**: local log-linear slope over a 7-day window
- **Attack rate**: reported (case sum ÷ population) and underreporting-adjusted (flat per-country multiplier from seroprevalence literature — order-of-magnitude only)
- **Cross-country lag**: normalised cross-correlation of standardised case curves, ±60-day window

## Files
| File | Description |
|---|---|
| `covid_analysis.py` | Full pipeline: data load → Rt → wave detection → SEIR fit → figures |
| `figure1_cases_rt.png` | Daily cases + time-varying Rt per country |
| `figure2_comparative.png` | Rt trajectories overlaid + R0 by wave |
| `figure3_phase_space.png` | Rt vs log(cases) phase portrait |
| `figure4_serial_interval.png` | Serial interval distribution used |
| `figure5_doubling_time.png` | Doubling time over time |
| `figure6_attack_rates.png` | Reported vs adjusted attack rate by wave |
| `figure7_cross_country_lag.png` | Cross-country transmission lag |
| `wave_summary.csv` | Per-wave R0, Rt, doubling time, attack rate |
| `cross_country_lag.csv` | Lag/correlation by country pair |

## Requirements
```
pandas
numpy
scipy
matplotlib
```

## Usage
```bash
python covid_analysis.py
```



