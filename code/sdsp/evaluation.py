"""Experiment protocol, metrics and statistical tests.

Protocol (fixed before looking at the test block):
    1. The series is split chronologically into fit, validation and test blocks (see data.Series).
    2. Every hyper-parameter (AR order, step size, forgetting factor, noise variances) is chosen
       by minimising the one-step MSE of r on the validation block only.
    3. With the chosen hyper-parameters, each model predicts every test day one step ahead,
       using only data up to the previous day (walk-forward; AR coefficients re-estimated
       every day on an expanding window; adaptive filters keep running).
    4. Scores on days whose true value was interpolated are excluded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from .data import Series
from .models import (ar_fit, ar_predict_h, ar_predict_next, kalman_local_level, kalman_tvar,
                     lms_predict, rls_predict)

ORDERS = (1, 2, 4, 8)
AR_ORDERS = tuple(range(0, 11))                        # p = 0 is the naive predictor (zero-mean AR)
LMS_FRACTIONS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)   # mu = c * 2 / (p * sigma^2)
NLMS_STEPS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
RLS_LAMBDAS = (0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0)
TVAR_QS = (1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
LL_RATIOS = tuple(10.0 ** np.arange(-3, 4.5, 0.5))      # q / r for the local-level model


# --------------------------------------------------------------------------------------
# Metrics (y = true values, yhat = predictions, aligned in time)
# --------------------------------------------------------------------------------------
def mse(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean((y - yhat) ** 2))


def rmse(y, yhat) -> float:
    return float(np.sqrt(mse(y, yhat)))


def mae(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs(y - yhat)))


def mape(y, yhat) -> float:
    """Mean absolute percentage error, in %."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(100 * np.mean(np.abs((y - yhat) / y)))


def directional_accuracy(p_prev, p_true, p_pred) -> float:
    """% of steps where the predicted price move has the same sign as the real move.

    A prediction of 'no change' (e.g. the naive baseline) counts as wrong, so it scores ~0 %.
    """
    p_prev, p_true, p_pred = (np.asarray(a, float) for a in (p_prev, p_true, p_pred))
    return float(100 * np.mean(np.sign(p_true - p_prev) == np.sign(p_pred - p_prev)))


def price_metrics(p_prev, p_true, p_pred) -> dict:
    return {
        "RMSE [USD]": rmse(p_true, p_pred),
        "MAE [USD]": mae(p_true, p_pred),
        "MAPE [%]": mape(p_true, p_pred),
        "Dir. acc. [%]": directional_accuracy(p_prev, p_true, p_pred),
    }


# --------------------------------------------------------------------------------------
# Predictors: each returns r_hat for ALL positions 0..N-1 (one-step, causal)
# --------------------------------------------------------------------------------------
def pred_naive(s: Series, **_):
    return np.zeros(len(s.r))


def pred_mean(s: Series, **_):
    """Drift model r_hat[n] = mean(r[1..n-1]) (AR(0) with non-zero mean)."""
    c = np.cumsum(s.r)
    out = np.zeros(len(s.r))
    for n in range(2, len(s.r)):
        out[n] = c[n - 1] / (n - 1)
    return out


def pred_ar(s: Series, p: int, demean: bool = False, positions=None, **_):
    """AR(p) via Levinson-Durbin, re-estimated at every step on r[1..n-1] (expanding window)."""
    out = np.zeros(len(s.r))
    pos = range(max(p, 1) + 1, len(s.r)) if positions is None else positions
    for n in pos:
        h = s.r[1:n]
        a, mu, _ = ar_fit(h, p, demean=demean)
        out[n] = ar_predict_next(h, a, mu)
    return out


def pred_lms(s: Series, p: int, c: float, **_):
    mu = c * 2.0 / (p * s.sigma2_fit)          # fraction c of the bound 2 / tr(R)
    xh, _ = lms_predict(s.r, p, mu)
    return xh


def pred_nlms(s: Series, p: int, mu: float, **_):
    xh, _ = lms_predict(s.r, p, mu, normalized=True, eps=p * s.sigma2_fit)
    return xh


def pred_rls(s: Series, p: int, lam: float, **_):
    xh, _ = rls_predict(s.r, p, lam, P0_scale=1.0 / s.sigma2_fit)
    return xh


def pred_tvar(s: Series, p: int, q: float, **_):
    xh, _ = kalman_tvar(s.r, p, q, s.sigma2_fit, P0=1.0)
    return xh


def pred_local_level(s: Series, ratio: float, **_):
    yhat, _, _ = kalman_local_level(s.x, ratio * s.sigma2_fit, s.sigma2_fit, P0=s.sigma2_fit)
    return yhat - np.r_[s.x[0], s.x[:-1]]      # r_hat[n] = s_hat[n|n-1] - x[n-1]


PREDICTORS = {
    "AR": pred_ar,
    "LMS": pred_lms,
    "NLMS": pred_nlms,
    "RLS": pred_rls,
    "Kalman TV-AR": pred_tvar,
    "Kalman local level": pred_local_level,
}


# --------------------------------------------------------------------------------------
# Tuning
# --------------------------------------------------------------------------------------
def tune(s: Series) -> tuple[dict, pd.DataFrame]:
    """Grid search on the validation block. Returns best params per model and the full grid."""
    rows = []
    base = s.mse_r(s.val, np.zeros(len(s.val)))

    def add(model, params, rhat):
        rows.append({"model": model, **params, "val_mse_rel": s.mse_r(s.val, rhat[s.val]) / base})

    for p in AR_ORDERS:
        add("AR", {"p": p}, pred_ar(s, p, positions=s.val))
    for p in ORDERS:
        for c in LMS_FRACTIONS:
            add("LMS", {"p": p, "c": c}, pred_lms(s, p, c))
        for mu in NLMS_STEPS:
            add("NLMS", {"p": p, "mu": mu}, pred_nlms(s, p, mu))
        for lam in RLS_LAMBDAS:
            add("RLS", {"p": p, "lam": lam}, pred_rls(s, p, lam))
        for q in TVAR_QS:
            add("Kalman TV-AR", {"p": p, "q": q}, pred_tvar(s, p, q))
    for ratio in LL_RATIOS:
        add("Kalman local level", {"ratio": ratio}, pred_local_level(s, ratio))
    grid = pd.DataFrame(rows)
    best = {}
    for m, g in grid.groupby("model", sort=False):
        row = g.loc[g["val_mse_rel"].idxmin()]
        best[m] = {k: (int(v) if k == "p" else float(v)) for k, v in row.items()
                   if k not in ("model", "val_mse_rel") and pd.notna(v)}
    return best, grid


# --------------------------------------------------------------------------------------
# Significance tests
# --------------------------------------------------------------------------------------
def dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 1):
    """Diebold-Mariano test (squared-error loss) with the Harvey-Leybourne-Newbold correction.

    d_t = e1_t^2 - e2_t^2. H0: E[d_t] = 0 (equal accuracy). Negative statistic => model 1 better.
    Long-run variance with h-1 autocovariance lags; HLN factor; Student-t with T-1 dof.
    Returns (statistic, two-sided p-value).
    """
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    T = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[: T - k] - dbar)) / T for k in range(h)]   # 1/T normalisation
    var = (gamma[0] + 2 * sum(gamma[1:])) / T
    if not var > 0:                                   # identical forecasts (e.g. AR(0) = naive): no test
        return float('nan'), float('nan')
    dm = dbar / np.sqrt(var)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = hln * dm
    pval = 2 * stats.t.sf(abs(stat), df=T - 1)
    return float(stat), float(pval)


def ljung_box_p(e: np.ndarray, lags: int = 10) -> float:
    return float(acorr_ljungbox(e, lags=[lags])["lb_pvalue"].iloc[0])


def variance_ratio(r: np.ndarray, q: int) -> tuple[float, float, float]:
    """Lo-MacKinlay VR(q) with overlapping q-sums and the heteroskedasticity-robust z*.

    VR(q) = Var(r[n] + ... + r[n-q+1]) / (q Var(r[n])) = 1 + 2 sum_{k=1}^{q-1} (1 - k/q) rho(k),
    which is 1 for a random walk; VR > 1 means multi-day trends, VR < 1 mean reversion.
    Returns (VR, z*, two-sided p-value). Under H0 (uncorrelated increments, possibly
    heteroskedastic) z* is asymptotically N(0, 1).
    """
    x = np.asarray(r, float) - np.mean(r)
    N = len(x)
    v1 = np.sum(x ** 2) / N
    s = np.convolve(x, np.ones(q), mode="valid")          # overlapping q-day sums
    vq = np.sum(s ** 2) / (len(s) * q)
    vr = vq / v1
    theta = 0.0
    den = np.sum(x ** 2) ** 2
    for k in range(1, q):
        delta_k = N * np.sum(x[k:] ** 2 * x[:-k] ** 2) / den
        theta += (2 * (q - k) / q) ** 2 * delta_k
    z = (vr - 1) / np.sqrt(theta / N)
    return float(vr), float(z), float(2 * stats.norm.sf(abs(z)))


def lag1(r: np.ndarray) -> float:
    x = np.asarray(r, float) - np.mean(r)
    return float(np.dot(x[1:], x[:-1]) / np.dot(x, x))


# --------------------------------------------------------------------------------------
# Testing
# --------------------------------------------------------------------------------------
def evaluate_test(s: Series, rhat_all: dict) -> tuple[pd.DataFrame, dict]:
    """Metrics table on the test block for every model; DM test against the naive model."""
    pos = s.test
    m = s.mask[pos]
    p_prev, p_true = s.p[pos - 1][m], s.p[pos][m]
    e_naive = (p_true - p_prev)
    rows, price_pred = [], {}
    for name, rh in rhat_all.items():
        p_pred = s.p[pos - 1] * np.exp(rh[pos])
        price_pred[name] = p_pred
        pp = p_pred[m]
        row = price_metrics(p_prev, p_true, pp)
        row["MSE r [1e-4]"] = 1e4 * s.mse_r(pos, rh[pos])
        row["Rel. MSE vs naive"] = row["MSE r [1e-4]"] / (1e4 * s.mse_r(pos, np.zeros(len(pos))))
        e = p_true - pp
        if name.startswith("Naive"):
            row["DM stat"], row["DM p"] = np.nan, np.nan
        else:
            row["DM stat"], row["DM p"] = dm_test(e, e_naive)
        row["LB p (resid.)"] = ljung_box_p(s.r[pos][m] - rh[pos][m])
        rows.append(pd.Series(row, name=name))
    return pd.DataFrame(rows), price_pred


def run_all(df: pd.DataFrame, col: str = "Close", best: dict | None = None):
    """Tune (unless `best` is given), predict and evaluate. Returns a dict with everything."""
    s = Series(df, col)
    grid = None
    if best is None:
        best, grid = tune(s)
    rhat = {"Naive (random walk)": pred_naive(s), "Mean (drift)": pred_mean(s)}
    for name, f in PREDICTORS.items():
        label = f"AR({best['AR']['p']})" if name == "AR" else name
        rhat[label] = f(s, **best[name])
    table, price_pred = evaluate_test(s, rhat)
    return {"series": s, "best": best, "grid": grid, "rhat": rhat, "table": table, "price_pred": price_pred}


def horizon_errors(s: Series, best: dict, horizons=(1, 5, 10)) -> pd.DataFrame:
    """RMSE (USD) of h-step price forecasts made from every test origin.

    Forecast of p[n+h-1] made at time n-1:  p_hat = p[n-1] * exp(sum of predicted returns).
    Naive: no change. AR: iterated forecast. RLS / Kalman TV-AR: weights frozen at time n-1
    and iterated like an AR model. Local level: flat forecast of the filtered level.
    """
    N = len(s.r)
    pA = best["AR"]["p"]
    _, W_rls = rls_predict(s.r, best["RLS"]["p"], best["RLS"]["lam"], P0_scale=1.0 / s.sigma2_fit)
    _, W_tv = kalman_tvar(s.r, best["Kalman TV-AR"]["p"], best["Kalman TV-AR"]["q"], s.sigma2_fit)
    ratio = best["Kalman local level"]["ratio"]
    _, s_filt, _ = kalman_local_level(s.x, ratio * s.sigma2_fit, s.sigma2_fit, P0=s.sigma2_fit)
    out = []
    for h in horizons:
        errs = {k: [] for k in ("Naive (random walk)", f"AR({pA})", "RLS", "Kalman TV-AR", "Kalman local level")}
        for n in s.test:
            tgt = n + h - 1
            if tgt >= N or not s.mask[tgt]:
                continue
            hist = s.r[1:n]
            true = s.p[tgt]
            base = s.p[n - 1]
            a, mu, _ = ar_fit(hist, pA, demean=False)
            errs["Naive (random walk)"].append(true - base)
            errs[f"AR({pA})"].append(true - base * np.exp(ar_predict_h(hist, a, mu, h).sum()))
            errs["RLS"].append(true - base * np.exp(ar_predict_h(hist, W_rls[n - 1], 0.0, h).sum()))
            errs["Kalman TV-AR"].append(true - base * np.exp(ar_predict_h(hist, W_tv[n - 1], 0.0, h).sum()))
            errs["Kalman local level"].append(true - np.exp(s_filt[n - 1]))
        row = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in errs.items()}
        row["h"] = h
        row["n"] = len(errs["Naive (random walk)"])
        out.append(row)
    return pd.DataFrame(out).set_index("h")


def year_table(r: pd.Series, min_days: int = 200) -> pd.DataFrame:
    """Full-history check: one row per calendar year (>= `min_days` returns) plus a row for all years.

    Per year: lag-1 correlation, Ljung-Box p at lag 10 on r (linear dependence) and on r^2
    (volatility clustering), and the Lo-MacKinlay variance ratio for q = 5 and 20.
    """
    rows = []
    groups = [(str(y), g) for y, g in r.groupby(r.index.year) if len(g) >= min_days]
    groups.append(("all", r))
    for name, g in groups:
        x = g.to_numpy()
        vr5, z5, _ = variance_ratio(x, 5)
        vr20, z20, _ = variance_ratio(x, 20)
        rows.append({
            "year": name, "N": len(x), "std": x.std(ddof=1), "rho1": lag1(x),
            "LB10 p (r)": ljung_box_p(x, 10), "LB10 p (r^2)": ljung_box_p(x ** 2, 10),
            "VR(5)": vr5, "z*(5)": z5, "VR(20)": vr20, "z*(20)": z20,
        })
    return pd.DataFrame(rows).set_index("year")
