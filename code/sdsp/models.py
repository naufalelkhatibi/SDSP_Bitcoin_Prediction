"""Prediction models for the log-return r[n] (all implemented from scratch with NumPy).

Every one-step predictor here produces r_hat[n] = prediction of r[n] using r[0..n-1] only.
The models follow Hayes (1996): Ch. 4 (all-pole modelling), Ch. 5 (Levinson-Durbin), Ch. 7 (Wiener
filtering / linear prediction, Sec. 7.4 Kalman filter), Ch. 8 (spectrum estimation) and Ch. 9 (LMS, NLMS, RLS).

Notation: p = model order, a = [a_1..a_p] with the AR model
    r[n] - mu = sum_k a_k (r[n-k] - mu) + e[n],   e[n] white, variance sigma^2.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_toeplitz


# --------------------------------------------------------------------------------------
# Correlation estimates and Levinson-Durbin
# --------------------------------------------------------------------------------------
def autocorr(x: np.ndarray, maxlag: int) -> np.ndarray:
    """Biased autocorrelation estimate r_x(k) = (1/N) sum_n x[n] x[n-k], k = 0..maxlag.

    The biased estimate guarantees a positive (semi-)definite Toeplitz matrix, so the
    Yule-Walker / Wiener-Hopf equations are always solvable and the AR model is stable.
    """
    x = np.asarray(x, float)
    N = len(x)
    return np.array([np.dot(x[k:], x[: N - k]) / N for k in range(maxlag + 1)])


def levinson_durbin(rx: np.ndarray, p: int):
    """Solve the Yule-Walker equations recursively (Hayes, Ch. 5).

    Returns
        a      : AR coefficients a_1..a_p in the *prediction* convention x_hat[n] = sum a_k x[n-k]
        eps    : prediction-error power for every order 0..p (eps[j] = sigma^2 of AR(j))
        gamma  : reflection coefficients gamma_1..gamma_p (|gamma| < 1 <=> stable)
    """
    rx = np.asarray(rx, float)
    a = np.zeros(0)
    eps = [rx[0]]
    gammas = []
    for j in range(p):
        # k_{j+1} = (r(j+1) - sum_i a_i r(j+1-i)) / eps_j. Hayes writes A(z) = 1 + sum a_k z^-k,
        # i.e. his a_k and gamma_j have the opposite sign of ours (prediction convention).
        acc = rx[j + 1] - np.dot(a, rx[j:0:-1]) if j > 0 else rx[1]
        k = acc / eps[-1]
        a = np.concatenate([a - k * a[::-1], [k]])
        eps.append(eps[-1] * (1 - k ** 2))
        gammas.append(k)
    return a, np.array(eps), np.array(gammas)


def ar_fit(x: np.ndarray, p: int, demean: bool = True):
    """Yule-Walker AR(p) fit. Returns (a, mu, sigma2)."""
    x = np.asarray(x, float)
    mu = x.mean() if demean else 0.0
    rx = autocorr(x - mu, p)
    if p == 0:
        return np.zeros(0), mu, rx[0]
    a, eps, _ = levinson_durbin(rx, p)
    return a, mu, eps[-1]


def ar_order_criteria(x: np.ndarray, pmax: int, demean: bool = True) -> dict:
    """AIC(p) = N ln eps_p + 2p and MDL(p) = N ln eps_p + p ln N (Hayes, Ch. 8, parametric methods)."""
    x = np.asarray(x, float)
    N = len(x)
    mu = x.mean() if demean else 0.0
    _, eps, gam = levinson_durbin(autocorr(x - mu, pmax), pmax)
    p = np.arange(pmax + 1)
    return {
        "p": p,
        "eps": eps,
        "AIC": N * np.log(eps) + 2 * p,
        "MDL": N * np.log(eps) + p * np.log(N),
        "pacf": np.concatenate([[1.0], gam]),  # reflection coefficients = sample PACF
    }


def ar_predict_next(x_hist: np.ndarray, a: np.ndarray, mu: float) -> float:
    """One-step prediction x_hat[n] = mu + sum_k a_k (x[n-k] - mu)."""
    p = len(a)
    if p == 0:
        return mu
    past = x_hist[-1 : -p - 1 : -1] - mu  # x[n-1], x[n-2], ..., x[n-p]
    return float(mu + np.dot(a, past))


def ar_predict_h(x_hist: np.ndarray, a: np.ndarray, mu: float, h: int) -> np.ndarray:
    """Iterated h-step forecast: predictions are fed back as inputs."""
    buf = list(np.asarray(x_hist, float))
    out = []
    for _ in range(h):
        nxt = ar_predict_next(np.array(buf), a, mu)
        out.append(nxt)
        buf.append(nxt)
    return np.array(out)


def ar_spectrum(a: np.ndarray, sigma2: float, nfft: int = 1024):
    """AR power spectrum P(e^jw) = sigma^2 / |1 - sum a_k e^{-jwk}|^2 on w in [0, pi]."""
    w = np.linspace(0, np.pi, nfft // 2 + 1)
    A = 1 - sum(a[k] * np.exp(-1j * w * (k + 1)) for k in range(len(a)))
    return w, sigma2 / np.abs(A) ** 2


# --------------------------------------------------------------------------------------
# Wiener FIR one-step predictor
# --------------------------------------------------------------------------------------
def wiener_predictor(x: np.ndarray, L: int, demean: bool = True):
    """FIR Wiener filter w (length L) that predicts x[n] from x[n-1..n-L].

    Wiener-Hopf equations:  R_x w = r_dx,  with R_x = Toeplitz(r_x(0..L-1)) and
    r_dx = [r_x(1), ..., r_x(L)] because the desired signal is d[n] = x[n] (one step ahead).
    These are exactly the Yule-Walker equations, so w equals the AR(L) coefficients.
    Solved here directly (Toeplitz solver), independent of the Levinson recursion,
    so the equality is a real numerical check. Minimum MSE: xi_min = r_x(0) - w^T r_dx.
    """
    x = np.asarray(x, float)
    mu = x.mean() if demean else 0.0
    rx = autocorr(x - mu, L)
    w = solve_toeplitz(rx[:L], rx[1 : L + 1])
    xi_min = rx[0] - np.dot(w, rx[1 : L + 1])
    return w, mu, xi_min


# --------------------------------------------------------------------------------------
# Adaptive filters (run sequentially over the whole series)
# --------------------------------------------------------------------------------------
def _regressor(x: np.ndarray, n: int, p: int) -> np.ndarray:
    """phi[n] = [x[n-1], ..., x[n-p]] (zeros before the start)."""
    v = np.zeros(p)
    for k in range(1, p + 1):
        if n - k >= 0:
            v[k - 1] = x[n - k]
    return v


def lms_predict(x: np.ndarray, p: int, mu_step: float, normalized: bool = False,
                eps: float = 1e-8, w0: np.ndarray | None = None):
    """(N)LMS one-step predictor. Returns predictions x_hat[n] (made BEFORE seeing x[n]) and weights.

    LMS :  w[n+1] = w[n] + mu e[n] phi[n]
    NLMS:  w[n+1] = w[n] + mu e[n] phi[n] / (eps + ||phi[n]||^2),   0 < mu < 2
    with e[n] = x[n] - w[n]^T phi[n].
    """
    x = np.asarray(x, float)
    N = len(x)
    w = np.zeros(p) if w0 is None else np.array(w0, float)
    xhat = np.zeros(N)
    W = np.zeros((N, p))
    for n in range(N):
        phi = _regressor(x, n, p)
        xhat[n] = w @ phi
        e = x[n] - xhat[n]
        step = mu_step / (eps + phi @ phi) if normalized else mu_step
        w = w + step * e * phi
        W[n] = w
    return xhat, W


def rls_predict(x: np.ndarray, p: int, lam: float, P0_scale: float = 100.0):
    """Exponentially weighted RLS one-step predictor (Hayes, Sec. 9.4).

    k[n]   = P[n-1] phi / (lam + phi^T P[n-1] phi)
    alpha  = x[n] - w[n-1]^T phi            (a-priori error -> this is our prediction error)
    w[n]   = w[n-1] + k[n] alpha
    P[n]   = (P[n-1] - k[n] phi^T P[n-1]) / lam,   P[0] = P0_scale * I
    (the course writes P[0] = delta^{-1} I, so P0_scale = 1/delta)
    """
    x = np.asarray(x, float)
    N = len(x)
    w = np.zeros(p)
    P = P0_scale * np.eye(p)
    xhat = np.zeros(N)
    W = np.zeros((N, p))
    for n in range(N):
        phi = _regressor(x, n, p)
        xhat[n] = w @ phi
        alpha = x[n] - xhat[n]
        Pphi = P @ phi
        k = Pphi / (lam + phi @ Pphi)
        w = w + k * alpha
        P = (P - np.outer(k, Pphi)) / lam
        W[n] = w
    return xhat, W


# --------------------------------------------------------------------------------------
# Kalman filters
# --------------------------------------------------------------------------------------
def kalman_local_level(y: np.ndarray, q: float, r: float, s0: float | None = None, P0: float = 1.0):
    """Scalar Kalman filter for the local-level (random walk + noise) model on the log-price:

        state:        s[n] = s[n-1] + w[n],     w ~ N(0, q)
        observation:  y[n] = s[n] + v[n],       v ~ N(0, r)

    Returns the one-step predictions y_hat[n|n-1] = s_hat[n|n-1] = s_hat[n-1|n-1],
    the filtered states and the Kalman gains. Only the ratio q/r matters for the mean.
    In steady state this is exponential smoothing with factor K_inf.
    """
    y = np.asarray(y, float)
    N = len(y)
    s = y[0] if s0 is None else s0
    P = P0
    yhat = np.zeros(N)
    sf = np.zeros(N)
    K = np.zeros(N)
    for n in range(N):
        # predict
        s_pred, P_pred = s, P + q
        yhat[n] = s_pred
        # update
        k = P_pred / (P_pred + r)
        s = s_pred + k * (y[n] - s_pred)
        P = (1 - k) * P_pred
        sf[n], K[n] = s, k
    return yhat, sf, K


def kalman_tvar(x: np.ndarray, p: int, q: float, r: float, P0: float = 1.0):
    """Kalman filter with the AR coefficients as a random-walk state (time-varying AR):

        state:        a[n] = a[n-1] + w[n],          w ~ N(0, q I)
        observation:  x[n] = phi[n]^T a[n] + v[n],   v ~ N(0, r),   phi[n] = [x[n-1..n-p]]

    With q = 0 this reduces to RLS with lam = 1 (Hayes discusses RLS in Sec. 9.4 and the Kalman filter in Sec. 7.4).
    """
    x = np.asarray(x, float)
    N = len(x)
    a = np.zeros(p)
    P = P0 * np.eye(p)
    Q = q * np.eye(p)
    xhat = np.zeros(N)
    A = np.zeros((N, p))
    for n in range(N):
        phi = _regressor(x, n, p)
        P = P + Q                       # predict (state transition = identity)
        xhat[n] = phi @ a
        S = phi @ P @ phi + r           # innovation variance
        k = P @ phi / S
        a = a + k * (x[n] - xhat[n])    # update
        P = P - np.outer(k, phi @ P)
        A[n] = a
    return xhat, A
