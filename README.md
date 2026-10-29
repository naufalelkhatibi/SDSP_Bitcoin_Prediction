# Bitcoin Price Prediction with Statistical Signal Processing

**EE4795 Statistical Digital Signal Processing**, TU Delft 2026/27 Q1, Project 9
Naufal el Khatibi (5315778) · Omar Shousha (6026508)

📄 **Report:** [`report/report.pdf`](report/report.pdf) · 📓 **Code:** [`code/btc_prediction.ipynb`](code/btc_prediction.ipynb)

## Summary

Can the course predictors (AR via Levinson–Durbin, FIR Wiener, LMS, NLMS, RLS and Kalman filters) forecast the next day's BTC-USD closing price better than the random walk (*tomorrow = today*)?

**No.** Daily log-returns are close to white noise, so the best linear predictor is (almost) zero and no course predictor beats the random walk.

| Model | Rel. MSE vs random walk (2021 test) | DM p | Holm p |
|---|---|---|---|
| Drift | 1.008 | 0.59 | 1.00 |
| AR(4) | 1.039 | 0.20 | 0.80 |
| ARMA(1,1) | 1.012 | 0.61 | 1.00 |
| LMS | 1.179 | 0.02 | 0.14 |
| NLMS | 1.117 | 0.01 | 0.10 |
| RLS | 1.205 | 0.03 | 0.16 |
| Kalman TV-AR | 1.225 | 0.02 | 0.13 |
| Kalman local level | 1.025 | 0.37 | 1.00 |

Rel. MSE > 1 means worse than the random walk. 73 test days (25 May – 5 Aug 2021); Diebold–Mariano test against the random walk, Holm-adjusted for multiple models.

- The adaptive filters were about 11 % better on the validation block, which contained temporary autocorrelation around the May 2021 crash, but 12–23 % worse on test.
- On recent data (Oct 2025 – Oct 2026), re-tuning drives AR and the local-level Kalman filter to the random walk itself.
- Over the full history (2014–2026), the returns are close to white in most years, while volatility clustering is significant in 7 of 12 years.

## Quick start

Python 3.10+:

```bash
pip install -r requirements.txt
cd code
jupyter nbconvert --to notebook --execute --inplace btc_prediction.ipynb   # about 1 min
```

The notebook shows every number, table and figure in the report and also saves them to `code/results/`.

## Repository structure

```
report/
  report.pdf             5-page report
code/
  btc_prediction.ipynb   analysis notebook, sections numbered as in the report
  sdsp/
    data.py              loading, cleaning (gap filling), chronological fit/validation/test split
    models.py            Levinson–Durbin, AR, Wiener, LMS/NLMS, RLS, Kalman (from scratch, NumPy)
    evaluation.py        tuning, walk-forward test, metrics, Diebold–Mariano, Ljung–Box, variance ratio
  data/
    course/              course dataset (BTC-USD.csv) and its cross-check against Yahoo Finance
    recent/              recent data, 1 Oct 2025 – 15 Oct 2026
    full/                full daily close history, 17 Sep 2014 – 15 Oct 2026
requirements.txt
```

## Report ↔ notebook

| Report section | Notebook section |
|---|---|
| 2 Data and data model | 2.1 Data and cross-check · 2.2 Statistics · 2.3 Spectral analysis |
| 3 Methodology | 3.1 Wiener–AR equivalence · 3.2 Tuning on the validation block |
| 4 Results, Table 2, Figs. 3–4 | 4.1 Test results · 4.2 Why validation and test disagree |
| 4 Results, Table 3 | 4.3 Multi-step forecasts · 4.4 Open, High, Low · 4.5 Recent data |
| 4 Results, Table 4 | 4.6 Full-history robustness |

## Data

| Set | Source | Period |
|---|---|---|
| Course | Yahoo Finance (course file) | 6 Aug 2020 – 6 Aug 2021 |
| Recent | Yahoo Finance chart API, fetched 16 Oct 2026 | 1 Oct 2025 – 15 Oct 2026 |
| Full history | Yahoo Finance chart API, fetched 16 Oct 2026 | 17 Sep 2014 – 15 Oct 2026 |

SHA-256 checksums:
```
10c78a6e57372f187da4a65b21ddb500d3e4926b57fb0ec53049bc821c8f2044  code/data/course/BTC-USD.csv
75c28e68202326f2a8bfe4d27eea5ac25ede729c118c01b8fd03dba3afea70b6  code/data/recent/BTC-USD_2025-10-01_2026-10-15.csv
398dec71e024f35c295d00580e8c74d89078f1e45f000cfaa26839172ae3a0be  code/data/full/BTC-USD_close_2014-09-17_2026-10-15.csv
```

- **Missing days:** 9, 12 and 13 Oct 2020 are empty in the course file. They are filled by log-linear interpolation and never scored.
- **Incomplete last row:** 6 Aug 2021 was saved before the day closed (4.7 % below the final close) and is dropped.
- **Split:** fit 233 returns, validation 58 (28 Mar – 24 May 2021), test 73 (25 May – 5 Aug 2021).

## Methodology rules

- **No look-ahead:** the prediction for day *n* uses only data up to day *n − 1*.
- **No choice on the test block:** all hyper-parameters are tuned on the validation block, and the data-model statistics use pre-test data only.
