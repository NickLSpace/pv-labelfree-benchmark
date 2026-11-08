"""
NaN-aware hourly aggregation of PV power / generation with a minimum-coverage rule.

Why
---
The original notebook filled gaps with 0 BEFORE resampling (``fillna(0)`` then ``sum()/4`` for
15-min power and ``sum()`` for generation).  A missing 15-min value therefore lowered the hourly
mean by 25 %, and a fully missing hour became a "real" zero.  Downstream, a zero hour with
GHI > 200 W/m2 cannot be told apart from an outage, so ``infer_filled_mask`` had to guess and
masked every all-zero sunny day as missing - which also hid real full-day outages from the
detector.

Rule implemented here
---------------------
NaN is kept until aggregation; an hour is aggregated only from the values that exist and only if
the hour is covered by at least ``min_count`` of the ``n_expected`` native samples:

* power (instantaneous, W):  hourly value = mean of the available samples        (count >= min_count)
* generation (energy per interval): hourly value = mean(available) * n_expected  (count >= min_count)
  (``energy_mode="sum"``: plain ``sum(min_count=...)`` - only exact when the hour is complete)
* otherwise the hour is NaN (later: ``observed = False``; tensors get 0 only inside the model)

Defaults: ``min_count = 3`` of 4 (15-min data).  Strict: 4 of 4.  Lenient: 2 of 4 (mean of the
two available values, i.e. the mean is recomputed over the available samples).  Hourly-native
channels (n_expected = 1) need their single value.

Negative values are clipped to 0 but NaN is preserved.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

MIN_COUNT_DEFAULT = 3
COVERAGE_PRESETS = {"strict": 4, "default": 3, "lenient": 2}


def _counts(s: pd.Series, freq: str) -> pd.Series:
    return s.notna().resample(freq).sum()


def aggregate_column(s: pd.Series, kind: str = "power", n_expected: int = 4, min_count: int = MIN_COUNT_DEFAULT,
                     freq: str = "1h", energy_mode: str = "mean_rescale") -> pd.Series:
    """Aggregate ONE native-resolution series (possibly on a finer merge grid) to ``freq``.

    kind        : "power" (mean) | "generation" (energy: mean * n_expected or sum)
    n_expected  : native samples per output period (4 for 15-min data, 1 for hourly data)
    min_count   : minimum number of available samples; clipped to [1, n_expected]
    """
    s = pd.to_numeric(s, errors="coerce").clip(lower=0.0)          # NaN preserved
    k = int(np.clip(min_count, 1, max(int(n_expected), 1)))
    r = s.resample(freq)
    cnt = r.count()
    if kind == "power":
        out = r.mean()
    elif kind == "generation":
        out = r.mean() * float(n_expected) if energy_mode == "mean_rescale" else r.sum(min_count=k)
    else:
        raise ValueError(kind)
    return out.where(cnt >= k)


def aggregate_pv(df: pd.DataFrame, power_cols: Sequence[str] = (), gen_cols: Sequence[str] = (),
                 n_expected: Optional[Dict[str, int]] = None, default_n: int = 4,
                 min_count: int = MIN_COUNT_DEFAULT, freq: str = "1h",
                 energy_mode: str = "mean_rescale") -> Dict[str, pd.Series]:
    """All PV columns; ``n_expected[col]`` overrides ``default_n`` (e.g. 1 for hourly stations)."""
    n_expected = n_expected or {}
    out = {}
    for c in gen_cols:
        out[c] = aggregate_column(df[c], "generation", n_expected.get(c, default_n), min_count, freq, energy_mode)
    for c in power_cols:
        out[c] = aggregate_column(df[c], "power", n_expected.get(c, default_n), min_count, freq, energy_mode)
    return out


def legacy_aggregate_column(s: pd.Series, kind: str = "power", n_expected: int = 4, freq: str = "1h") -> pd.Series:
    """The previous behaviour (zero fill before aggregation) - kept only for the before/after check."""
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).clip(lower=0.0)
    r = s.resample(freq)
    if n_expected == 1:
        return r.max()
    return r.sum() / n_expected if kind == "power" else r.sum()


# ----------------------------------------------------------------------------- diagnostics
def coverage_table(df: pd.DataFrame, cols: Sequence[str], n_expected: Optional[Dict[str, int]] = None,
                   default_n: int = 4, freq: str = "1h", ghi: Optional[pd.Series] = None,
                   ghi_day: float = 200.0) -> pd.DataFrame:
    """Share of hours by number of available native samples (0 .. n_expected), daylight only if GHI given."""
    n_expected = n_expected or {}
    rows = []
    for c in cols:
        n = n_expected.get(c, default_n)
        cnt = _counts(df[c], freq)
        if ghi is not None:
            g = ghi.resample(freq).mean().reindex(cnt.index)
            cnt = cnt[g > ghi_day]
        r = {"column": c, "n_expected": n, "hours": int(len(cnt))}
        for k in range(n + 1):
            r[f"share_{k}_of_{n}"] = float((cnt == k).mean()) if len(cnt) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def compare_aggregation(df: pd.DataFrame, power_cols: Sequence[str], n_expected: Optional[Dict[str, int]] = None,
                        default_n: int = 4, ghi: Optional[pd.Series] = None, ghi_day: float = 200.0,
                        min_counts: Iterable[int] = (4, 3, 2), freq: str = "1h") -> pd.DataFrame:
    """Before / after table for the aggregation rule (power columns, daylight hours with GHI > ghi_day).

    Columns: share of daylight hours that are zero (legacy) / NaN (new); hours with partial coverage
    and the mean relative bias of the legacy value on those hours (legacy / NaN-aware - 1)."""
    n_expected = n_expected or {}
    gh = ghi.resample(freq).mean() if ghi is not None else None
    rows = []
    for mode in ["legacy (zero fill)"] + [f"min_count={k}" for k in min_counts]:
        z, nanh, part, bias, tot = 0, 0, 0, [], 0
        for c in power_cols:
            n = n_expected.get(c, default_n)
            leg = legacy_aggregate_column(df[c], "power", n, freq)
            cnt = _counts(df[c], freq).reindex(leg.index)
            day = (gh.reindex(leg.index) > ghi_day) if gh is not None else pd.Series(True, index=leg.index)
            tot += int(day.sum())
            if mode.startswith("legacy"):
                z += int(((leg <= 0) & day).sum())
                pm = (cnt > 0) & (cnt < n) & day
                part += int(pm.sum())
                new = aggregate_column(df[c], "power", n, 1, freq)
                with np.errstate(all="ignore"):
                    b = (leg[pm] / new[pm] - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
                bias += list(b.to_numpy())
            else:
                k = int(mode.split("=")[1])
                new = aggregate_column(df[c], "power", n, k, freq)
                z += int(((new <= 0) & day).sum())
                nanh += int((new.isna() & day).sum())
                part += int(((cnt >= min(k, n)) & (cnt < n) & day).sum())
        rows.append({"aggregation": mode, "daylight hours": tot,
                     "zero hours, %": 100 * z / tot if tot else np.nan,
                     "NaN hours, %": 100 * nanh / tot if tot else np.nan,
                     "partially covered hours used, %": 100 * part / tot if tot else np.nan,
                     "mean bias of partial hours, %": 100 * float(np.mean(bias)) if bias else np.nan})
    return pd.DataFrame(rows)
