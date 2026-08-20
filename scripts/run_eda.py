#!/usr/bin/env python
"""EDA over the three raw source tables.

Writes reports/eda/EDA.md, reports/eda/eda_report.json and reports/eda/figures/.
Read-only with respect to data/raw.

    python scripts/run_eda.py
"""

import argparse
import json
from itertools import permutations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

TARGET = "avg_duration_months"
NUMERIC = [
    "autocall_barrier_pct",
    "protection_barrier_pct",
    "no_call_period_months",
    "quoted_implied_vol",
    "notional_credits",
    "tenor_months",
    "basket_size",
]
CATEGORICAL = ["product_type", "basket_type", "observation_frequency", "counterparty", "trader_id"]
DERIVED = ["tenor_months", "basket_size", "obs_freq_months", "duration_ratio"]

# observation_frequency arrives in English, Spanish and code form; all of it
# means "months between autocall observations".
FREQ_MONTHS = {
    "1d": 1 / 21, "daily": 1 / 21, "diario": 1 / 21,
    "1m": 1, "m": 1, "monthly": 1, "mensual": 1, "1 month": 1,
    "2m": 2, "2 months": 2,
    "3m": 3, "q": 3, "quarterly": 3, "trimestral": 3, "3 months": 3,
    "6m": 6, "semiannual": 6, "semestral": 6, "6 months": 6,
    "1y": 12, "12m": 12, "y": 12, "annual": 12, "anual": 12, "12 months": 12,
}


def load(raw_dir):
    rfqs = pd.read_csv(
        raw_dir / "rfqs.csv", parse_dates=["requested_date", "start_date", "end_date"]
    )
    vol = pd.read_csv(raw_dir / "daily_volatility.csv", parse_dates=["date"])
    ref = pd.read_csv(raw_dir / "underlyings_reference.csv")

    rfqs["tenor_months"] = (rfqs.end_date - rfqs.start_date).dt.days / 30.44
    rfqs["basket_size"] = rfqs.underlyings.str.split("|").str.len()
    rfqs["obs_freq_months"] = rfqs.observation_frequency.str.strip().str.lower().map(FREQ_MONTHS)
    rfqs["duration_ratio"] = rfqs[TARGET] / rfqs.tenor_months

    vol = vol.sort_values(["underlying", "date"])
    vol["gap_days"] = vol.groupby("underlying").date.diff().dt.days
    return rfqs, vol, ref


def profile(df):
    """One row per column: type, nulls, cardinality."""
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "nulls": df.isna().sum().values,
            "% null": (df.isna().mean() * 100).round(2).values,
            "distinct": df.nunique().values,
        }
    )


def eta_squared(values, groups):
    """Share of the variance in `values` explained by the grouping."""
    grand = values.mean()
    stats = values.groupby(groups, dropna=False).agg(["size", "mean"])
    between = (stats["size"] * (stats["mean"] - grand) ** 2).sum()
    return between / ((values - grand) ** 2).sum()


def variance_explained(df, cols, target, draws=20, seed=0):
    """eta squared per categorical, next to the value random labels would give.

    Splitting n rows into k groups explains some variance by construction, and
    the more levels a column has the more it gets for free, so the shuffled
    baseline is what makes the raw number readable.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for col in cols:
        sub = df[[col, target]].dropna()
        noise = [eta_squared(sub[target], rng.permutation(sub[col].values)) for _ in range(draws)]
        rows.append(
            {
                "column": col,
                "levels": sub[col].nunique(),
                "eta squared": round(eta_squared(sub[target], sub[col]), 4),
                "shuffled": round(float(np.mean(noise)), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("eta squared", ascending=False, ignore_index=True)


def correlation_by_group(df, value_col, group_col, target):
    """Spearman correlation of value_col vs target, computed within each group_col level."""
    rows = []
    for level, g in df.groupby(group_col):
        sub = g[[value_col, target]].dropna()
        rows.append(
            {
                group_col: level,
                "n": len(sub),
                "spearman vs target": round(sub[value_col].corr(sub[target], "spearman"), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman vs target", ignore_index=True)


def r_squared(df, y_col, x_cols):
    """OLS R^2 of y_col on x_cols, dropping rows with nulls in either."""
    sub = df[[y_col] + x_cols].dropna()
    x = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in x_cols])
    y = sub[y_col].values
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    return 1 - (resid**2).sum() / ((y - y.mean()) ** 2).sum()


def volatility_overlap(df):
    """How much of the quoted vol is already implied by the other vol measures."""
    combos = [
        ("structural_base_vol only", ["sb_mean"]),
        ("realized_vol_63d only", ["rv_mean"]),
        ("market-wide level only", ["mkt_vol"]),
        ("structural + realized", ["sb_mean", "rv_mean"]),
        ("structural + realized + market-wide", ["sb_mean", "rv_mean", "mkt_vol"]),
    ]
    return pd.DataFrame(
        {"predicting quoted_implied_vol from": name, "R^2": round(r_squared(df, "quoted_implied_vol", cols), 3)}
        for name, cols in combos
    )


def stationarity(vol, alpha=0.05):
    """Augmented Dickey-Fuller test of realized_vol_63d, per underlying.

    Null hypothesis is a unit root (non-stationary); a low p-value rejects it.
    Run before looking at periodicity, since an FFT peak is only meaningful as
    a cycle once the series is known to oscillate around a fixed level rather
    than drift.
    """
    rows = []
    for underlying, g in vol.groupby("underlying"):
        series = g.sort_values("date").realized_vol_63d.dropna()
        stat, pvalue, lags, nobs, _, _ = adfuller(series, autolag="AIC")
        rows.append(
            {
                "underlying": underlying,
                "n_obs": nobs,
                "lags": lags,
                "ADF statistic": round(stat, 3),
                "p-value": round(pvalue, 4),
                "stationary at 5%": pvalue < alpha,
            }
        )
    return pd.DataFrame(rows).sort_values("underlying", ignore_index=True)


def dominant_period(series):
    """Trading-day period of the strongest non-zero frequency in an FFT."""
    x = (series - series.mean()).values
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1)
    i = np.argmax(power[1:]) + 1  # skip the zero-frequency (mean) bin
    return 1 / freqs[i]


def periodicity(vol):
    """Dominant FFT cycle length of realized_vol_63d, per underlying."""
    rows = []
    for underlying, g in vol.groupby("underlying"):
        period = dominant_period(g.sort_values("date").realized_vol_63d)
        rows.append(
            {
                "underlying": underlying,
                "n_obs": len(g),
                "dominant period (trading days)": round(period),
                "dominant period (years)": round(period / 252, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("dominant period (trading days)", ignore_index=True)


def dependencies(df, cols, threshold=0.999):
    """Column pairs where the first one determines the second."""
    found = []
    for a, b in permutations(cols, 2):
        sub = df[[a, b]].dropna()
        modal_share = sub.groupby(a)[b].apply(lambda s: s.value_counts().iloc[0] / len(s))
        weights = sub.groupby(a)[b].size()
        purity = (modal_share * weights).sum() / weights.sum()
        if purity >= threshold:
            found.append(
                {"column": a, "determines": b, "purity": round(purity, 3), "levels": sub[a].nunique()}
            )
    return pd.DataFrame(found)


def quality_checks(rfqs, vol, ref, legs):
    """Fixed list of conditions, run every time. Passing checks are reported too."""
    executed = rfqs.executed.astype(bool)
    labelled = rfqs[TARGET].notna()
    unmapped = rfqs.observation_frequency[rfqs.obs_freq_months.isna()].unique()

    rows = [
        ("rfqs", "duplicated rfq_id", rfqs.rfq_id.duplicated().sum()),
        ("rfqs", f"executed, no {TARGET}", (executed & ~labelled).sum()),
        ("rfqs", f"not executed, has {TARGET}", (~executed & labelled).sum()),
        ("rfqs", "not executed, has start/end dates", (~executed & rfqs.end_date.notna()).sum()),
        ("rfqs", f"negative {TARGET}", (rfqs[TARGET] < 0).sum()),
        ("rfqs", f"{TARGET} above nominal tenor", (rfqs[TARGET] > rfqs.tenor_months).sum()),
        ("rfqs", f"{TARGET} below no-call period", (rfqs[TARGET] < rfqs.no_call_period_months).sum()),
        ("rfqs", "end_date before start_date", (rfqs.end_date < rfqs.start_date).sum()),
        ("rfqs", "start_date before requested_date", (rfqs.start_date < rfqs.requested_date).sum()),
        ("rfqs", "observation_frequency labels", rfqs.observation_frequency.nunique()),
        ("rfqs", "  of which unmapped", len(unmapped)),
        ("rfqs", "  distinct intervals they encode", rfqs.obs_freq_months.nunique()),
        ("rfqs", "columns with nulls", (rfqs[NUMERIC + CATEGORICAL].isna().any()).sum()),
        ("daily_volatility", "duplicated (date, underlying)", vol.duplicated(["date", "underlying"]).sum()),
        ("daily_volatility", "null realized_vol_63d", vol.realized_vol_63d.isna().sum()),
        ("daily_volatility", "negative realized_vol_63d", (vol.realized_vol_63d < 0).sum()),
        ("daily_volatility", "gaps longer than 7 days", (vol.gap_days > 7).sum()),
        ("underlyings_reference", "duplicated underlying", ref.underlying.duplicated().sum()),
        ("underlyings_reference", "sectors / tickers", f"{ref.sector.nunique()} / {len(ref)}"),
        ("integration", "RFQ tickers missing from reference", legs.loc[~legs.in_reference, "underlying"].nunique()),
        ("integration", "RFQ tickers missing from vol panel", legs.loc[~legs.in_vol, "underlying"].nunique()),
        ("integration", "legs with no vol history at requested_date", (~legs.covered).sum()),
    ]
    return pd.DataFrame(rows, columns=["table", "check", "result"])


def make_figures(rfqs, vol, ref, fig_dir):
    """Save the plots and return [(filename, caption), ...]."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    for stale in fig_dir.glob("*.png"):
        stale.unlink()
    out = []

    def save(name, caption):
        plt.tight_layout()
        plt.savefig(fig_dir / name, dpi=130)
        plt.close()
        out.append((name, caption))

    lab = rfqs[rfqs[TARGET].notna()]

    _, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(lab[TARGET], bins=60, color="#3b6ea5", edgecolor="white")
    ax[0].set(xlabel="months", ylabel="RFQs", title=TARGET)
    ax[1].hist(lab.duration_ratio, bins=60, color="#a5533b", edgecolor="white")
    ax[1].set(xlabel="ratio", title=f"{TARGET} / nominal tenor")
    save("target_distribution.png", "Target, in months and as a fraction of the nominal tenor.")

    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(lab.tenor_months, lab[TARGET], s=4, alpha=0.15, color="#3b6ea5")
    plt.plot([0, lab.tenor_months.max()], [0, lab.tenor_months.max()], "k--", lw=1)
    plt.xlabel("nominal tenor (months)")
    plt.ylabel(TARGET)
    plt.title("Duration vs nominal tenor (dashed: equal)")
    save("target_vs_tenor.png", "Average duration against nominal tenor.")

    groups = lab.groupby("product_type")[TARGET]
    order = groups.median().sort_values().index
    plt.figure(figsize=(9, 5))
    plt.boxplot([groups.get_group(k) for k in order], tick_labels=order, showfliers=False)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(TARGET)
    plt.title("Target by product_type")
    save("target_by_product_type.png", "Target distribution per product type.")

    _, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, ["quoted_implied_vol", "autocall_barrier_pct", "obs_freq_months"]):
        sub = lab[[col, TARGET]].dropna()
        ax.scatter(sub[col], sub[TARGET], s=4, alpha=0.12, color="#3b6ea5")
        trend = sub.groupby(pd.qcut(sub[col], 20, duplicates="drop"), observed=True).mean()
        ax.plot(trend[col], trend[TARGET], color="#a5533b", lw=2)
        ax.set(xlabel=col, ylabel=TARGET)
    save("target_vs_drivers.png", "Target against three contract terms, with binned means.")

    cols = NUMERIC + ["obs_freq_months", TARGET]
    corr = lab[cols].corr("spearman")
    plt.figure(figsize=(8, 6.5))
    plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(shrink=0.8)
    plt.title("Spearman correlation")
    save("correlation_matrix.png", "Spearman correlation between numeric terms and the target.")

    monthly = vol.set_index("date").groupby([pd.Grouper(freq="MS"), "underlying"]).realized_vol_63d.mean()
    plt.figure(figsize=(12, 5.5))
    monthly.unstack().plot(ax=plt.gca(), lw=1)
    plt.ylabel("realized_vol_63d (monthly mean)")
    plt.title("Realised volatility by underlying")
    plt.legend(ncol=5, fontsize=7)
    save("volatility_timeseries.png", "Monthly mean realised volatility per underlying.")

    merged = ref.set_index("underlying").join(vol.groupby("underlying").realized_vol_63d.mean())
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(merged.structural_base_vol, merged.realized_vol_63d, color="#3b6ea5")
    for name, row in merged.iterrows():
        plt.annotate(name, (row.structural_base_vol, row.realized_vol_63d), fontsize=8,
                     xytext=(3, 3), textcoords="offset points")
    top = merged[["structural_base_vol", "realized_vol_63d"]].max().max() * 1.15
    plt.plot([0, top], [0, top], "k--", lw=1)
    plt.xlabel("structural_base_vol")
    plt.ylabel("mean realized_vol_63d")
    plt.title("Reference vs realised volatility")
    save("structural_vs_realized_vol.png", "Reference volatility against what the market showed.")

    return out


def build_report(rfqs, vol, ref):
    """Everything the report shows, as a dict of DataFrames and scalars."""
    lab = rfqs[rfqs[TARGET].notna()]
    executed = rfqs.executed.astype(bool)

    legs = rfqs.assign(underlying=rfqs.underlyings.str.split("|")).explode("underlying")
    legs = legs[["rfq_id", "requested_date", "underlying"]]
    legs["in_reference"] = legs.underlying.isin(ref.underlying)
    legs["in_vol"] = legs.underlying.isin(vol.underlying)
    legs["covered"] = legs.requested_date >= legs.underlying.map(vol.groupby("underlying").date.min())

    # Realised vol as-of the request date, plus the static structural vol, per leg,
    # then averaged across the legs of each basket.
    legs_vol = pd.merge_asof(
        legs.sort_values("requested_date"),
        vol[["date", "underlying", "realized_vol_63d"]].sort_values("date"),
        left_on="requested_date", right_on="date", by="underlying", direction="backward",
    ).merge(ref[["underlying", "structural_base_vol"]], on="underlying", how="left")
    vol_agg = legs_vol.groupby("rfq_id").agg(
        rv_mean=("realized_vol_63d", "mean"), sb_mean=("structural_base_vol", "mean")
    )
    rfqs_vol = rfqs.merge(vol_agg, on="rfq_id", how="left")

    # Market-wide level: the cross-sectional mean of realized_vol_63d across all 14
    # underlyings on a given day, joined as-of requested_date. Unlike rv_mean this
    # doesn't depend on which tickers are in the basket, so it reads the shared
    # regime cycle found in the periodicity analysis even for single-name RFQs.
    market_daily = (
        vol.groupby("date", as_index=False).realized_vol_63d.mean()
        .rename(columns={"realized_vol_63d": "mkt_vol"}).sort_values("date")
    )
    mkt_vol = pd.merge_asof(
        rfqs[["rfq_id", "requested_date"]].sort_values("requested_date"), market_daily,
        left_on="requested_date", right_on="date", direction="backward",
    )[["rfq_id", "mkt_vol"]]
    rfqs_vol = rfqs_vol.merge(mkt_vol, on="rfq_id", how="left")

    per_underlying = (
        vol.groupby("underlying")
        .agg(
            rows=("date", "size"),
            first=("date", "min"),
            last=("date", "max"),
            max_gap=("gap_days", "max"),
            mean=("realized_vol_63d", "mean"),
            median=("realized_vol_63d", "median"),
            std=("realized_vol_63d", "std"),
            min=("realized_vol_63d", "min"),
            max=("realized_vol_63d", "max"),
        )
        .reset_index()
        .round({"max_gap": 0, "mean": 3, "median": 3, "std": 3, "min": 3, "max": 3})
    )
    reference = ref.merge(
        vol.groupby("underlying").realized_vol_63d.mean().round(3).rename("mean_realised"),
        on="underlying",
    )

    by_category = {}
    for col in ["product_type", "obs_freq_months", "basket_type", "basket_size",
                "no_call_period_months"]:
        by_category[col] = (
            lab.groupby(col)[TARGET].agg(["count", "mean", "median", "std"]).round(2).reset_index()
        )

    return {
        "files": pd.DataFrame(
            [
                {"file": name, "rows": len(df), "columns": df.shape[1] - len(derived),
                 "duplicate rows": df.duplicated().sum()}
                for name, df, derived in [
                    ("rfqs.csv", rfqs, DERIVED),
                    ("daily_volatility.csv", vol, ["gap_days"]),
                    ("underlyings_reference.csv", ref, []),
                ]
            ]
        ),
        "rfq_columns": profile(rfqs.drop(columns=DERIVED)),
        "rfq_numeric": rfqs[NUMERIC].describe().T.round(3).reset_index(names="column"),
        "rfq_categorical": {
            col: (
                rfqs[col].value_counts().rename_axis("value").reset_index(name="count")
                .assign(share=lambda d: (d["count"] / len(rfqs) * 100).round(2))
                .head(20)
            )
            for col in CATEGORICAL
        },
        "frequency_map": (
            rfqs.obs_freq_months.value_counts().sort_index()
            .rename_axis("months between observations").reset_index(name="RFQs")
        ),
        "target": lab[[TARGET, "duration_ratio"]].describe().T.round(3).reset_index(names="column"),
        "correlations": (
            lab[NUMERIC + ["obs_freq_months", TARGET]]
            .corr("spearman")[TARGET].drop(TARGET).round(3)
            .rename_axis("feature").reset_index(name="spearman vs target")
            .sort_values("spearman vs target", key=abs, ascending=False)
        ),
        "by_category": by_category,
        "variance_explained": variance_explained(
            lab,
            CATEGORICAL + ["basket_size", "obs_freq_months", "no_call_period_months"],
            TARGET,
        ),
        "dependencies": dependencies(
            rfqs, ["product_type", "basket_type", "basket_size", "obs_freq_months",
                   "counterparty", "trader_id"]
        ),
        "per_underlying": per_underlying,
        "vol_stationarity": stationarity(vol),
        "vol_periodicity": periodicity(vol),
        "vol_seasonality": (
            vol.groupby(vol.date.dt.month).realized_vol_63d.mean().round(3)
            .rename_axis("month").reset_index(name="mean realized_vol_63d")
        ),
        "reference": reference,
        "vol_correlation": (
            rfqs_vol[["quoted_implied_vol", "rv_mean", "sb_mean", "mkt_vol"]]
            .corr().round(3).rename_axis("feature").reset_index()
        ),
        "vol_r2": volatility_overlap(rfqs_vol),
        "vol_target_correlation": (
            rfqs_vol[rfqs_vol[TARGET].notna()][["rv_mean", "sb_mean", "mkt_vol", TARGET]]
            .corr("spearman")[TARGET].drop(TARGET).round(3)
            .rename_axis("feature").reset_index(name="spearman vs target")
        ),
        "mkt_vol_by_product": correlation_by_group(
            rfqs_vol[rfqs_vol[TARGET].notna()], "mkt_vol", "product_type", TARGET
        ),
        "checks": quality_checks(rfqs, vol, ref, legs),
        "scalars": {
            "n_rfqs": len(rfqs),
            "n_executed": int(executed.sum()),
            "n_labelled": len(lab),
            "target_above_tenor": int((rfqs[TARGET] > rfqs.tenor_months).sum()),
            "target_at_tenor": int((lab[TARGET] >= lab.tenor_months - 0.5).sum()),
            "requested_date": [str(rfqs.requested_date.min().date()), str(rfqs.requested_date.max().date())],
            "vol_panel": [str(vol.date.min().date()), str(vol.date.max().date())],
            "legs": len(legs),
            "legs_covered": int(legs.covered.sum()),
            "structural_vs_realised_corr": round(
                reference.structural_base_vol.corr(reference.mean_realised), 3
            ),
        },
    }


def table(df, floatfmt=",.3f"):
    """Markdown table without the scientific notation pandas defaults to.

    tabulate applies floatfmt to every numeric column in an all-numeric frame,
    even int ones, unless at least one column is non-numeric, so integer
    columns are cast to str rather than int to keep them free of decimals.
    """
    df = df.copy()
    int_cols = ["count", "levels", "RFQs", "n_obs", "month", "dominant period (trading days)"]
    for col in df.columns.intersection(int_cols):
        df[col] = df[col].astype(int).astype(str)
    return df.to_markdown(index=False, floatfmt=floatfmt)


def write_markdown(rep, figures, path):
    s = rep["scalars"]
    md = [
        "# EDA: raw data profile",
        "",
        "Generated by `scripts/run_eda.py`. Companion file: `eda_report.json`.",
        "",
        "## Files",
        "",
        table(rep["files"]),
        "",
        "## rfqs.csv",
        "",
        (
            f"{s['n_rfqs']:,} quote requests between {s['requested_date'][0]} and "
            f"{s['requested_date'][1]}. {s['n_executed']:,} were executed and "
            f"{s['n_labelled']:,} carry `{TARGET}`."
        ),
        "",
        table(rep["rfq_columns"]),
        "",
        "### Numeric columns",
        "",
        table(rep["rfq_numeric"]),
        "",
        "`tenor_months` (`end_date - start_date`) and `basket_size` are derived here.",
        "",
        "### Categorical columns",
        "",
    ]
    for col, counts in rep["rfq_categorical"].items():
        md += [f"**{col}** ({counts.shape[0]} values shown):", "",
               table(counts), ""]
    md += [
        "Normalised observation frequency:",
        "",
        table(rep["frequency_map"]),
        "",
        f"## Target: {TARGET}",
        "",
        table(rep["target"]),
        "",
        (
            f"{s['target_at_tenor']:,} products ended within half a month of their nominal "
            f"tenor; {s['target_above_tenor']:,} report a duration above it."
        ),
        "",
        "### Correlation with the target",
        "",
        table(rep["correlations"]),
        "",
        "### Variance explained by each categorical",
        "",
        (
            "Eta squared is the share of the target's variance that falls between the levels "
            "of a column rather than within them. `shuffled` is the same statistic after "
            "randomising the labels, averaged over 20 draws: it is what the column would score "
            "on noise alone, and it grows with the number of levels."
        ),
        "",
        table(rep["variance_explained"], floatfmt=",.4f"),
        "",
    ]
    for col, counts in rep["by_category"].items():
        md += [f"Target by `{col}`:", "", table(counts), ""]

    md += ["### Column dependencies", ""]
    if rep["dependencies"].empty:
        md += ["No pair reaches 0.999 purity.", ""]
    else:
        md += [
            (
                "Purity is the weighted share of rows in the modal value of the second "
                "column, within each level of the first."
            ),
            "",
            table(rep["dependencies"]),
            "",
        ]

    md += [
        "## daily_volatility.csv",
        "",
        f"Panel from {s['vol_panel'][0]} to {s['vol_panel'][1]}.",
        "",
        table(rep["per_underlying"]),
        "",
        "### Stationarity",
        "",
        (
            "Augmented Dickey-Fuller test on `realized_vol_63d`, per underlying (lag order "
            "chosen by AIC). The null hypothesis is a unit root; a p-value under 0.05 rejects "
            "it in favour of stationarity."
        ),
        "",
        table(rep["vol_stationarity"], floatfmt=",.4f"),
        "",
        "### Periodicity",
        "",
        (
            "Dominant period per underlying, taken as the strongest non-zero-frequency "
            "component of an FFT of `realized_vol_63d` (demeaned, one sample per trading day). "
            "Meaningful as a cycle only given the stationarity result above: an FFT peak on a "
            "non-stationary (trending) series would not indicate periodicity."
        ),
        "",
        table(rep["vol_periodicity"], floatfmt=",.2f"),
        "",
        "Mean `realized_vol_63d` by calendar month, pooled across all underlyings:",
        "",
        table(rep["vol_seasonality"]),
        "",
        (
            "Note: `realized_vol_63d` is a trailing 63-trading-day rolling statistic, so "
            "autocorrelation at short lags (under ~63 days) is expected from the window overlap "
            "alone and is not evidence of periodicity by itself."
        ),
        "",
        "## underlyings_reference.csv",
        "",
        table(rep["reference"]),
        "",
        (
            "Correlation between `structural_base_vol` and the mean realised volatility: "
            f"{s['structural_vs_realised_corr']}."
        ),
        "",
        "## Volatility measures",
        "",
        (
            "Four volatility figures exist: `quoted_implied_vol` (rfqs.csv, one per RFQ), "
            "`realized_vol_63d` (daily_volatility.csv, joined as-of `requested_date` and "
            "averaged across the basket's legs as `rv_mean`), `structural_base_vol` "
            "(underlyings_reference.csv, static per ticker, averaged the same way as `sb_mean`), "
            "and `mkt_vol` (the cross-sectional mean of `realized_vol_63d` across all "
            "underlyings on a given day, joined as-of `requested_date`). Unlike `rv_mean`, "
            "`mkt_vol` does not depend on which tickers are in the basket."
        ),
        "",
        table(rep["vol_correlation"]),
        "",
        table(rep["vol_r2"]),
        "",
        "Spearman correlation with the target (executed RFQs only):",
        "",
        table(rep["vol_target_correlation"]),
        "",
        "The same, for `mkt_vol`, computed separately within each `product_type`:",
        "",
        table(rep["mkt_vol_by_product"]),
        "",
        "## Joining the tables",
        "",
        (
            f"{s['legs']:,} (RFQ, underlying) pairs; {s['legs_covered']:,} have volatility "
            f"history on or before `requested_date`."
        ),
        "",
        "## Data quality checks",
        "",
        table(rep["checks"]),
        "",
        "## Figures",
        "",
    ]
    for name, caption in figures:
        md += [f"{caption}", "", f"![{name}](figures/{name})", ""]

    path.write_text("\n".join(md), encoding="utf-8")


def to_json(rep):
    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if np.isnan(o) else float(o)
        if isinstance(o, pd.Timestamp):
            return o.date().isoformat()
        return str(o)

    out = {}
    for key, value in rep.items():
        if isinstance(value, pd.DataFrame):
            out[key] = value.to_dict(orient="records")
        elif isinstance(value, dict) and all(isinstance(v, pd.DataFrame) for v in value.values()):
            out[key] = {k: v.to_dict(orient="records") for k, v in value.items()}
        else:
            out[key] = value
    return json.dumps(out, indent=2, default=default)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/eda"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rfqs, vol, ref = load(args.raw_dir)
    rep = build_report(rfqs, vol, ref)
    figures = make_figures(rfqs, vol, ref, args.out_dir / "figures")
    write_markdown(rep, figures, args.out_dir / "EDA.md")
    (args.out_dir / "eda_report.json").write_text(to_json(rep), encoding="utf-8")

    print(f"{args.out_dir}: EDA.md, eda_report.json, {len(figures)} figures")


if __name__ == "__main__":
    main()
