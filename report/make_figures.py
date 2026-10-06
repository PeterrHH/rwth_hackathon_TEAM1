"""Figures for the one-page summary (report/summary.tex). Run from the repository root:

    python report/make_figures.py

Needs the saved submission gbm_step in results/predictions/ (see task0_baseline.ipynb).
"""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecast import Config, evaluate, features, tuning  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#e6e5e0"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984", "lines.linewidth": 1.2, "savefig.bbox": "tight",
})
local_day = lambda ts: ts.dt.tz_convert("Europe/Berlin").dt.date.to_numpy()

# --- 1. Daily total demand over the test year --------------------------------------------
truth = evaluate.load_predictions("gbm_step").rename(columns={"prediction": "gbm"})
day = local_day(truth["Timestamp"])
daily = truth.groupby(day)[["kwh", "gbm"]].sum() / 1000  # MWh
readings = truth.groupby(day).size()
daily = daily[readings >= 0.5 * readings.median()]  # drop the partial last day
daily.index = pd.to_datetime(daily.index)

fig, ax = plt.subplots(figsize=(7.1, 1.55))
ax.plot(daily.index, daily["kwh"], color=INK, linewidth=1.5, label="actual")
ax.plot(daily.index, daily["gbm"], color=BLUE, label="gradient boosting")
ax.set(ylabel="MWh per day", title="Daily demand of all households, test year (Mar 2023 – Feb 2024)",
       ylim=(0, daily.max().max() * 1.2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.legend(ncol=2, loc="upper left")
fig.savefig(OUT / "daily_demand.pdf")
fig.savefig(OUT / "daily_demand.png", dpi=200)  # for RESULTS_SUMMARY.md

# --- 2. Daily error per household by PV status ------------------------------------------
config = Config()
df, _ = features.build_features(config)
test = df[df["is_test"]]
pred = test[["Household_ID", "Timestamp"]].merge(
    truth[["Household_ID", "Timestamp", "gbm"]], on=["Household_ID", "Timestamp"], how="left")["gbm"].to_numpy()
groups = {"PV": test["has_pv"].eq(1).to_numpy(), "no PV": test["has_pv"].eq(0).to_numpy(),
          "unknown": test["has_pv"].isna().to_numpy()}
pv = pd.DataFrame({g: tuning.score_rows(test[m], pred[m]).loc["household_day", ["MAE", "R2"]]
                   for g, m in groups.items()}).T
print("household-day MAE / R² by PV status\n", pv.round(3))

fig, ax = plt.subplots(figsize=(3.4, 1.6))
bars = ax.bar(pv.index, pv["MAE"], 0.6, color=[ORANGE, BLUE, AQUA])
ax.bar_label(bars, labels=[f"{m:.2f} kWh" for m in pv["MAE"]],
             fontsize=6.5, padding=1)
ax.set(ylabel="MAE (kWh per day)", title="Daily error per household by PV status",
       ylim=(0, pv["MAE"].max() * 1.2))
fig.savefig(OUT / "pv_groups.pdf")
fig.savefig(OUT / "pv_groups.png", dpi=200)  # for RESULTS_SUMMARY.md

# --- 3. Buying the median vs the 75 % quantile (validation Jan–Feb 2023) -------------------
FEATURES = features.standard_features()
fit_rows, val_rows = tuning.holdout_split(df, config, start="2023-01-01", sample=1_000_000)
quantiles = {}
for q in (0.5, 0.75):
    model = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300, learning_rate=0.1,
                                          categorical_features="from_dtype", random_state=0)
    quantiles[q] = model.fit(fit_rows[FEATURES], fit_rows["kwh"]).predict(val_rows[FEATURES]).clip(min=0)
buy = pd.DataFrame({"actual": val_rows["kwh"].to_numpy(), "median": quantiles[0.5],
                    "q75": quantiles[0.75]}).groupby(local_day(val_rows["Timestamp"])).sum() / 1000
buy.index = pd.to_datetime(buy.index)
cost = {c: (3 * (buy["actual"] - buy[c]).clip(lower=0) + (buy[c] - buy["actual"]).clip(lower=0)).mean()
        for c in ("median", "q75")}
print("mean daily cost (MWh-equivalents, under = 3x over):", {k: round(v, 2) for k, v in cost.items()},
      "| days short with median:", int((buy["median"] < buy["actual"]).sum()), "of", len(buy))

fig, ax = plt.subplots(figsize=(3.4, 1.6))
ax.plot(buy.index, buy["actual"], color=INK, linewidth=1.5, label="actual")
ax.plot(buy.index, buy["median"], color=ORANGE, label=f"buy median (cost {cost['median']:.1f})")
ax.plot(buy.index, buy["q75"], color=BLUE, label=f"buy 75 % quantile (cost {cost['q75']:.1f})")
ax.set(ylabel="MWh per day", title="Buying the median vs the 75 % quantile (Jan–Feb 2023)")
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=6.5, handlelength=1.5)
fig.savefig(OUT / "quantile_buying.pdf")
fig.savefig(OUT / "quantile_buying.png", dpi=200)  # for RESULTS_SUMMARY.md
print("figures written to", OUT)
