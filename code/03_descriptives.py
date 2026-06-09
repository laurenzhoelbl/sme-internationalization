"""
03_descriptives.py
------------------
Summary statistics and exploratory figures.

Research design:
Y: Operational efficiency = Asset turnover = sale / at
X: Intangible asset intensity = intan / at
Moderator: Firm size = log(at)
Interaction: Intangible asset intensity x Firm size

Input:  data/processed/panel_clean.parquet
Output: output/tables/summary_statistics.csv
        output/figures/correlation_matrix.png
        output/figures/dv_distribution.png
        output/figures/main_relationship.png
        output/figures/sample_composition.png
        data/processed/panel_with_vars.parquet

Notes on pandas index alignment
--------------------------------
When subsetting a DataFrame and then assigning new columns, always reset
the index to avoid the silent misalignment bug:

    high = df[df["sales"] > 400].copy()
    high.reset_index(drop=True, inplace=True)  # ← always do this
    high["score"] = pd.Series([10, 20])        # now aligns correctly
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path


# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.family": "sans-serif"})
WU_BLUE = "#002f5f"
WU_RED  = "#c8102e"


# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = Path("data/processed/panel_clean.parquet")
PANEL_OUT   = Path("data/processed/panel_with_vars.parquet")
TABLE_PATH  = Path("output/tables")
FIGURE_PATH = Path("output/figures")

TABLE_PATH.mkdir(parents=True, exist_ok=True)
FIGURE_PATH.mkdir(parents=True, exist_ok=True)


# ── Helper ────────────────────────────────────────────────────────────────────
def winsorize(series, lower=0.01, upper=0.99):
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    print(f"  {series.name:<20} [{lo:>8.4f}, {hi:>9.4f}]")
    return series.clip(lo, hi)


# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)
print(f"Loaded {len(df):,} observations | {df['gvkey'].nunique():,} firms")


# ── 0. Data quality filters + variable construction ───────────────────────────
print(f"Starting: {len(df):,} rows")

# Data quality filters from Session 6 checklist
df = df[(df["at"] > 0.1) & (df["sale"] > 0) & (df["seq"] > 0)].copy()
print(f"After quality filters (at>0.1, sale>0, seq>0): {len(df):,}")

# Remove micro-firms with extremely small asset bases
df = df[df["at"] >= 1].copy()
print(f"After micro-firm filter (at>=1): {len(df):,}")

# EU SME filter: emp is reported in thousands, so emp < 0.25 means fewer than 250 employees
sme_mask = (df["emp"] < 0.25) | (df["at"] <= 43)
df = df[sme_mask].copy()
print(f"After SME filter: {len(df):,}")

# ── DEPENDENT VARIABLE ────────────────────────────────────────────────────────
# Operational efficiency = sale / at
df.loc[:, "asset_turnover"] = df["sale"] / df["at"]

# ── INDEPENDENT VARIABLE ──────────────────────────────────────────────────────
# Intangible asset intensity = intan / at
df.loc[:, "intan_intensity"] = df["intan"] / df["at"]

# ── MODERATOR ─────────────────────────────────────────────────────────────────
# Firm size = log(at)
df.loc[:, "ln_at"] = np.log(df["at"])

# H2 interaction: Intangible asset intensity x Firm size
df.loc[:, "intan_x_size"] = df["intan_intensity"] * df["ln_at"]

# ── CONTROLS ──────────────────────────────────────────────────────────────────
df.loc[:, "leverage"]       = df["dltt"].fillna(0) / df["at"]
df.loc[:, "capx_intensity"] = df["capx"].fillna(0) / df["at"]
df.loc[:, "cash_ratio"]     = df["che"].fillna(0)  / df["at"]

# ── COVERAGE ──────────────────────────────────────────────────────────────────
research_vars = [
    "asset_turnover",
    "intan_intensity",
    "ln_at",
    "intan_x_size",
    "leverage",
    "capx_intensity",
    "cash_ratio",
]

print("\nVariable coverage:")
for v in research_vars:
    n   = df[v].notna().sum()
    nz  = (df[v].notna() & (df[v] != 0)).sum()
    pct = n / len(df) * 100
    print(f"  {v:<20}  {n:>7,}  ({pct:>5.1f}%)  non-zero: {nz:>7,}")

# Drop missing core variables
CORE_VARS = ["asset_turnover", "intan_intensity", "ln_at", "leverage"]
n_before = len(df)
df = df.dropna(subset=CORE_VARS).copy()
print(f"\nDropped {n_before - len(df):,} rows with missing core variables")
print(f"Working sample: {len(df):,} firm-years | {df['gvkey'].nunique():,} firms")

# Minimum 3 observations per firm
obs = df.groupby("gvkey")["fyear"].count()
valid = obs[obs >= 3].index
n_before = len(df)
df = df[df["gvkey"].isin(valid)].copy()
print(f"Min 3 obs: {n_before:,} -> {len(df):,}")
print(f"Final: {len(df):,} firm-years | {df['gvkey'].nunique():,} firms")
print(f"Years: {df['fyear'].min()} - {df['fyear'].max()}")
print(
    f"Firms with intangibles (intan_intensity > 0): "
    f"{(df['intan_intensity'] > 0).sum():,} firm-years "
    f"({(df['intan_intensity'] > 0).mean() * 100:.1f}%)"
)

# Winsorize ratio variables at 1%-99%
WINSORIZE_VARS = [
    "asset_turnover",
    "intan_intensity",
    "leverage",
    "capx_intensity",
    "cash_ratio",
]

print("\nWinsorize ranges (1%-99%):")
for v in WINSORIZE_VARS:
    df.loc[:, v] = winsorize(df[v])

# Recompute interaction after winsorizing X
df.loc[:, "intan_x_size"] = df["intan_intensity"] * df["ln_at"]

# Save panel with constructed variables for Session 7
df.to_parquet(PANEL_OUT, index=False)
print(f"\nSaved panel_with_vars.parquet")


# ── 1. Summary Statistics ─────────────────────────────────────────────────────
VAR_LABELS = {
    "asset_turnover":  "Asset turnover (sale/at)",
    "intan_intensity": "Intangible intensity (intan/at)",
    "ln_at":           "Firm size (log assets)",
    "leverage":        "Leverage (dltt/at)",
    "capx_intensity":  "CAPX intensity (capx/at)",
    "cash_ratio":      "Cash ratio (che/at)",
}

summary = (
    df[list(VAR_LABELS.keys())]
    .rename(columns=VAR_LABELS)
    .describe(percentiles=[0.25, 0.5, 0.75])
    .T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    .round(3)
)

print("\n=== Summary Statistics ===")
print(summary.to_string())
summary.to_csv(TABLE_PATH / "summary_statistics.csv")
print("Saved summary_statistics.csv")


# ── 2. Correlation Matrix ─────────────────────────────────────────────────────
corr_vars = list(VAR_LABELS.keys())
corr = df[corr_vars].rename(columns=VAR_LABELS).corr().round(2)

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="RdYlBu_r", center=0, vmin=-1, vmax=1,
    linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
)
ax.set_title("Correlation Matrix — Key Variables", fontsize=13, pad=12, color=WU_BLUE)
fig.tight_layout()
fig.savefig(FIGURE_PATH / "correlation_matrix.png", dpi=150)
plt.close()
print("Saved correlation_matrix.png")


# ── 3. DV Distribution ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].hist(df["asset_turnover"], bins=60, color=WU_BLUE, alpha=0.8)
axes[0].set_title("Asset turnover — Distribution", color=WU_BLUE)
axes[0].set_xlabel("SALE / AT")
axes[0].set_ylabel("Frequency")

axes[1].hist(df["intan_intensity"], bins=60, color=WU_BLUE, alpha=0.8)
axes[1].set_title("Intangible intensity — Distribution", color=WU_BLUE)
axes[1].set_xlabel("INTAN / AT")
axes[1].set_ylabel("Frequency")

fig.tight_layout()
fig.savefig(FIGURE_PATH / "dv_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved dv_distribution.png")


# ── 4. Intangibles–Efficiency Relationship (H1 + H2 preview) ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: raw scatter + bin means
axes[0].scatter(
    df["intan_intensity"],
    df["asset_turnover"],
    alpha=0.04,
    s=5,
    color=WU_BLUE,
)

bins = pd.cut(df["intan_intensity"], bins=20)
bin_means = df.groupby(bins, observed=True)[["intan_intensity", "asset_turnover"]].mean()

axes[0].plot(
    bin_means["intan_intensity"],
    bin_means["asset_turnover"],
    color=WU_RED,
    lw=2.5,
    label="Bin mean",
)

axes[0].set_xlabel("Intangible asset intensity (INTAN / AT)")
axes[0].set_ylabel("Asset turnover (SALE / AT)")
axes[0].set_title("Intangible intensity vs. operational efficiency", color=WU_BLUE)
axes[0].legend()

# Right: by firm size tercile (H2 preview)
# Reset index before assigning the tercile column — avoids pandas alignment bug
df_plot = df.copy()
df_plot.reset_index(drop=True, inplace=True)

df_plot["size_tercile"] = pd.qcut(
    df_plot["ln_at"],
    q=3,
    labels=["Small firms", "Medium firms", "Large firms"],
)

palette = {
    "Small firms": "#2166ac",
    "Medium firms": "#f4a582",
    "Large firms": WU_RED,
}

for label, group in df_plot.groupby("size_tercile", observed=True):
    group_reset = group.reset_index(drop=True)
    bins_g = pd.cut(group_reset["intan_intensity"], bins=15)
    bm = group_reset.groupby(bins_g, observed=True)[["intan_intensity", "asset_turnover"]].mean()

    axes[1].plot(
        bm["intan_intensity"],
        bm["asset_turnover"],
        lw=2,
        label=label,
        color=palette[label],
    )

axes[1].set_xlabel("Intangible asset intensity (INTAN / AT)")
axes[1].set_ylabel("Asset turnover (SALE / AT)")
axes[1].set_title("Relationship by firm size tercile", color=WU_BLUE)
axes[1].legend()

fig.suptitle(
    "Intangible Assets & Operational Efficiency — European SMEs",
    fontsize=13,
    y=1.02,
    color=WU_BLUE,
)

fig.tight_layout()
fig.savefig(FIGURE_PATH / "main_relationship.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved main_relationship.png")


# ── 5. Sample Composition ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

country_counts = df["loc"].value_counts().head(10)
axes[0].barh(country_counts.index[::-1], country_counts.values[::-1], color=WU_BLUE)
axes[0].set_xlabel("Firm-year observations")
axes[0].set_title("Top 10 Countries in Sample", color=WU_BLUE)

year_counts = df["fyear"].value_counts().sort_index()
axes[1].bar(year_counts.index, year_counts.values, color=WU_BLUE)
axes[1].set_xlabel("Fiscal Year")
axes[1].set_ylabel("Observations")
axes[1].set_title("Sample Coverage by Year", color=WU_BLUE)

fig.tight_layout()
fig.savefig(FIGURE_PATH / "sample_composition.png", dpi=150)
plt.close()
print("Saved sample_composition.png")

print("\nDescriptives complete. Check output/tables/, output/figures/, and data/processed/")