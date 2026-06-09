"""
04_regression.py
----------------
Panel fixed-effects regressions testing H1 and H2.

Research design
---------------
Y: Operational efficiency = asset_turnover = sale / at
X: Intangible asset intensity = intan_intensity = intan / at
Moderator: Firm size = ln_at = log(at)
Interaction: intan_x_size = intan_intensity * ln_at

Input:  data/processed/panel_with_vars.parquet
Output: output/tables/regression_results.csv

Models
------
(1) OLS-style pooled baseline: asset_turnover ~ intan_intensity + controls
(2) TWFE H1 test:            asset_turnover ~ intan_intensity + controls + firm FE + year FE
(3) TWFE H2 test:            asset_turnover ~ intan_intensity + intan_x_size + controls + firm FE + year FE

Estimator
---------
linearmodels PanelOLS with firm-clustered standard errors.
Stars: *** p<0.01, ** p<0.05, * p<0.10
"""

import warnings
from pathlib import Path

import pandas as pd
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")


# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/processed/panel_with_vars.parquet")
TABLE_PATH = Path("output/tables")
TABLE_PATH.mkdir(parents=True, exist_ok=True)


# ── Research design variables ─────────────────────────────────────────────────
DV = "asset_turnover"
X_MAIN = "intan_intensity"
INTERACT = "intan_x_size"
CONTROLS = ["ln_at", "leverage", "capx_intensity", "cash_ratio"]


# ── Load & Set Panel Index ────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)

required = ["gvkey", "fyear", DV, X_MAIN, INTERACT, *CONTROLS]
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"Missing required variables: {missing}")

df = df.dropna(subset=required).copy()
df = df.set_index(["gvkey", "fyear"])

print(f"Panel: {len(df):,} obs | {df.index.get_level_values('gvkey').nunique():,} firms")


# ── Helper: compatible standard error access ──────────────────────────────────
def get_se(res, var: str):
    """Return standard error for a variable across linearmodels/statsmodels-like results."""
    if hasattr(res, "std_errors"):
        return res.std_errors[var]
    if hasattr(res, "bse"):
        return res.bse[var]
    raise AttributeError("Could not find standard errors on result object.")


# ── Helper: regression runner ─────────────────────────────────────────────────
def run_model(dep: str, indep: list[str], entity_fe: bool = False, time_fe: bool = False):
    """
    Estimate a panel model with optional firm and year fixed effects.
    Uses firm-clustered standard errors.
    """
    formula_vars = indep + CONTROLS
    formula = f"{dep} ~ 1 + {' + '.join(formula_vars)}"

    if entity_fe:
        formula += " + EntityEffects"
    if time_fe:
        formula += " + TimeEffects"

    sub = df[[dep, *formula_vars]].dropna()
    mod = PanelOLS.from_formula(formula, data=sub, drop_absorbed=True)

    return mod.fit(cov_type="clustered", cluster_entity=True)


# ── Estimate three models ─────────────────────────────────────────────────────
print("\nEstimating models...")

res1 = run_model(DV, [X_MAIN], entity_fe=False, time_fe=False)
print("  Model 1 (pooled baseline) done")

res2 = run_model(DV, [X_MAIN], entity_fe=True, time_fe=True)
print("  Model 2 (TWFE H1) done")

res3 = run_model(DV, [X_MAIN, INTERACT], entity_fe=True, time_fe=True)
print("  Model 3 (TWFE H2 moderation) done")


# ── Build Results Table ───────────────────────────────────────────────────────
KEY_VARS = [X_MAIN, INTERACT, *CONTROLS]

model_labels = ["(1) OLS", "(2) TWFE", "(3) TWFE+H2"]
models = [res1, res2, res3]

rows = []
for label, res in zip(model_labels, models):
    col = {"Model": label}

    for var in KEY_VARS:
        if var in res.params.index:
            coef = res.params[var]
            se = get_se(res, var)
            pval = res.pvalues[var]
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            col[var] = f"{coef:.3f}{stars}"
            col[f"{var}_se"] = f"({se:.3f})"
        else:
            col[var] = ""
            col[f"{var}_se"] = ""

    col["N"] = f"{int(res.nobs):,}"
    col["R²"] = f"{res.rsquared:.3f}"
    rows.append(col)

results_df = pd.DataFrame(rows).set_index("Model").T

print("\n=== Regression Results ===")
print(results_df.to_string())

results_df.to_csv(TABLE_PATH / "regression_results.csv")
print("\nSaved regression_results.csv")


# ── H1 Diagnostic ─────────────────────────────────────────────────────────────
print("\n--- H1 Diagnostic ---")
b_h1 = res2.params.get(X_MAIN)
p_h1 = res2.pvalues.get(X_MAIN)

if b_h1 is not None and p_h1 is not None:
    stars = "***" if p_h1 < 0.01 else "**" if p_h1 < 0.05 else "*" if p_h1 < 0.1 else "(n.s.)"
    print(f"  β(Intangible intensity) = {b_h1:.3f} {stars}  (p = {p_h1:.3f})")

    if b_h1 > 0 and p_h1 < 0.1:
        print("  → H1 supported: intangible asset intensity is positively associated with operational efficiency.")
    elif b_h1 < 0 and p_h1 < 0.1:
        print("  → H1 not supported: the association is statistically significant but negative.")
    else:
        print("  → H1 not supported at conventional significance levels.")


# ── H2 Diagnostic ─────────────────────────────────────────────────────────────
print("\n--- H2 Diagnostic ---")
b_h2 = res3.params.get(INTERACT)
p_h2 = res3.pvalues.get(INTERACT)

if b_h2 is not None and p_h2 is not None:
    stars = "***" if p_h2 < 0.01 else "**" if p_h2 < 0.05 else "*" if p_h2 < 0.1 else "(n.s.)"
    print(f"  β(Intangible intensity × Firm size) = {b_h2:.3f} {stars}  (p = {p_h2:.3f})")

    if b_h2 > 0 and p_h2 < 0.1:
        print("  → H2 supported: firm size positively moderates the intangible intensity–efficiency relationship.")
    elif b_h2 < 0 and p_h2 < 0.1:
        print("  → H2 not supported: the moderation effect is statistically significant but negative.")
    else:
        print("  → H2 not supported at conventional significance levels.")


# ── OLS vs FE comparison ──────────────────────────────────────────────────────
print("\n--- OLS vs FE Comparison ---")
if X_MAIN in res1.params.index and X_MAIN in res2.params.index:
    b_ols = res1.params[X_MAIN]
    b_fe = res2.params[X_MAIN]

    if b_ols != 0:
        diff_pct = ((b_fe - b_ols) / abs(b_ols)) * 100
        print(f"  β(Intangible intensity), OLS  = {b_ols:.3f}")
        print(f"  β(Intangible intensity), TWFE = {b_fe:.3f}")
        print(f"  Difference: {diff_pct:.1f}%")

        if abs(diff_pct) > 20:
            print(
                "  → The difference is substantial, suggesting that firm and year fixed effects "
                "capture important omitted heterogeneity."
            )
        else:
            print(
                "  → The difference is relatively small, suggesting that the estimated H1 coefficient "
                "is similar in the pooled and two-way fixed effects models."
            )

print("""
─────────────────────────────────────────────────────────────
Interpretation guide:
  Stars: *** p<0.01, ** p<0.05, * p<0.10
  SEs in parentheses, clustered at firm level
  Model (1): pooled baseline
  Models (2)-(3): firm FE + year FE (two-way fixed effects)
─────────────────────────────────────────────────────────────
""")