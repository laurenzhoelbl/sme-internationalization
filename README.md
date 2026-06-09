# Intangible Assets and Operational Efficiency in European SMEs

ExInt II: Research Designs in SME Research | WU Vienna | SS 2026

Name: Laurenz Hölbl

## Research Question

How does intangible asset intensity affect operational efficiency among European SMEs, and does firm size moderate this relationship?

## Hypotheses

- H1: Intangible asset intensity is associated with improved operational efficiency among European SMEs.
- H2: Firm size positively moderates the relationship between intangible asset intensity and operational efficiency, because larger SMEs may be better able to deploy intangible resources effectively.

## Variables

### Dependent variable (Y)

| Construct | Data Item(s) | Formula |
|-----------|--------------|---------|
| Asset turnover | SALE, AT | SALE / AT |

Asset turnover is operationalized as sales divided by total assets (SALE / AT). It captures how efficiently a firm uses its asset base to generate revenue and is therefore used as a proxy for operational efficiency.

### Independent variable (X)

| Construct | Data Item(s) | Formula |
|-----------|--------------|---------|
| Intangible asset intensity | INTAN, AT | INTAN / AT |

Intangible asset intensity is operationalized as total intangible assets divided by total assets (INTAN / AT). It is used as a financial-statement proxy for knowledge-based and digital capabilities. Direct firm-level data on AI-supported customer service adoption is not available in Compustat Global, so the empirical design focuses on intangible assets as a broader observable proxy.

### Controls

| Construct | Data Item(s) | Formula |
|-----------|--------------|---------|
| Firm size | AT | log(AT) |
| Leverage | DLTT, AT | DLTT / AT |
| CAPX intensity | CAPX, AT | CAPX / AT |
| Cash ratio | CHE, AT | CHE / AT |
| Industry | SIC or NAICS | categorical fixed effect / dummy |

## Data

| Item | Detail |
|------|--------|
| Source | WRDS / Compustat Global |
| Table | comp_global_daily.g_funda |
| Downloaded | 2026-06-03 |
| License | WRDS subscriber agreement |
| Fiscal years | 2015–2024 |
| Raw rows | 338,475 |
| Clean rows | 26,091 |
| Clean columns | 444 |