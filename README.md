# AI-Supported Customer Service in SMEs

ExInt II: Research Designs in SME Research | WU Vienna | SS 2026

Name: Laurenz Hölbl

## Research Question

How does the implementation of AI-supported customer service affect operational efficiency in SMEs, and what implementation challenges emerge during the adoption process?

## Hypotheses

- H1: The implementation of AI-supported customer service is associated with improved operational efficiency in SMEs.
- H2: SMEs face implementation challenges related to employee acceptance, process integration, and data quality when adopting AI-supported customer service.

## Variables

### Dependent variable (Y)

| Construct | Data Item(s) | Formula |
|-----------|--------------|---------|
| RoA | IB, AT | IB / AT |
RoA is operationalized as income before extraordinary items divided by total assets (IB / AT), because IB is highly populated in the cleaned Compustat Global panel.

### Independent variable (X)

| Construct | Data Item(s) | Formula |
|-----------|--------------|---------|
| R&D Intensity | XRD, AT | XRD / AT |
R&D intensity is operationalized as XRD / AT. However, XRD has only 44.5% completeness in the cleaned panel. Therefore, alternative operationalizations such as capital intensity (CAPX / AT), cash holdings (CHE / AT), or tangibility (PPENT / AT) may be considered.

### Controls

| Construct | Data Item(s) | Formula |
|-----------|--------------|---------|
| Firm size | AT | log(AT) |
| Leverage | DLTT, DLC, SEQ | (DLTT + DLC) / SEQ |
| Firm age | FYEAR, INCO | FYEAR - INCO |
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