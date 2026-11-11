# Sassandra River Basin CMIP6 climate projections

Reproducibility resources for the study of mean climate, climate extremes and meteorological drought in the Sassandra River Basin, Côte d’Ivoire and Guinea.

## Final methodological decisions

- **Climate data:** ten existing bias-adjusted and statistically downscaled NASA NEX-GDDP-CMIP6 models under SSP2-4.5 and SSP5-8.5.
- **Historical comparison baseline:** 1995–2014 for all three future periods: 2021–2040, 2041–2060 and 2081–2100.
- **Additional bias adjustment:** a separate monthly precipitation QDM sensitivity experiment was calibrated over 1981–2000 and evaluated against CHIRPS over 2001–2014. Mean out-of-sample precipitation performance worsened: RMSE 47.621 → 50.167 mm, MAE 35.936 → 36.563 mm, correlation 0.826 → 0.805 and KGE 0.808 → 0.780. **That additional adjustment was not used for the final projections.**
- **Daily temperature QC:** three days with `tasmax_c < tasmin_c` were excluded from DTR computations only; the original observations were retained in the audit.
- **Extreme indices:** TXmean, TNmean, TXx, TNn, DTR, PRCPTOT, Rx1day, Rx5day, CDD, CWD, TX90p, TN90p and WSDI. Temperature percentile thresholds use a model-specific 1995–2014 reference with a five-day calendar window and leave-one-year-out estimation; this is not the formal ETCCDI bootstrap.
- **Drought:** SPI-3 and SPI-12, calibrated to 1995–2014 using month-specific gamma fits; drought threshold SPI ≤ −1.0. No final SPI-6 results are reported.
- **Ensemble periods:** mean of ten *model-paired* future-minus-baseline changes; figure uncertainty bars show ±1 inter-model sample standard deviation, not a confidence interval.
- **Trends:** conventional Mann–Kendall and Sen slopes over 2021–2100 annual **multi-model-median** series, with separate residual autocorrelation and HAC regression sensitivity checks. HAC p-values are not modified Mann–Kendall p-values.

## Repository files

| File | Purpose |
| --- | --- |
| `ENGINE DATA EXTRACTION SCRIPT` | Original Earth Engine extraction workflow; examine the asset paths before running. |
| `Sassandra_CMIP6_Final_Complete_Colab_Script.py` | **Legacy pre-reconciliation analysis. Do not use its QDM/SPI-6 outputs for the revised manuscript.** This script is retained for historical provenance only. |
| `Sassandra_Final_Verified_Results.py` | Current figure and table post-processor that reads the verified Step 9, 12, 13 and 15 CSV outputs. It does not apply extra QDM. |

The Step 9–19 Google Colab reconstruction files and their audited CSV exports are the authoritative source for revised manuscript results. The post-processor does **not** recreate the full climate or SPI reconstruction from raw satellite data.

## Run the verified post-processor

In Google Colab, mount your Google Drive, then run:

```python
from google.colab import drive
drive.mount('/content/drive')
%run Sassandra_Final_Verified_Results.py
```

The script reads the default folder:

```
/content/drive/MyDrive/Sassandra_CMIP6_Final_Verification
```

Or run locally with explicit paths:

```bash
python Sassandra_Final_Verified_Results.py \
  --source /path/to/Sassandra_CMIP6_Final_Verification \
  --output /path/to/verified_figures
```

The script checks input schemas, historical baselines, ten-model completeness and three reporting periods before writing figure and summary files. It stops on missing or ambiguous source exports rather than substituting unsupported data.

## Reproducibility boundaries

The current repository does not yet contain all source data or complete Step 9–19 reconstruction notebooks. The newer script is therefore a reproducible **post-processing step**, conditional on access to the verified output CSVs; it should not be described as a complete raw-data-to-results pipeline. Full reproduction requires archiving the original source-data inventory, Earth Engine extraction settings, Colab reconstruction code, dependency versions and the final quality-control reports.

## Citation

Please cite the peer-reviewed article once its final bibliographic details are available. No DOI is assigned in this README.
