# =============================================================================
# CREATE PROFESSIONAL README.md
# =============================================================================

readme_path = REPOSITORY_DIR / "README.md"

readme_text = f"""# Bias-Corrected CMIP6 Climate Projections for the Sassandra River Basin

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-JavaScript-green.svg)](https://earthengine.google.com/)
[![CMIP6](https://img.shields.io/badge/Climate%20Models-CMIP6-orange.svg)](https://wcrp-cmip.org/cmip6/)

## Overview

This repository contains the reproducible Google Earth Engine and Google Colab workflows used to evaluate historical climate simulations and assess future changes in mean climate, climate extremes, and drought across the Sassandra River Basin in Côte d’Ivoire and Guinea.

The analysis integrates CHIRPS precipitation, ERA5-Land temperature, and ten CMIP6 climate models. Monthly Quantile Delta Mapping is applied to reduce systematic model biases before assessing future changes under SSP2-4.5 and SSP5-8.5.

## Study Objectives

The workflow was developed to:

1. evaluate the historical performance of selected CMIP6 models;
2. apply bias correction to precipitation and temperature simulations;
3. assess projected changes in annual precipitation and temperature;
4. quantify changes in rainfall and temperature extremes;
5. evaluate future drought frequency, duration, and severity;
6. characterize inter-model spread and agreement.

## Study Area

The study focuses on the Sassandra River Basin, a major transboundary river system extending from Guinea through western Côte d’Ivoire to the Gulf of Guinea.

Google Earth Engine asset:

```text
projects/ee-george946/assets/SASAN
