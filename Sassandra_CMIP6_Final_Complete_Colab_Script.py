# =============================================================================
# SASSANDRA RIVER BASIN MULTI-DECADAL CLIMATE PROJECTION ANALYSIS
# FINAL ROBUST GOOGLE COLAB SCRIPT
#
# This version:
# 1. Searches the entire Google Drive for all Earth Engine exports.
# 2. Works even when duplicate folders have the same visible name.
# 3. Loads the combined historical CMIP6 CSV.
# 4. Loads all separate SSP2-4.5 and SSP5-8.5 CSV files.
# 5. Evaluates CMIP6 models.
# 6. Applies monthly Quantile Delta Mapping.
# 7. Calculates mean-climate changes.
# 8. Calculates precipitation and temperature extremes.
# 9. Calculates SPI-3, SPI-6 and SPI-12 drought characteristics.
# 10. Produces main manuscript and supplementary outputs.
# =============================================================================


# =============================================================================
# 1. INSTALL REQUIRED PACKAGES
# =============================================================================

!pip -q install openpyxl rasterio scikit-learn scipy tqdm


# =============================================================================
# 2. IMPORT LIBRARIES
# =============================================================================

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio

from scipy import stats
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)


# =============================================================================
# 3. MOUNT GOOGLE DRIVE
# =============================================================================

from google.colab import drive

drive.mount(
    "/content/drive",
    force_remount=True
)


# =============================================================================
# 4. SEARCH THE WHOLE GOOGLE DRIVE
# =============================================================================

DRIVE_ROOT = Path("/content/drive/MyDrive")

if not DRIVE_ROOT.exists():
    raise FileNotFoundError(
        "Google Drive was not mounted correctly."
    )


def recursive_find(patterns, required=True):
    """
    Search the whole My Drive for files matching one or more patterns.
    """

    if isinstance(patterns, str):
        patterns = [patterns]

    matches = []

    for pattern in patterns:
        matches.extend(
            DRIVE_ROOT.rglob(pattern)
        )

    unique_matches = []

    for file in sorted(matches):
        if file.is_file() and file not in unique_matches:
            unique_matches.append(file)

    if not unique_matches and required:
        raise FileNotFoundError(
            "No file was found for these patterns:\n"
            + "\n".join(patterns)
        )

    return unique_matches


print("=" * 88)
print("SEARCHING THE ENTIRE GOOGLE DRIVE")
print("=" * 88)

chirps_matches = recursive_find([
    "SASAN_CHIRPS_Daily_1981_2014*.csv",
    "*CHIRPS*Daily*1981*2014*.csv"
])

era5_matches = recursive_find([
    "SASAN_ERA5Land_Daily_Temperature_1981_2014*.csv",
    "*ERA5Land*Daily*Temperature*1981*2014*.csv"
])

historical_matches = recursive_find([
    "SASAN_CMIP6_Historical_Daily_1981_2014*.csv",
    "*CMIP6*Historical*Daily*1981*2014*.csv"
])

future_matches = recursive_find([
    "SASAN_CMIP6_*_ssp245_Daily_2015_2100*.csv",
    "SASAN_CMIP6_*_ssp585_Daily_2015_2100*.csv",
    "*CMIP6*ssp245*Daily*2015*2100*.csv",
    "*CMIP6*ssp585*Daily*2015*2100*.csv"
])

rainfall_tif_matches = recursive_find(
    [
        "SASAN_Historical_Annual_Precipitation_1995_2014*.tif",
        "*Historical*Annual*Precipitation*1995*2014*.tif"
    ],
    required=False
)

temperature_tif_matches = recursive_find(
    [
        "SASAN_Historical_Temperature_1995_2014*.tif",
        "*Historical*Temperature*1995*2014*.tif"
    ],
    required=False
)

chirps_file = chirps_matches[0]
era5_file = era5_matches[0]
historical_file = historical_matches[0]

rainfall_tif = (
    rainfall_tif_matches[0]
    if rainfall_tif_matches
    else None
)

temperature_tif = (
    temperature_tif_matches[0]
    if temperature_tif_matches
    else None
)

print("CHIRPS:")
print(chirps_file)

print("\nERA5-Land:")
print(era5_file)

print("\nHistorical CMIP6:")
print(historical_file)

print("\nFuture CMIP6 files:", len(future_matches))

for file in future_matches:
    print("✓", file)

if rainfall_tif:
    print("\nHistorical precipitation GeoTIFF:")
    print(rainfall_tif)

if temperature_tif:
    print("\nHistorical temperature GeoTIFF:")
    print(temperature_tif)


# =============================================================================
# 5. CREATE A NEW, UNIQUE OUTPUT FOLDER
# =============================================================================

OUTPUT_ROOT = (
    DRIVE_ROOT
    / "SASSANDRA_CMIP6_FINAL_ANALYSIS"
)

RESULTS_DIR = OUTPUT_ROOT / "FINAL_RESULTS"
FIGURES_DIR = RESULTS_DIR / "MAIN_FIGURES"
TABLES_DIR = RESULTS_DIR / "MAIN_TABLES"
SUPPLEMENT_DIR = RESULTS_DIR / "SUPPLEMENTARY_MATERIALS"
PROCESSED_DIR = RESULTS_DIR / "PROCESSED_DATA"
DIAGNOSTICS_DIR = RESULTS_DIR / "DIAGNOSTICS"

for folder in [
    OUTPUT_ROOT,
    RESULTS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    SUPPLEMENT_DIR,
    PROCESSED_DIR,
    DIAGNOSTICS_DIR
]:
    folder.mkdir(
        parents=True,
        exist_ok=True
    )

print("\nOutput folder:")
print(RESULTS_DIR)


# =============================================================================
# 6. ANALYSIS SETTINGS
# =============================================================================

HISTORICAL_START = 1981
HISTORICAL_END = 2014

BASELINE_START = 1995
BASELINE_END = 2014

FUTURE_PERIODS = {
    "Near future (2021–2040)": (2021, 2040),
    "Mid-century (2041–2060)": (2041, 2060),
    "Late century (2081–2100)": (2081, 2100)
}

PERIOD_ORDER = list(
    FUTURE_PERIODS.keys()
)

SCENARIO_LABELS = {
    "ssp245": "SSP2-4.5",
    "ssp585": "SSP5-8.5"
}

SCENARIO_ORDER = [
    "ssp245",
    "ssp585"
]

VARIABLES = [
    "precipitation_mm_day",
    "tasmax_c",
    "tasmin_c"
]

WET_DAY_THRESHOLD = 1.0
HEAVY_RAIN_THRESHOLD = 20.0

SPI_SCALES = [
    3,
    6,
    12
]

DPI = 900


# =============================================================================
# 7. GENERAL FUNCTIONS
# =============================================================================

def add_date_fields(dataframe):
    data = dataframe.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["date"]
    )

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["doy"] = data["date"].dt.dayofyear

    return data


def convert_numeric(dataframe, columns):
    data = dataframe.copy()

    for column in columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    return data


def assign_period(year):
    for period, (
        start_year,
        end_year
    ) in FUTURE_PERIODS.items():

        if start_year <= int(year) <= end_year:
            return period

    return None


def longest_run(condition):
    condition = np.asarray(
        condition,
        dtype=bool
    )

    longest = 0
    current = 0

    for value in condition:
        if value:
            current += 1
            longest = max(
                longest,
                current
            )
        else:
            current = 0

    return int(longest)


def total_days_in_runs(
    condition,
    minimum_length=6
):
    condition = np.asarray(
        condition,
        dtype=bool
    )

    total = 0
    current = 0

    for value in condition:
        if value:
            current += 1
        else:
            if current >= minimum_length:
                total += current

            current = 0

    if current >= minimum_length:
        total += current

    return int(total)


def save_figure(
    figure,
    filename
):
    path = FIGURES_DIR / filename

    figure.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(figure)

    print("Saved:", path)


# =============================================================================
# 8. LOAD CHIRPS
# =============================================================================

chirps_raw = pd.read_csv(
    chirps_file
)

chirps_raw = add_date_fields(
    chirps_raw
)

possible_precipitation_columns = [
    "precipitation_mm_day",
    "precipitation",
    "rainfall_mm",
    "rainfall",
    "pr"
]

chirps_precipitation_column = None

for column in possible_precipitation_columns:
    if column in chirps_raw.columns:
        chirps_precipitation_column = column
        break

if chirps_precipitation_column is None:
    raise KeyError(
        "No precipitation column was found in the CHIRPS CSV.\n"
        f"Available columns: {list(chirps_raw.columns)}"
    )

chirps_raw = convert_numeric(
    chirps_raw,
    [chirps_precipitation_column]
)

chirps = chirps_raw[
    [
        "date",
        chirps_precipitation_column
    ]
].rename(
    columns={
        chirps_precipitation_column:
            "precipitation_mm_day"
    }
)

chirps.loc[
    chirps["precipitation_mm_day"] < 0,
    "precipitation_mm_day"
] = np.nan

chirps = chirps.dropna()

print("\nCHIRPS records:", len(chirps))


# =============================================================================
# 9. LOAD ERA5-LAND
# =============================================================================

era5 = pd.read_csv(
    era5_file
)

era5 = add_date_fields(
    era5
)

era5 = convert_numeric(
    era5,
    [
        "tasmax_c",
        "tasmin_c"
    ]
)

required_era5_columns = [
    "tasmax_c",
    "tasmin_c"
]

for column in required_era5_columns:
    if column not in era5.columns:
        raise KeyError(
            f"ERA5-Land column '{column}' was not found.\n"
            f"Available columns: {list(era5.columns)}"
        )

era5 = era5[
    [
        "date",
        "tasmax_c",
        "tasmin_c"
    ]
].dropna()

era5 = era5[
    era5["tasmin_c"]
    <=
    era5["tasmax_c"]
].copy()

print("ERA5-Land records:", len(era5))


# =============================================================================
# 10. COMBINE OBSERVATIONAL DATA
# =============================================================================

observed = pd.merge(
    chirps,
    era5,
    on="date",
    how="inner"
)

observed = add_date_fields(
    observed
)

observed = observed[
    observed["year"].between(
        HISTORICAL_START,
        HISTORICAL_END
    )
].copy()

observed = observed.dropna(
    subset=VARIABLES
)

observed = observed.sort_values(
    "date"
).drop_duplicates(
    subset=["date"]
)

print("Combined observation records:", len(observed))
print(
    "Observation period:",
    observed["date"].min(),
    "to",
    observed["date"].max()
)


# =============================================================================
# 11. LOAD HISTORICAL CMIP6
# =============================================================================

historical = pd.read_csv(
    historical_file
)

historical = add_date_fields(
    historical
)

historical = convert_numeric(
    historical,
    VARIABLES
)

required_historical_columns = [
    "date",
    "model",
    *VARIABLES
]

for column in required_historical_columns:
    if column not in historical.columns:
        raise KeyError(
            f"Historical CMIP6 column '{column}' is missing.\n"
            f"Available columns: {list(historical.columns)}"
        )

historical["model"] = (
    historical["model"]
    .astype(str)
    .str.strip()
)

historical["scenario"] = "historical"

historical = historical[
    historical["year"].between(
        HISTORICAL_START,
        HISTORICAL_END
    )
].copy()

historical.loc[
    historical["precipitation_mm_day"] < 0,
    "precipitation_mm_day"
] = 0

historical = historical.dropna(
    subset=[
        "model",
        *VARIABLES
    ]
)

historical = historical.drop_duplicates(
    subset=[
        "date",
        "model"
    ]
)

print("Historical CMIP6 records:", len(historical))
print(
    "Historical models:",
    sorted(
        historical["model"].unique()
    )
)


# =============================================================================
# 12. LOAD ALL FUTURE CMIP6 FILES
# =============================================================================

future_frames = []

for file in tqdm(
    future_matches,
    desc="Loading future CMIP6 files"
):

    try:
        frame = pd.read_csv(
            file
        )

        frame = add_date_fields(
            frame
        )

        frame = convert_numeric(
            frame,
            VARIABLES
        )

        required_columns = [
            "date",
            "model",
            "scenario",
            *VARIABLES
        ]

        missing = [
            column
            for column in required_columns
            if column not in frame.columns
        ]

        if missing:
            print(
                "Skipped:",
                file.name,
                "Missing:",
                missing
            )
            continue

        frame["model"] = (
            frame["model"]
            .astype(str)
            .str.strip()
        )

        frame["scenario"] = (
            frame["scenario"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        frame = frame[
            frame["scenario"].isin(
                SCENARIO_ORDER
            )
        ]

        frame = frame[
            frame["year"].between(
                2015,
                2100
            )
        ]

        frame.loc[
            frame["precipitation_mm_day"] < 0,
            "precipitation_mm_day"
        ] = 0

        frame = frame.dropna(
            subset=[
                "model",
                "scenario",
                *VARIABLES
            ]
        )

        future_frames.append(
            frame
        )

    except Exception as error:
        print(
            "Could not load:",
            file,
            error
        )

if not future_frames:
    raise RuntimeError(
        "No usable future CMIP6 data were loaded."
    )

future = pd.concat(
    future_frames,
    ignore_index=True
)

future = future.drop_duplicates(
    subset=[
        "date",
        "model",
        "scenario"
    ]
)

print("\nFuture CMIP6 records:", len(future))
print(
    "Future models:",
    sorted(
        future["model"].unique()
    )
)
print(
    "Scenarios:",
    sorted(
        future["scenario"].unique()
    )
)


# =============================================================================
# 13. IDENTIFY MODELS AVAILABLE IN ALL PERIODS
# =============================================================================

historical_models = set(
    historical["model"].unique()
)

ssp245_models = set(
    future.loc[
        future["scenario"] == "ssp245",
        "model"
    ].unique()
)

ssp585_models = set(
    future.loc[
        future["scenario"] == "ssp585",
        "model"
    ].unique()
)

valid_models = sorted(
    historical_models
    .intersection(ssp245_models)
    .intersection(ssp585_models)
)

if len(valid_models) < 3:
    raise RuntimeError(
        "Fewer than three models have historical, SSP2-4.5 "
        "and SSP5-8.5 records."
    )

historical = historical[
    historical["model"].isin(
        valid_models
    )
].copy()

future = future[
    future["model"].isin(
        valid_models
    )
].copy()

print("\nModels retained:", len(valid_models))

for model in valid_models:
    print("✓", model)


# =============================================================================
# 14. PERFORMANCE METRICS
# =============================================================================

def calculate_kge(
    observed_values,
    simulated_values
):
    obs = np.asarray(
        observed_values,
        dtype=float
    )

    sim = np.asarray(
        simulated_values,
        dtype=float
    )

    valid = (
        np.isfinite(obs)
        &
        np.isfinite(sim)
    )

    obs = obs[valid]
    sim = sim[valid]

    if len(obs) < 3:
        return np.nan

    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim)

    obs_std = np.std(
        obs,
        ddof=1
    )

    sim_std = np.std(
        sim,
        ddof=1
    )

    if obs_mean == 0 or obs_std == 0:
        return np.nan

    correlation = np.corrcoef(
        obs,
        sim
    )[0, 1]

    alpha = sim_std / obs_std
    beta = sim_mean / obs_mean

    return 1 - np.sqrt(
        (correlation - 1) ** 2
        +
        (alpha - 1) ** 2
        +
        (beta - 1) ** 2
    )


def calculate_metrics(
    observed_values,
    simulated_values
):
    obs = np.asarray(
        observed_values,
        dtype=float
    )

    sim = np.asarray(
        simulated_values,
        dtype=float
    )

    valid = (
        np.isfinite(obs)
        &
        np.isfinite(sim)
    )

    obs = obs[valid]
    sim = sim[valid]

    if len(obs) < 3:
        return {
            "n": len(obs),
            "bias": np.nan,
            "relative_bias_percent": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "correlation": np.nan,
            "kge": np.nan
        }

    bias = np.mean(
        sim - obs
    )

    relative_bias = (
        bias / np.mean(obs) * 100
        if np.mean(obs) != 0
        else np.nan
    )

    mae = np.mean(
        np.abs(
            sim - obs
        )
    )

    rmse = np.sqrt(
        mean_squared_error(
            obs,
            sim
        )
    )

    correlation = np.corrcoef(
        obs,
        sim
    )[0, 1]

    return {
        "n": len(obs),
        "bias": bias,
        "relative_bias_percent":
            relative_bias,
        "mae": mae,
        "rmse": rmse,
        "correlation": correlation,
        "kge": calculate_kge(
            obs,
            sim
        )
    }


def monthly_series(
    dataframe,
    variable
):
    series = (
        dataframe
        .set_index("date")[variable]
        .sort_index()
    )

    if variable == "precipitation_mm_day":
        return series.resample(
            "MS"
        ).sum(
            min_count=1
        )

    return series.resample(
        "MS"
    ).mean()


# =============================================================================
# 15. QUANTILE DELTA MAPPING
# =============================================================================

def empirical_probabilities(values):
    values = np.asarray(
        values,
        dtype=float
    )

    order = np.argsort(
        values
    )

    probabilities = np.empty(
        len(values),
        dtype=float
    )

    probabilities[order] = (
        np.arange(
            1,
            len(values) + 1
        )
        -
        0.5
    ) / len(values)

    return probabilities


def empirical_quantiles(
    sample,
    probabilities
):
    sample = np.asarray(
        sample,
        dtype=float
    )

    sample = sample[
        np.isfinite(sample)
    ]

    if len(sample) == 0:
        return np.full(
            len(probabilities),
            np.nan
        )

    return np.quantile(
        sample,
        np.clip(
            probabilities,
            0.0001,
            0.9999
        ),
        method="linear"
    )


def qdm(
    observations,
    model_reference,
    target,
    mode
):
    observations = np.asarray(
        observations,
        dtype=float
    )

    model_reference = np.asarray(
        model_reference,
        dtype=float
    )

    target = np.asarray(
        target,
        dtype=float
    )

    output = np.full(
        len(target),
        np.nan
    )

    valid = np.isfinite(
        target
    )

    target_valid = target[
        valid
    ]

    if len(target_valid) == 0:
        return output

    probabilities = empirical_probabilities(
        target_valid
    )

    observation_quantiles = empirical_quantiles(
        observations,
        probabilities
    )

    model_quantiles = empirical_quantiles(
        model_reference,
        probabilities
    )

    if mode == "multiplicative":

        denominator = np.where(
            np.abs(model_quantiles) < 0.01,
            0.01,
            model_quantiles
        )

        corrected = (
            observation_quantiles
            *
            target_valid
            /
            denominator
        )

        corrected = np.maximum(
            corrected,
            0
        )

    else:

        corrected = (
            observation_quantiles
            +
            target_valid
            -
            model_quantiles
        )

    output[valid] = corrected

    return output


def monthly_qdm(
    observations,
    historical_model,
    target,
    variable
):
    corrected = np.full(
        len(target),
        np.nan
    )

    mode = (
        "multiplicative"
        if variable == "precipitation_mm_day"
        else "additive"
    )

    for month in range(
        1,
        13
    ):

        observed_values = observations.loc[
            observations["month"] == month,
            variable
        ].dropna().to_numpy()

        model_values = historical_model.loc[
            historical_model["month"] == month,
            variable
        ].dropna().to_numpy()

        target_mask = (
            target["month"] == month
        )

        target_values = target.loc[
            target_mask,
            variable
        ].to_numpy()

        if (
            len(observed_values) < 20
            or len(model_values) < 20
            or len(target_values) == 0
        ):
            continue

        corrected_values = qdm(
            observed_values,
            model_values,
            target_values,
            mode
        )

        corrected[
            np.where(
                target_mask.to_numpy()
            )[0]
        ] = corrected_values

    if variable == "precipitation_mm_day":
        corrected[
            corrected < WET_DAY_THRESHOLD
        ] = 0

    return corrected


# =============================================================================
# 16. APPLY BIAS CORRECTION
# =============================================================================

historical_corrected_frames = []
future_corrected_frames = []

for model in tqdm(
    valid_models,
    desc="Bias-correcting models"
):

    historical_model = (
        historical[
            historical["model"] == model
        ]
        .sort_values("date")
        .reset_index(drop=True)
        .copy()
    )

    future_model = (
        future[
            future["model"] == model
        ]
        .sort_values(
            [
                "scenario",
                "date"
            ]
        )
        .reset_index(drop=True)
        .copy()
    )

    for variable in VARIABLES:

        historical_model[
            f"{variable}_corrected"
        ] = monthly_qdm(
            observed,
            historical_model,
            historical_model,
            variable
        )

        future_model[
            f"{variable}_corrected"
        ] = np.nan

        for scenario in SCENARIO_ORDER:

            mask = (
                future_model["scenario"]
                ==
                scenario
            )

            scenario_data = (
                future_model.loc[
                    mask
                ]
                .reset_index(drop=True)
                .copy()
            )

            corrected_values = monthly_qdm(
                observed,
                historical_model,
                scenario_data,
                variable
            )

            future_model.loc[
                mask,
                f"{variable}_corrected"
            ] = corrected_values

    historical_corrected_frames.append(
        historical_model
    )

    future_corrected_frames.append(
        future_model
    )

historical_corrected = pd.concat(
    historical_corrected_frames,
    ignore_index=True
)

future_corrected = pd.concat(
    future_corrected_frames,
    ignore_index=True
)

corrected_columns = [
    f"{variable}_corrected"
    for variable in VARIABLES
]

historical_corrected = historical_corrected.dropna(
    subset=corrected_columns
)

future_corrected = future_corrected.dropna(
    subset=corrected_columns
)

print(
    "\nBias-corrected historical rows:",
    len(historical_corrected)
)

print(
    "Bias-corrected future rows:",
    len(future_corrected)
)


# =============================================================================
# 17. MODEL EVALUATION
# =============================================================================

performance_rows = []

for model in tqdm(
    valid_models,
    desc="Evaluating models"
):

    raw_model = historical[
        historical["model"] == model
    ]

    corrected_model = historical_corrected[
        historical_corrected["model"] == model
    ]

    for variable in VARIABLES:

        observed_monthly = monthly_series(
            observed,
            variable
        )

        raw_monthly = monthly_series(
            raw_model,
            variable
        )

        # Select only the corrected variable before renaming it.
        # The earlier version renamed the corrected column while retaining the
        # original raw column with the same name. That created duplicate column
        # labels, causing monthly_series() to return a DataFrame rather than a
        # Series and producing: TypeError: 'str' object is not callable.
        corrected_input = corrected_model[
            [
                "date",
                f"{variable}_corrected"
            ]
        ].rename(
            columns={
                f"{variable}_corrected":
                    variable
            }
        )

        corrected_monthly = monthly_series(
            corrected_input,
            variable
        )

        raw_comparison = pd.concat(
            [
                observed_monthly.rename(
                    "observed"
                ),
                raw_monthly.rename(
                    "simulated"
                )
            ],
            axis=1
        ).dropna()

        corrected_comparison = pd.concat(
            [
                observed_monthly.rename(
                    "observed"
                ),
                corrected_monthly.rename(
                    "simulated"
                )
            ],
            axis=1
        ).dropna()

        performance_rows.append({
            "model": model,
            "variable": variable,
            "stage": "Raw",
            **calculate_metrics(
                raw_comparison["observed"],
                raw_comparison["simulated"]
            )
        })

        performance_rows.append({
            "model": model,
            "variable": variable,
            "stage": "Bias corrected",
            **calculate_metrics(
                corrected_comparison["observed"],
                corrected_comparison["simulated"]
            )
        })

performance = pd.DataFrame(
    performance_rows
)


# =============================================================================
# 18. ANNUAL CLIMATE-EXTREME INDICES
# =============================================================================

baseline_observations = observed[
    observed["year"].between(
        BASELINE_START,
        BASELINE_END
    )
]

tmax90 = baseline_observations[
    "tasmax_c"
].quantile(
    0.90
)

tmin90 = baseline_observations[
    "tasmin_c"
].quantile(
    0.90
)


def annual_indices(
    dataframe,
    precipitation_column,
    tmax_column,
    tmin_column
):
    rows = []

    for year, group in dataframe.groupby(
        "year"
    ):

        group = group.sort_values(
            "date"
        )

        precipitation = group[
            precipitation_column
        ].to_numpy(
            dtype=float
        )

        tmax = group[
            tmax_column
        ].to_numpy(
            dtype=float
        )

        tmin = group[
            tmin_column
        ].to_numpy(
            dtype=float
        )

        if np.isfinite(
            precipitation
        ).sum() < 300:
            continue

        five_day = (
            pd.Series(
                precipitation
            )
            .rolling(
                5,
                min_periods=5
            )
            .sum()
        )

        warm_days = (
            tmax > tmax90
        )

        warm_nights = (
            tmin > tmin90
        )

        rows.append({
            "year": int(year),

            "PRCPTOT_mm":
                np.nansum(
                    precipitation[
                        precipitation
                        >=
                        WET_DAY_THRESHOLD
                    ]
                ),

            "Rx1day_mm":
                np.nanmax(
                    precipitation
                ),

            "Rx5day_mm":
                np.nanmax(
                    five_day
                ),

            "R20_days":
                np.nansum(
                    precipitation
                    >=
                    HEAVY_RAIN_THRESHOLD
                ),

            "CDD_days":
                longest_run(
                    precipitation
                    <
                    WET_DAY_THRESHOLD
                ),

            "CWD_days":
                longest_run(
                    precipitation
                    >=
                    WET_DAY_THRESHOLD
                ),

            "TXx_c":
                np.nanmax(
                    tmax
                ),

            "TNn_c":
                np.nanmin(
                    tmin
                ),

            "TX90p_percent":
                np.nanmean(
                    warm_days
                )
                *
                100,

            "TN90p_percent":
                np.nanmean(
                    warm_nights
                )
                *
                100,

            "WSDI_days":
                total_days_in_runs(
                    warm_days,
                    minimum_length=6
                )
        })

    return pd.DataFrame(
        rows
    )


historical_indices_frames = []
future_indices_frames = []

for model in tqdm(
    valid_models,
    desc="Calculating climate extremes"
):

    historical_model = historical_corrected[
        historical_corrected["model"]
        ==
        model
    ]

    historical_model_indices = annual_indices(
        historical_model,
        "precipitation_mm_day_corrected",
        "tasmax_c_corrected",
        "tasmin_c_corrected"
    )

    historical_model_indices[
        "model"
    ] = model

    historical_model_indices[
        "scenario"
    ] = "historical"

    historical_indices_frames.append(
        historical_model_indices
    )

    for scenario in SCENARIO_ORDER:

        future_model = future_corrected[
            (
                future_corrected["model"]
                ==
                model
            )
            &
            (
                future_corrected["scenario"]
                ==
                scenario
            )
        ]

        future_model_indices = annual_indices(
            future_model,
            "precipitation_mm_day_corrected",
            "tasmax_c_corrected",
            "tasmin_c_corrected"
        )

        future_model_indices[
            "model"
        ] = model

        future_model_indices[
            "scenario"
        ] = scenario

        future_indices_frames.append(
            future_model_indices
        )

historical_indices = pd.concat(
    historical_indices_frames,
    ignore_index=True
)

future_indices = pd.concat(
    future_indices_frames,
    ignore_index=True
)

future_indices["period"] = (
    future_indices["year"]
    .apply(
        assign_period
    )
)

future_indices = future_indices.dropna(
    subset=["period"]
)


# =============================================================================
# 19. PROJECTED MEAN CLIMATE
# =============================================================================

historical_annual = (
    historical_corrected
    .groupby(
        [
            "model",
            "year"
        ],
        as_index=False
    )
    .agg(
        annual_precipitation_mm=(
            "precipitation_mm_day_corrected",
            "sum"
        ),
        tasmax_c=(
            "tasmax_c_corrected",
            "mean"
        ),
        tasmin_c=(
            "tasmin_c_corrected",
            "mean"
        )
    )
)

historical_annual = historical_annual[
    historical_annual["year"].between(
        BASELINE_START,
        BASELINE_END
    )
]

future_annual = (
    future_corrected
    .groupby(
        [
            "model",
            "scenario",
            "year"
        ],
        as_index=False
    )
    .agg(
        annual_precipitation_mm=(
            "precipitation_mm_day_corrected",
            "sum"
        ),
        tasmax_c=(
            "tasmax_c_corrected",
            "mean"
        ),
        tasmin_c=(
            "tasmin_c_corrected",
            "mean"
        )
    )
)

future_annual["period"] = (
    future_annual["year"]
    .apply(
        assign_period
    )
)

future_annual = future_annual.dropna(
    subset=["period"]
)

baseline_means = (
    historical_annual
    .groupby(
        "model",
        as_index=False
    )
    .agg(
        baseline_precipitation_mm=(
            "annual_precipitation_mm",
            "mean"
        ),
        baseline_tasmax_c=(
            "tasmax_c",
            "mean"
        ),
        baseline_tasmin_c=(
            "tasmin_c",
            "mean"
        )
    )
)

future_means = (
    future_annual
    .groupby(
        [
            "model",
            "scenario",
            "period"
        ],
        as_index=False
    )
    .agg(
        future_precipitation_mm=(
            "annual_precipitation_mm",
            "mean"
        ),
        future_tasmax_c=(
            "tasmax_c",
            "mean"
        ),
        future_tasmin_c=(
            "tasmin_c",
            "mean"
        )
    )
)

mean_change = pd.merge(
    future_means,
    baseline_means,
    on="model",
    how="left"
)

mean_change[
    "precipitation_change_percent"
] = (
    (
        mean_change[
            "future_precipitation_mm"
        ]
        -
        mean_change[
            "baseline_precipitation_mm"
        ]
    )
    /
    mean_change[
        "baseline_precipitation_mm"
    ]
    *
    100
)

mean_change[
    "tasmax_change_c"
] = (
    mean_change[
        "future_tasmax_c"
    ]
    -
    mean_change[
        "baseline_tasmax_c"
    ]
)

mean_change[
    "tasmin_change_c"
] = (
    mean_change[
        "future_tasmin_c"
    ]
    -
    mean_change[
        "baseline_tasmin_c"
    ]
)


# =============================================================================
# 20. PROJECTED EXTREME CHANGES
# =============================================================================

INDEX_COLUMNS = [
    "PRCPTOT_mm",
    "Rx1day_mm",
    "Rx5day_mm",
    "R20_days",
    "CDD_days",
    "CWD_days",
    "TXx_c",
    "TNn_c",
    "TX90p_percent",
    "TN90p_percent",
    "WSDI_days"
]

historical_index_means = (
    historical_indices[
        historical_indices["year"].between(
            BASELINE_START,
            BASELINE_END
        )
    ]
    .groupby(
        "model",
        as_index=False
    )[INDEX_COLUMNS]
    .mean()
)

future_index_means = (
    future_indices
    .groupby(
        [
            "model",
            "scenario",
            "period"
        ],
        as_index=False
    )[INDEX_COLUMNS]
    .mean()
)

extreme_changes = pd.merge(
    future_index_means,
    historical_index_means,
    on="model",
    suffixes=(
        "_future",
        "_historical"
    )
)

relative_indices = [
    "PRCPTOT_mm",
    "Rx1day_mm",
    "Rx5day_mm",
    "R20_days",
    "CDD_days",
    "CWD_days"
]

absolute_indices = [
    "TXx_c",
    "TNn_c",
    "TX90p_percent",
    "TN90p_percent",
    "WSDI_days"
]

for index in relative_indices:

    baseline = extreme_changes[
        f"{index}_historical"
    ].replace(
        0,
        np.nan
    )

    extreme_changes[
        f"{index}_change"
    ] = (
        (
            extreme_changes[
                f"{index}_future"
            ]
            -
            extreme_changes[
                f"{index}_historical"
            ]
        )
        /
        baseline
        *
        100
    )

for index in absolute_indices:

    extreme_changes[
        f"{index}_change"
    ] = (
        extreme_changes[
            f"{index}_future"
        ]
        -
        extreme_changes[
            f"{index}_historical"
        ]
    )


# =============================================================================
# 21. SPI CALCULATION
# =============================================================================

def calculate_spi(
    monthly_precipitation,
    scale
):
    accumulated = (
        monthly_precipitation
        .rolling(
            scale,
            min_periods=scale
        )
        .sum()
    )

    spi = pd.Series(
        np.nan,
        index=accumulated.index
    )

    for month in range(
        1,
        13
    ):

        values = accumulated[
            accumulated.index.month
            ==
            month
        ].dropna()

        if len(values) < 10:
            continue

        zero_probability = (
            np.mean(
                values <= 0
            )
        )

        positive_values = values[
            values > 0
        ]

        if len(positive_values) < 5:
            continue

        shape, location, scale_parameter = (
            stats.gamma.fit(
                positive_values,
                floc=0
            )
        )

        probability = (
            zero_probability
            +
            (
                1 - zero_probability
            )
            *
            stats.gamma.cdf(
                np.maximum(
                    values,
                    1e-8
                ),
                shape,
                loc=location,
                scale=scale_parameter
            )
        )

        probability = np.clip(
            probability,
            1e-6,
            1 - 1e-6
        )

        spi.loc[
            values.index
        ] = stats.norm.ppf(
            probability
        )

    return spi


drought_rows = []
spi_frames = []

for model in tqdm(
    valid_models,
    desc="Calculating SPI drought indices"
):

    for scenario in SCENARIO_ORDER:

        model_data = future_corrected[
            (
                future_corrected["model"]
                ==
                model
            )
            &
            (
                future_corrected["scenario"]
                ==
                scenario
            )
        ].copy()

        monthly_precipitation = (
            model_data
            .set_index("date")[
                "precipitation_mm_day_corrected"
            ]
            .sort_index()
            .resample("MS")
            .sum(
                min_count=1
            )
        )

        for scale in SPI_SCALES:

            spi = calculate_spi(
                monthly_precipitation,
                scale
            )

            spi_frame = pd.DataFrame({
                "date": spi.index,
                "spi": spi.values,
                "model": model,
                "scenario": scenario,
                "spi_scale": scale
            })

            spi_frame["year"] = (
                spi_frame["date"]
                .dt.year
            )

            spi_frame["period"] = (
                spi_frame["year"]
                .apply(
                    assign_period
                )
            )

            spi_frames.append(
                spi_frame
            )

            for period, (
                start_year,
                end_year
            ) in FUTURE_PERIODS.items():

                period_spi = spi[
                    (
                        spi.index.year
                        >=
                        start_year
                    )
                    &
                    (
                        spi.index.year
                        <=
                        end_year
                    )
                ].dropna()

                if len(period_spi) == 0:
                    continue

                drought_condition = (
                    period_spi < -1.0
                )

                durations = []
                severities = []

                current_duration = 0
                current_severity = 0

                for value in period_spi:

                    if value < -1.0:
                        current_duration += 1
                        current_severity += -value

                    else:
                        if current_duration > 0:
                            durations.append(
                                current_duration
                            )

                            severities.append(
                                current_severity
                            )

                        current_duration = 0
                        current_severity = 0

                if current_duration > 0:
                    durations.append(
                        current_duration
                    )

                    severities.append(
                        current_severity
                    )

                drought_rows.append({
                    "model": model,
                    "scenario": scenario,
                    "spi_scale": scale,
                    "period": period,

                    "drought_month_frequency_percent":
                        np.mean(
                            period_spi < -1.0
                        )
                        *
                        100,

                    "severe_drought_frequency_percent":
                        np.mean(
                            period_spi < -1.5
                        )
                        *
                        100,

                    "extreme_drought_frequency_percent":
                        np.mean(
                            period_spi < -2.0
                        )
                        *
                        100,

                    "number_of_events":
                        len(durations),

                    "mean_duration_months":
                        np.mean(durations)
                        if durations
                        else 0,

                    "maximum_duration_months":
                        np.max(durations)
                        if durations
                        else 0,

                    "mean_severity":
                        np.mean(severities)
                        if severities
                        else 0,

                    "maximum_severity":
                        np.max(severities)
                        if severities
                        else 0
                })

spi_data = pd.concat(
    spi_frames,
    ignore_index=True
)

drought_summary = pd.DataFrame(
    drought_rows
)


# =============================================================================
# 22. ENSEMBLE SUMMARY
# =============================================================================

def ensemble_summary(
    dataframe,
    group_columns,
    value_columns
):
    rows = []

    for keys, group in dataframe.groupby(
        group_columns
    ):

        if not isinstance(keys, tuple):
            keys = (keys,)

        result = dict(
            zip(
                group_columns,
                keys
            )
        )

        result["number_of_models"] = (
            group["model"].nunique()
        )

        for column in value_columns:

            values = pd.to_numeric(
                group[column],
                errors="coerce"
            ).dropna()

            result[
                f"{column}_median"
            ] = values.median()

            result[
                f"{column}_q25"
            ] = values.quantile(
                0.25
            )

            result[
                f"{column}_q75"
            ] = values.quantile(
                0.75
            )

            result[
                f"{column}_iqr"
            ] = (
                values.quantile(0.75)
                -
                values.quantile(0.25)
            )

            positive = np.mean(
                values > 0
            )

            negative = np.mean(
                values < 0
            )

            result[
                f"{column}_agreement_percent"
            ] = max(
                positive,
                negative
            ) * 100

        rows.append(
            result
        )

    return pd.DataFrame(
        rows
    )


mean_ensemble = ensemble_summary(
    mean_change,
    [
        "scenario",
        "period"
    ],
    [
        "precipitation_change_percent",
        "tasmax_change_c",
        "tasmin_change_c"
    ]
)

extreme_variables = [
    "Rx1day_mm_change",
    "Rx5day_mm_change",
    "CDD_days_change",
    "TX90p_percent_change",
    "TN90p_percent_change",
    "WSDI_days_change"
]

extreme_ensemble = ensemble_summary(
    extreme_changes,
    [
        "scenario",
        "period"
    ],
    extreme_variables
)

drought_variables = [
    "drought_month_frequency_percent",
    "severe_drought_frequency_percent",
    "extreme_drought_frequency_percent",
    "number_of_events",
    "mean_duration_months",
    "maximum_duration_months",
    "mean_severity",
    "maximum_severity"
]

drought_ensemble = ensemble_summary(
    drought_summary,
    [
        "scenario",
        "spi_scale",
        "period"
    ],
    drought_variables
)


# =============================================================================
# 23. SAVE TABLES
# =============================================================================

performance.to_csv(
    TABLES_DIR
    /
    "Table_2_CMIP6_Model_Performance.csv",
    index=False
)

mean_ensemble.to_csv(
    TABLES_DIR
    /
    "Table_3_Projected_Mean_Climate.csv",
    index=False
)

extreme_ensemble.to_csv(
    TABLES_DIR
    /
    "Table_4_Projected_Climate_Extremes.csv",
    index=False
)

drought_ensemble.to_csv(
    TABLES_DIR
    /
    "Table_5_Projected_Drought_Characteristics.csv",
    index=False
)

with pd.ExcelWriter(
    TABLES_DIR
    /
    "SASSANDRA_Main_Manuscript_Tables.xlsx",
    engine="openpyxl"
) as writer:

    performance.to_excel(
        writer,
        sheet_name="Table 2 Model Performance",
        index=False
    )

    mean_ensemble.to_excel(
        writer,
        sheet_name="Table 3 Mean Climate",
        index=False
    )

    extreme_ensemble.to_excel(
        writer,
        sheet_name="Table 4 Extremes",
        index=False
    )

    drought_ensemble.to_excel(
        writer,
        sheet_name="Table 5 Drought",
        index=False
    )

mean_change.to_csv(
    SUPPLEMENT_DIR
    /
    "Table_S1_Model_Level_Mean_Climate.csv",
    index=False
)

extreme_changes.to_csv(
    SUPPLEMENT_DIR
    /
    "Table_S2_Model_Level_Extremes.csv",
    index=False
)

drought_summary.to_csv(
    SUPPLEMENT_DIR
    /
    "Table_S3_Model_Level_Drought.csv",
    index=False
)

historical_corrected.to_csv(
    PROCESSED_DIR
    /
    "Historical_Bias_Corrected.csv",
    index=False
)

future_corrected.to_csv(
    PROCESSED_DIR
    /
    "Future_Bias_Corrected.csv",
    index=False
)

historical_indices.to_csv(
    PROCESSED_DIR
    /
    "Historical_Climate_Indices.csv",
    index=False
)

future_indices.to_csv(
    PROCESSED_DIR
    /
    "Future_Climate_Indices.csv",
    index=False
)

spi_data.to_csv(
    PROCESSED_DIR
    /
    "Future_SPI_Time_Series.csv",
    index=False
)



# =============================================================================
# 23B. DISPLAY FINAL MANUSCRIPT TABLES IN COLAB
# =============================================================================

from IPython.display import display, Markdown

display(
    Markdown(
        "## Final manuscript tables"
    )
)

# Table 2: model-level evaluation before and after bias correction
performance_display = (
    performance
    .sort_values(
        [
            "variable",
            "model",
            "stage"
        ]
    )
    .reset_index(
        drop=True
    )
)

display(
    Markdown(
        "### Table 2. CMIP6 model performance before and after bias correction"
    )
)

display(
    performance_display.round(
        {
            "bias": 3,
            "relative_bias_percent": 2,
            "mae": 3,
            "rmse": 3,
            "correlation": 3,
            "kge": 3
        }
    )
)


# Compact ensemble performance summary for direct reporting
performance_summary = (
    performance
    .groupby(
        [
            "variable",
            "stage"
        ],
        as_index=False
    )
    .agg(
        number_of_models=(
            "model",
            "nunique"
        ),
        median_bias=(
            "bias",
            "median"
        ),
        median_relative_bias_percent=(
            "relative_bias_percent",
            "median"
        ),
        median_mae=(
            "mae",
            "median"
        ),
        median_rmse=(
            "rmse",
            "median"
        ),
        median_correlation=(
            "correlation",
            "median"
        ),
        median_kge=(
            "kge",
            "median"
        )
    )
)

performance_summary.to_csv(
    TABLES_DIR
    /
    "Table_2B_Ensemble_Model_Performance_Summary.csv",
    index=False
)

display(
    Markdown(
        "### Table 2B. Ensemble median model-performance summary"
    )
)

display(
    performance_summary.round(
        3
    )
)


display(
    Markdown(
        "### Table 3. Projected changes in mean climate"
    )
)

display(
    mean_ensemble.round(
        3
    )
)


display(
    Markdown(
        "### Table 4. Projected changes in climate extremes"
    )
)

display(
    extreme_ensemble.round(
        3
    )
)


display(
    Markdown(
        "### Table 5. Projected drought characteristics"
    )
)

display(
    drought_ensemble.round(
        3
    )
)


# Additional model-level tables used for supplementary materials
display(
    Markdown(
        "## Supplementary model-level tables"
    )
)

display(
    Markdown(
        "### Table S1. Model-level changes in mean climate"
    )
)

display(
    mean_change.round(
        3
    )
)


display(
    Markdown(
        "### Table S2. Model-level changes in climate extremes"
    )
)

display(
    extreme_changes.round(
        3
    )
)


display(
    Markdown(
        "### Table S3. Model-level drought characteristics"
    )
)

display(
    drought_summary.round(
        3
    )
)


# Save the expanded workbook containing both manuscript and summary tables
with pd.ExcelWriter(
    TABLES_DIR
    /
    "SASSANDRA_Final_Complete_Results_Tables.xlsx",
    engine="openpyxl"
) as writer:

    performance_display.to_excel(
        writer,
        sheet_name="Model Performance",
        index=False
    )

    performance_summary.to_excel(
        writer,
        sheet_name="Performance Summary",
        index=False
    )

    mean_ensemble.to_excel(
        writer,
        sheet_name="Mean Climate",
        index=False
    )

    extreme_ensemble.to_excel(
        writer,
        sheet_name="Climate Extremes",
        index=False
    )

    drought_ensemble.to_excel(
        writer,
        sheet_name="Drought",
        index=False
    )

    mean_change.to_excel(
        writer,
        sheet_name="S1 Model Mean Climate",
        index=False
    )

    extreme_changes.to_excel(
        writer,
        sheet_name="S2 Model Extremes",
        index=False
    )

    drought_summary.to_excel(
        writer,
        sheet_name="S3 Model Drought",
        index=False
    )

print(
    "Displayed and saved all final manuscript tables."
)



# =============================================================================
# 24. FIGURE 1: HISTORICAL CLIMATE MAPS
# =============================================================================

if rainfall_tif and temperature_tif:

    with rasterio.open(
        rainfall_tif
    ) as source:

        rainfall_map = source.read(
            1,
            masked=True
        )

        rainfall_extent = [
            source.bounds.left,
            source.bounds.right,
            source.bounds.bottom,
            source.bounds.top
        ]

    with rasterio.open(
        temperature_tif
    ) as source:

        tmax_map = source.read(
            1,
            masked=True
        )

        tmin_map = source.read(
            2,
            masked=True
        )

        temperature_extent = [
            source.bounds.left,
            source.bounds.right,
            source.bounds.bottom,
            source.bounds.top
        ]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(
            17,
            5.5
        )
    )

    image = axes[0].imshow(
        rainfall_map,
        extent=rainfall_extent,
        origin="upper"
    )

    axes[0].set_title(
        "(a) Mean annual precipitation"
    )

    axes[0].set_xlabel(
        "Longitude"
    )

    axes[0].set_ylabel(
        "Latitude"
    )

    figure.colorbar(
        image,
        ax=axes[0],
        label="mm year⁻¹",
        shrink=0.8
    )

    image = axes[1].imshow(
        tmax_map,
        extent=temperature_extent,
        origin="upper"
    )

    axes[1].set_title(
        "(b) Mean maximum temperature"
    )

    axes[1].set_xlabel(
        "Longitude"
    )

    axes[1].set_ylabel(
        "Latitude"
    )

    figure.colorbar(
        image,
        ax=axes[1],
        label="°C",
        shrink=0.8
    )

    image = axes[2].imshow(
        tmin_map,
        extent=temperature_extent,
        origin="upper"
    )

    axes[2].set_title(
        "(c) Mean minimum temperature"
    )

    axes[2].set_xlabel(
        "Longitude"
    )

    axes[2].set_ylabel(
        "Latitude"
    )

    figure.colorbar(
        image,
        ax=axes[2],
        label="°C",
        shrink=0.8
    )

    figure.tight_layout()

    save_figure(
        figure,
        "Figure_1_Historical_Climate.png"
    )


# =============================================================================
# 25. FIGURE 2: MODEL PERFORMANCE
# =============================================================================

performance_plot = (
    performance
    .groupby(
        [
            "variable",
            "stage"
        ],
        as_index=False
    )
    .agg(
        rmse=(
            "rmse",
            "median"
        ),
        correlation=(
            "correlation",
            "median"
        ),
        kge=(
            "kge",
            "median"
        )
    )
)

figure, axes = plt.subplots(
    1,
    3,
    figsize=(
        16,
        5
    )
)

for axis, (
    metric,
    title
) in zip(
    axes,
    [
        (
            "rmse",
            "Median RMSE"
        ),
        (
            "correlation",
            "Median correlation"
        ),
        (
            "kge",
            "Median KGE"
        )
    ]
):

    pivot = performance_plot.pivot(
        index="variable",
        columns="stage",
        values=metric
    )

    pivot.plot(
        kind="bar",
        ax=axis
    )

    axis.set_title(
        title
    )

    axis.set_xlabel(
        ""
    )

    axis.set_ylabel(
        title
    )

    axis.tick_params(
        axis="x",
        rotation=25
    )

    axis.legend(
        frameon=False
    )

figure.tight_layout()

save_figure(
    figure,
    "Figure_2_Model_Evaluation_and_Bias_Correction.png"
)


# =============================================================================
# 26. FIGURE 3: PROJECTED MEAN CLIMATE
# =============================================================================

figure, axes = plt.subplots(
    1,
    3,
    figsize=(
        17,
        5.5
    )
)

plot_variables = [
    (
        "precipitation_change_percent",
        "Precipitation change (%)"
    ),
    (
        "tasmax_change_c",
        "Maximum temperature change (°C)"
    ),
    (
        "tasmin_change_c",
        "Minimum temperature change (°C)"
    )
]

for axis, (
    variable,
    label
) in zip(
    axes,
    plot_variables
):

    plot_data = (
        mean_change
        .groupby(
            [
                "scenario",
                "period"
            ],
            as_index=False
        )[variable]
        .median()
    )

    pivot = plot_data.pivot(
        index="period",
        columns="scenario",
        values=variable
    ).reindex(
        PERIOD_ORDER
    )

    pivot.columns = [
        SCENARIO_LABELS.get(
            column,
            column
        )
        for column in pivot.columns
    ]

    pivot.plot(
        kind="bar",
        ax=axis
    )

    axis.axhline(
        0,
        linewidth=0.8
    )

    axis.set_title(
        label
    )

    axis.set_ylabel(
        label
    )

    axis.set_xlabel(
        ""
    )

    axis.tick_params(
        axis="x",
        rotation=25
    )

    axis.legend(
        frameon=False
    )

figure.tight_layout()

save_figure(
    figure,
    "Figure_3_Projected_Mean_Climate.png"
)


# =============================================================================
# 27. FIGURE 4: CLIMATE EXTREMES
# =============================================================================

extreme_labels = {
    "Rx1day_mm_change":
        "Rx1day (%)",
    "Rx5day_mm_change":
        "Rx5day (%)",
    "CDD_days_change":
        "CDD (%)",
    "TX90p_percent_change":
        "TX90p",
    "TN90p_percent_change":
        "TN90p",
    "WSDI_days_change":
        "WSDI"
}

figure, axes = plt.subplots(
    1,
    2,
    figsize=(
        17,
        6
    )
)

for axis, scenario in zip(
    axes,
    SCENARIO_ORDER
):

    plot_data = (
        extreme_changes[
            extreme_changes["scenario"]
            ==
            scenario
        ]
        .groupby(
            "period"
        )[extreme_variables]
        .median()
        .reindex(
            PERIOD_ORDER
        )
        .T
    )

    image = axis.imshow(
        plot_data.to_numpy(),
        aspect="auto"
    )

    axis.set_xticks(
        np.arange(
            len(plot_data.columns)
        )
    )

    axis.set_xticklabels(
        plot_data.columns,
        rotation=25,
        ha="right"
    )

    axis.set_yticks(
        np.arange(
            len(plot_data.index)
        )
    )

    axis.set_yticklabels([
        extreme_labels.get(
            index,
            index
        )
        for index in plot_data.index
    ])

    axis.set_title(
        SCENARIO_LABELS[
            scenario
        ]
    )

    for row in range(
        plot_data.shape[0]
    ):

        for column in range(
            plot_data.shape[1]
        ):

            axis.text(
                column,
                row,
                f"{plot_data.iloc[row, column]:.1f}",
                ha="center",
                va="center",
                fontsize=8
            )

    figure.colorbar(
        image,
        ax=axis,
        shrink=0.8,
        label="Projected change"
    )

figure.tight_layout()

save_figure(
    figure,
    "Figure_4_Projected_Climate_Extremes.png"
)


# =============================================================================
# 28. FIGURE 5: DROUGHT
# =============================================================================

figure, axes = plt.subplots(
    1,
    2,
    figsize=(
        16,
        5.5
    )
)

for axis, scenario in zip(
    axes,
    SCENARIO_ORDER
):

    plot_data = (
        drought_summary[
            (
                drought_summary["scenario"]
                ==
                scenario
            )
            &
            (
                drought_summary["spi_scale"]
                .isin(
                    [
                        3,
                        12
                    ]
                )
            )
        ]
        .groupby(
            [
                "period",
                "spi_scale"
            ],
            as_index=False
        )[
            "drought_month_frequency_percent"
        ]
        .median()
    )

    pivot = plot_data.pivot(
        index="period",
        columns="spi_scale",
        values="drought_month_frequency_percent"
    ).reindex(
        PERIOD_ORDER
    )

    pivot.columns = [
        f"SPI-{int(column)}"
        for column in pivot.columns
    ]

    pivot.plot(
        kind="bar",
        ax=axis
    )

    axis.set_title(
        SCENARIO_LABELS[
            scenario
        ]
    )

    axis.set_xlabel(
        ""
    )

    axis.set_ylabel(
        "Drought-month frequency (%)"
    )

    axis.tick_params(
        axis="x",
        rotation=25
    )

    axis.legend(
        frameon=False
    )

figure.tight_layout()

save_figure(
    figure,
    "Figure_5_Projected_Drought_Characteristics.png"
)


# =============================================================================
# 29. FIGURE 6: MODEL AGREEMENT
# =============================================================================

agreement_variables = [
    "precipitation_change_percent",
    "tasmax_change_c",
    "tasmin_change_c"
]

agreement_labels = [
    "Precipitation",
    "Maximum temperature",
    "Minimum temperature"
]

figure, axes = plt.subplots(
    1,
    2,
    figsize=(
        16,
        5.5
    )
)

for axis, scenario in zip(
    axes,
    SCENARIO_ORDER
):

    scenario_data = (
        mean_ensemble[
            mean_ensemble["scenario"]
            ==
            scenario
        ]
        .set_index(
            "period"
        )
        .reindex(
            PERIOD_ORDER
        )
    )

    matrix = np.array([
        scenario_data[
            f"{variable}_agreement_percent"
        ].to_numpy()
        for variable in agreement_variables
    ])

    image = axis.imshow(
        matrix,
        aspect="auto",
        vmin=50,
        vmax=100
    )

    axis.set_xticks(
        np.arange(
            len(PERIOD_ORDER)
        )
    )

    axis.set_xticklabels(
        PERIOD_ORDER,
        rotation=25,
        ha="right"
    )

    axis.set_yticks(
        np.arange(
            len(agreement_labels)
        )
    )

    axis.set_yticklabels(
        agreement_labels
    )

    axis.set_title(
        SCENARIO_LABELS[
            scenario
        ]
    )

    for row in range(
        matrix.shape[0]
    ):

        for column in range(
            matrix.shape[1]
        ):

            axis.text(
                column,
                row,
                f"{matrix[row, column]:.0f}%",
                ha="center",
                va="center"
            )

    figure.colorbar(
        image,
        ax=axis,
        shrink=0.8,
        label="Model agreement (%)"
    )

figure.tight_layout()

save_figure(
    figure,
    "Figure_6_Multi_Model_Agreement.png"
)


# =============================================================================
# 30. ANALYSIS MANIFEST
# =============================================================================

manifest = {
    "study_area":
        "Sassandra River Basin",

    "asset":
        "projects/ee-george946/assets/SASAN",

    "historical_period":
        "1981–2014",

    "baseline_period":
        "1995–2014",

    "future_periods":
        FUTURE_PERIODS,

    "scenarios":
        SCENARIO_LABELS,

    "models":
        valid_models,

    "number_of_models":
        len(valid_models),

    "chirps_file":
        str(chirps_file),

    "era5_file":
        str(era5_file),

    "historical_cmip6_file":
        str(historical_file),

    "number_of_future_files":
        len(future_matches),

    "output_folder":
        str(RESULTS_DIR)
}

with open(
    RESULTS_DIR
    /
    "Analysis_Manifest.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        manifest,
        file,
        indent=2,
        default=str
    )


# =============================================================================
# 31. FINAL COMPLETION REPORT
# =============================================================================

print("\n" + "=" * 88)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 88)

print("\nMain figures:")

for file in sorted(
    FIGURES_DIR.glob(
        "*.png"
    )
):
    print("✓", file.name)

print("\nMain tables:")

for file in sorted(
    TABLES_DIR.glob(
        "*"
    )
):
    print("✓", file.name)

print("\nSupplementary materials:")

for file in sorted(
    SUPPLEMENT_DIR.glob(
        "*"
    )
):
    print("✓", file.name)

print("\nProcessed datasets:")

for file in sorted(
    PROCESSED_DIR.glob(
        "*.csv"
    )
):
    print("✓", file.name)

print("\nFinal results location:")
print(RESULTS_DIR)

print("=" * 88)