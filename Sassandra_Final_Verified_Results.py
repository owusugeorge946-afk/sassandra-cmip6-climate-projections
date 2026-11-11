"""Rebuild manuscript figures/tables from audited Sassandra CMIP6 CSV outputs.

This script intentionally does not rerun or apply an additional QDM adjustment.
It reads the independently verified Step 9–19 products saved by Google Colab.

Run in Colab:
    %run Sassandra_Final_Verified_Results.py

Or locally:
    python Sassandra_Final_Verified_Results.py --source /path/to/Sassandra_CMIP6_Final_Verification
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS_EXPECTED = 10
PERIODS = [("2021–2040", 2021, 2040), ("2041–2060", 2041, 2060), ("2081–2100", 2081, 2100)]
SCENARIOS = ("ssp245", "ssp585")
LABELS = {"ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}
COLORS = {"ssp245": "#32658A", "ssp585": "#B45142"}
HISTORICAL = {"TXmean_c": 31.416, "TNmean_c": 20.444, "DTRmean_c": 10.972,
              "PRCPTOT_mm": 1548.051, "RX1day_mm": 22.945, "RX5day_mm": 67.304}
FIG5 = [("TXmean_c", "Mean maximum temperature", "°C"),
        ("TNmean_c", "Mean minimum temperature", "°C"),
        ("PRCPTOT_mm", "Annual precipitation", "mm/year"),
        ("RX1day_mm", "Maximum 1-day precipitation", "mm"),
        ("RX5day_mm", "Maximum 5-day precipitation", "mm"),
        ("DTRmean_c", "Diurnal temperature range", "°C")]
FIG6 = [("RX1day_mm", "Maximum 1-day rainfall", "mm"),
        ("RX5day_mm", "Maximum 5-day rainfall", "mm"),
        ("CDD_days", "Consecutive dry days", "days"),
        ("TX90p_pct", "Warm-day frequency", "percentage points"),
        ("TN90p_pct", "Warm-night frequency", "percentage points"),
        ("WSDI_days", "Warm-spell duration", "days")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.facecolor": "white", "figure.facecolor": "white"})


def period_key(value: str) -> str:
    nums = re.findall(r"20\d{2}|2100", str(value))
    if len(nums) >= 2:
        return f"{nums[-2]}–{nums[-1]}"
    raise ValueError(f"Cannot identify the projection period: {value!r}")


def discover(source: Path, required: set[str], preferred: str) -> tuple[Path, pd.DataFrame]:
    """Find one CSV by its schema; reject ambiguous, unverified substitutions."""
    matches = []
    for path in sorted(source.rglob("*.csv")):
        try:
            cols = set(pd.read_csv(path, nrows=0).columns)
        except (ValueError, UnicodeError, OSError):
            continue
        if required.issubset(cols):
            score = int(preferred.lower() in path.name.lower())
            matches.append((score, path))
    if not matches:
        raise FileNotFoundError(f"No CSV with columns {sorted(required)} under {source}")
    matches.sort(key=lambda item: (-item[0], str(item[1])))
    best = [x for x in matches if x[0] == matches[0][0]]
    if len(best) != 1:
        raise ValueError("Ambiguous input files; keep only one matching verified export for "
                         + preferred + ": " + ", ".join(str(x[1]) for x in best))
    path = best[0][1]
    return path, pd.read_csv(path)


def periods(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["period_key"] = frame["period"].map(period_key)
    return frame


def validated_summary(frame: pd.DataFrame, variables: set[str], label: str) -> pd.DataFrame:
    frame = periods(frame)
    frame = frame[frame["variable"].isin(variables)].copy()
    expected = {(s, p, v) for s in SCENARIOS for p, _, _ in PERIODS for v in variables}
    records = list(zip(frame.scenario, frame.period_key, frame.variable))
    if len(records) != len(expected) or set(records) != expected:
        raise ValueError(f"{label}: missing/duplicate model-period comparisons")
    if not (frame["models"].eq(MODELS_EXPECTED)).all():
        raise ValueError(f"{label}: expected {MODELS_EXPECTED} models per comparison")
    for v, baseline in HISTORICAL.items():
        sub = frame[frame.variable.eq(v)]
        if not sub.empty and "historical_mean" in sub.columns:
            if not np.allclose(sub.historical_mean, baseline, atol=.01):
                raise ValueError(f"{label}: {v} baseline disagrees with the verified 1995–2014 value")
    return frame


def assemble(source: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    provenance = {}
    parts = []
    groups = [
        ("Step9", {"TXmean_c", "TNmean_c", "DTRmean_c", "PRCPTOT_mm", "RX1day_mm", "RX5day_mm"}, "Step9_Exact_Period_Ensemble_Summary"),
        ("Step12", {"CDD_days", "CWD_days"}, "Step12"),
        ("Step13", {"TX90p_pct", "TN90p_pct", "WSDI_days"}, "Step13"),
    ]
    for label, variables, preferred in groups:
        path, data = discover(source,
             {"scenario", "period", "variable", "ensemble_mean_change", "model_sd", "models"}, preferred)
        provenance[label] = str(path)
        parts.append(validated_summary(data, variables, label))
    combined = pd.concat(parts, ignore_index=True)
    if combined.duplicated(["scenario", "period_key", "variable"]).any():
        raise ValueError("Duplicated comparison after combining verified analyses")
    return combined, provenance


def plot_period_changes(data: pd.DataFrame, metrics: list[tuple[str, str, str]], filename: Path):
    fig, axes = plt.subplots(3, 2, figsize=(10, 11))
    for i, (ax, (variable, name, unit)) in enumerate(zip(axes.flat, metrics)):
        for sc, xshift in (("ssp245", -.10), ("ssp585", .10)):
            sub = data[(data.variable == variable) & (data.scenario == sc)]
            sub = sub.set_index("period_key")
            rows = [sub.loc[p] for p, _, _ in PERIODS]
            mean = [r.ensemble_mean_change for r in rows]
            sd = [r.model_sd for r in rows]
            ax.errorbar(np.arange(3) + xshift, mean, yerr=sd,
                        marker="o", mfc="white", mew=1.5, capsize=3,
                        lw=1.4, color=COLORS[sc], label=LABELS[sc])
        ax.axhline(0, lw=.8, ls="--", color="0.60")
        ax.set_xticks(np.arange(3), [p for p, _, _ in PERIODS], fontsize=8)
        ax.set_ylabel(f"Change ({unit})")
        ax.set_title(f"({chr(97 + i)}) {name}", loc="left", fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=.15)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(top=.95, wspace=.28, hspace=.34)
    fig.savefig(filename, dpi=1000, bbox_inches="tight")
    plt.close(fig)


def plot_annual_trends(source: Path, out: Path, provenance: dict[str, str]):
    try:
        path, frame = discover(source, {"scenario", "year", "TXmean_c", "TNmean_c", "PRCPTOT_mm"}, "Step15")
    except FileNotFoundError:
        return  # no verified annual ensemble series in this exported folder
    frame = frame[frame.scenario.isin(SCENARIOS) & frame.year.between(2021, 2100)]
    if len(frame) != 160 or frame.duplicated(["scenario", "year"]).any():
        raise ValueError("Annual climate ensemble must have 80 years per scenario")
    provenance["Step15_annual"] = str(path)
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax, (var, name) in zip(axs, [("PRCPTOT_mm", "Annual precipitation (mm)"),
                                ("TXmean_c", "Mean maximum temperature (°C)"),
                                ("TNmean_c", "Mean minimum temperature (°C)")]):
        for sc in SCENARIOS:
            d = frame[frame.scenario.eq(sc)].sort_values("year")
            ax.plot(d.year, d[var], lw=1.4, color=COLORS[sc], label=LABELS[sc])
        ax.set_ylabel(name);ax.grid(alpha=.15)
    axs[0].legend(ncol=2, frameon=False)
    axs[-1].set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(out / "Figure_8_Annual_Ensemble_Median_Trends.png", dpi=1000, bbox_inches="tight")
    plt.close(fig)


def report(source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    series, provenance = assemble(source)
    plot_period_changes(series, FIG5, output / "Figure_5_Verified_Period_Changes.png")
    plot_period_changes(series, FIG6, output / "Figure_6_Verified_Extreme_Changes.png")
    series.sort_values(["scenario", "variable", "period_key"]).to_csv(
        output / "Verified_Model_Paired_Ensemble_Changes.csv", index=False)
    plot_annual_trends(source, output, provenance)
    manifest = {"source_directory": str(source), "models": MODELS_EXPECTED,
                "baseline": "1995–2014", "scenarios": list(SCENARIOS),
                "additional_qdm_applied": False, "spi_scales_verified": [3, 12],
                "ensemble_periods": [x[0] for x in PERIODS],
                "error_bars": "sample standard deviation across ten model-paired changes",
                "inputs": provenance,
                "limitations": ["Existing verified source CSVs are required", 
                                "This is post-processing, not a replacement for the Step 9–19 Colab calculations"]}
    (output / "Reconstruction_Manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Verified post-processing complete:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=Path("/content/drive/MyDrive/Sassandra_CMIP6_Final_Verification"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    src = args.source.expanduser().resolve()
    if not src.is_dir():
        raise SystemExit("Verified output folder not found: " + str(src) + "\nMount Drive or supply --source.")
    report(src, args.output or src / "GitHub_Final_Reconstructed_Figures")
