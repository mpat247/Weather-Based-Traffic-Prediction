#!/usr/bin/env python3
"""
basicPreprocess.py – builds basic_final_dataset.* in repo root
(Only TMC traffic + Env‑Canada weather, no ERA5 / collision extras.)

Run:
    python basicPreprocess.py
"""

import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# ───────────────────────────── #
#  Constants / paths
# ───────────────────────────── #
ROOT = Path(__file__).resolve().parent
RAW  = ROOT                      # CSVs live next to this script
EARTH_R_KM = 6371.0088           # (unused, kept for future)

# ───────────────────────────── #
#  Simple helpers
# ───────────────────────────── #
def loadCsv(name: str) -> pd.DataFrame:
    """Read a CSV in the repo root with low‑memory off."""
    return pd.read_csv(RAW / name, low_memory=False)


def loadDatasets():
    """Return the two dataframes we actually need."""
    df_tmc = loadCsv("tmc_raw_data_2010_2019.csv")
    df_env = loadCsv("hourly_final.csv")      # Env Canada
    return df_tmc, df_env


# ───────────────────────────── #
#  Feature builders
# ───────────────────────────── #
def buildTrueVolume(df_tmc: pd.DataFrame) -> pd.DataFrame:
    """
    Sum any *_car / *_truck / *_bike … columns into total_traffic_volume.
    """
    mode_cols = [
        c for c in df_tmc.columns
        if c.lower().endswith(("_car", "_cars", "_truck", "_bus", "_bike", "_peds"))
    ]
    if not mode_cols:
        raise RuntimeError("Could not locate turning‑movement count columns.")
    df_tmc["total_traffic_volume"] = df_tmc[mode_cols].sum(axis=1)
    return df_tmc


def addCongestionLabel(df: pd.DataFrame) -> pd.DataFrame:
    q33, q66 = df["total_traffic_volume"].quantile([0.33, 0.66]).values
    df["congestion_level"] = pd.cut(
        df.total_traffic_volume,
        [-np.inf, q33, q66, np.inf],
        labels=["Low", "Medium", "High"]
    )
    return df


def buildWeatherFeatures(df_env: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Env‑Canada hourly file into:
      • datetime_hour
      • temp_c, precip_flag, wind_speed
      • lat_round, lon_round  (for spatial merge)

    Handles both:
      - columns named lat / lon
      - columns named x / y  (rename → lon / lat)
    """
    # ── coords ──────────────────────────────────────────────
    if {"lat", "lon"}.issubset(df_env.columns) is False:
        # Env‑Canada convention: x = longitude, y = latitude
        if {"x", "y"}.issubset(df_env.columns):
            df_env = df_env.rename(columns={"x": "lon", "y": "lat"})
        else:
            raise KeyError(
                "hourly_final.csv needs coordinate columns. "
                "Expected lat/lon or x/y."
            )

    # ── timestamp ───────────────────────────────────────────
    df_env["timestamp"] = (
        pd.to_datetime(df_env["LOCAL_DATE"], errors="coerce")
        + pd.to_timedelta(df_env["LOCAL_HOUR"], unit="h")
    )
    df_env["datetime_hour"] = df_env["timestamp"].dt.floor("h")

    # ── basic weather metrics ───────────────────────────────
    df_env["temp_c"] = df_env["TEMP"]
    df_env["precip_flag"] = (df_env.get("PRECIP_AMOUNT", 0) > 0).astype(int)

    if {"WIND_U_10", "WIND_V_10"}.issubset(df_env.columns):
        u, v = df_env["WIND_U_10"], df_env["WIND_V_10"]
        df_env["wind_speed"] = np.sqrt(u * u + v * v)
    else:
        df_env["wind_speed"] = np.nan

    # ── rounding for merge ──────────────────────────────────
    df_env["lat_round"] = df_env["lat"].round(3)
    df_env["lon_round"] = df_env["lon"].round(3)

    return df_env



# ───────────────────────────── #
#  Master build
# ───────────────────────────── #
def buildDataset(start="2015-01-01", end="2020-12-31") -> pd.DataFrame:
    df_tmc, df_env = loadDatasets()

    # ---- TMC ----
    df_tmc = df_tmc.rename(columns={"longitude": "lon", "latitude": "lat"})
    df_tmc["count_date"] = pd.to_datetime(df_tmc["count_date"], errors="coerce")
    df_tmc = df_tmc.query("@start <= count_date <= @end").copy()
    df_tmc = buildTrueVolume(df_tmc)
    df_tmc["datetime_hour"] = df_tmc["count_date"].dt.floor("h")
    df_tmc["lat_round"] = df_tmc["lat"].round(3)
    df_tmc["lon_round"] = df_tmc["lon"].round(3)

    # ---- Weather ----
    df_env = buildWeatherFeatures(df_env)

    # ---- Merge on hour + rounded coords ----
    merged = pd.merge(
        df_tmc,
        df_env[[
            "datetime_hour", "lat_round", "lon_round",
            "temp_c", "wind_speed", "precip_flag"
        ]],
        on=["datetime_hour", "lat_round", "lon_round"],
        how="left"
    )

    merged = addCongestionLabel(merged)

    cols = [
        "datetime_hour", "lat", "lon",
        "temp_c", "wind_speed", "precip_flag",
        "total_traffic_volume", "congestion_level"
    ]
    return merged[cols]


# ───────────────────────────── #
#  CLI
# ───────────────────────────── #
def main():
    ap = argparse.ArgumentParser("Rebuild basic_final_dataset.*")
    ap.add_argument("--csv",     default="basic_final_dataset.csv")
    ap.add_argument("--parquet", default="basic_final_dataset.parquet")
    args = ap.parse_args()

    print("⏳  Building basic_final_dataset …")
    df_final = buildDataset()
    df_final.to_csv(ROOT / args.csv, index=False)
    df_final.to_parquet(ROOT / args.parquet, index=False)
    print(f"✅  Saved {len(df_final):,} rows → {args.csv} & {args.parquet}")


if __name__ == "__main__":
    main()
