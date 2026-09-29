"""
eda_freight_data.py
=====================
Exploratory data analysis for freight rate data. 
Generates and saves plots to an Images/ folder rather than
just displaying them, so the output is reusable in a report/notebook.

Function-level design (no classes), each function independently
runnable on its own DataFrame slice:
    load_and_engineer, plot_target_distribution, plot_missingness,
    plot_numeric_distributions, plot_categorical_counts,
    plot_correlation_heatmap, plot_feature_vs_target,
    plot_rate_by_equipment, plot_rate_over_time,
    plot_lane_geography, plot_circuity_vs_rate, run_eda
"""

import logging
import os


import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("freight_eda")

sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------- #
# 1. Load + engineer
# --------------------------------------------------------------------------- #
def load_and_engineer(data_path: str, target: str = "posted_rate") -> pd.DataFrame:
    """Load raw CSV and apply the same engineered features used in training
    (temporal, geo, distance-bucket) so EDA reflects the real feature set.

    """
    df = pd.read_csv(data_path)
    df = drop_id_columns(df)
    df = engineer_features(df)
    log.info("load_and_engineer: %d rows, %d columns", len(df), df.shape[1])
    return df


def _save(fig, outdir: str, filename: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path)
    return path








def drop_id_columns(df: pd.DataFrame, id_columns: list = "posted_rate") -> pd.DataFrame:
    """Remove identifier columns (e.g. load_id) that are never predictive.
    """
    id_columns = id_columns if id_columns is not None else ID_COLUMNS
    present = [c for c in id_columns if c in df.columns]
    if present:
        log.info("drop_id_columns: dropping %s", present)
    return df.drop(columns=present, errors="ignore")




# --------------------------------------------------------------------------- #
# 3. Temporal feature engineering
# --------------------------------------------------------------------------- #
def engineer_temporal_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Derive day_of_week, month, is_weekend from a date column.
    """
    df = df.copy()
    if date_col in df.columns:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        df["day_of_week"] = parsed.dt.day_name()
        df["month"] = parsed.dt.month.astype("Int64").astype(str)
        df["is_weekend"] = parsed.dt.dayofweek.isin([5, 6]).astype(int)
    else:
        df["day_of_week"] = "unknown"
        df["month"] = "unknown"
        df["is_weekend"] = 0
    return df



# --------------------------------------------------------------------------- #
# 4. Geo feature engineering
# --------------------------------------------------------------------------- #
def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized haversine distance in miles between two lat/lon arrays.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 3958.8 * np.arcsin(np.sqrt(a))



def engineer_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive origin/destination region, haversine_distance, and
    route_circuity (route distance / straight-line distance).

    """
    df = df.copy()

    for col, new_col in (("pickup", "origin_region"), ("delivery", "destination_region")):
        if col in df.columns:
            df[new_col] = df[col].astype(str).str.split(",").str[-1].str.strip()
        elif new_col not in df.columns:
            df[new_col] = "unknown"

    geo_cols = {"pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"}
    if geo_cols.issubset(df.columns):
        df["haversine_distance"] = haversine_miles(
            df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"]
        )
        if "distance" in df.columns:
            df["route_circuity"] = df["distance"] / df["haversine_distance"].replace(0, np.nan)
        else:
            df["route_circuity"] = np.nan
    else:
        df["haversine_distance"] = np.nan
        df["route_circuity"] = np.nan

    return df


# --------------------------------------------------------------------------- #
# 5. Distance bucketing
# --------------------------------------------------------------------------- #
def engineer_distance_buckets(df: pd.DataFrame, distance_col: str = "distance") -> pd.DataFrame:
    """Bucket raw distance into short/medium/long/very_long/cross_country.

    """
    df = df.copy()
    if distance_col in df.columns:
        df["distance_bucket"] = pd.cut(
            df[distance_col],
            bins=[-0.01, 250, 500, 1000, 2000, np.inf],
            labels=["short", "medium", "long", "very_long", "cross_country"],
        ).astype(str)
    else:
        df["distance_bucket"] = "unknown"
    return df



# --------------------------------------------------------------------------- #
# 6. Full feature engineering (composes the above)
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame, date_col: str = "date", distance_col: str = "distance") -> pd.DataFrame:
    """Run all feature engineering steps in sequence.
    """
    df = engineer_temporal_features(df, date_col=date_col)
    df = engineer_geo_features(df)
    df = engineer_distance_buckets(df, distance_col=distance_col)
    return df


# --------------------------------------------------------------------------- #
# 7. Feature list definitions
# --------------------------------------------------------------------------- #
def get_feature_lists() -> tuple:
    """Return (numeric_features, categorical_features) column name lists.

    """
    numeric_features = [
        "distance",
        "weight",
        "market_index",
        "quote_signal",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "haversine_distance",
        "route_circuity",
    ]
    categorical_features = [
        "origin_region",
        "destination_region",
        "equipment",
        "day_of_week",
        "month",
        "distance_bucket",
    ]
    return numeric_features, categorical_features













# --------------------------------------------------------------------------- #
# 2. Target distribution
# --------------------------------------------------------------------------- #
def plot_target_distribution(df: pd.DataFrame, target: str, outdir: str) -> str:
    """Histogram + boxplot of the target variable (posted_rate).

    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df[target].dropna(), kde=True, ax=axes[0], color="#2E86AB")
    axes[0].set_title(f"Distribution of {target}")
    axes[0].set_xlabel(target)

    sns.boxplot(x=df[target].dropna(), ax=axes[1], color="#2E86AB")
    axes[1].set_title(f"{target} — outlier check")
    fig.tight_layout()
    return _save(fig, outdir, "01_target_distribution.png")


# --------------------------------------------------------------------------- #
# 3. Missingness
# --------------------------------------------------------------------------- #
def plot_missingness(df: pd.DataFrame, outdir: str) -> str:
    """Bar chart of % missing values per column (only columns with >0
    missing are shown; returns early with a placeholder note if none).
    """
    missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(missing_pct) + 1)))
    if missing_pct.empty:
        ax.text(0.5, 0.5, "No missing values detected", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
    else:
        sns.barplot(x=missing_pct.values, y=missing_pct.index, ax=ax, color="#E63946")
        ax.set_xlabel("% missing")
        ax.set_title("Missing values by column")
    fig.tight_layout()
    return _save(fig, outdir, "02_missingness.png")


# --------------------------------------------------------------------------- #
# 4. Numeric feature distributions
# --------------------------------------------------------------------------- #
def plot_numeric_distributions(df: pd.DataFrame, numeric_features: list, outdir: str) -> str:
    """Grid of histograms, one per numeric feature.
    """
    cols = [c for c in numeric_features if c in df.columns]
    n_cols = 3
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#457B9D")
        axes[i].set_title(col)
    for j in range(len(cols), len(axes)):
        axes[j].set_axis_off()
    fig.suptitle("Numeric feature distributions", y=1.02)
    fig.tight_layout()
    return _save(fig, outdir, "03_numeric_distributions.png")


# --------------------------------------------------------------------------- #
# 5. Categorical feature counts
# --------------------------------------------------------------------------- #
def plot_categorical_counts(df: pd.DataFrame, categorical_features: list, outdir: str, top_n: int = 10) -> str:
    """Grid of bar charts, one per categorical feature (top N categories
    by frequency for high-cardinality columns like origin_region).
    """
    cols = [c for c in categorical_features if c in df.columns]
    n_cols = 2
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(cols):
        counts = df[col].value_counts().head(top_n)
        sns.barplot(x=counts.values, y=counts.index, ax=axes[i], color="#1D3557")
        axes[i].set_title(f"{col} (top {min(top_n, len(counts))})")
        axes[i].set_xlabel("count")
    for j in range(len(cols), len(axes)):
        axes[j].set_axis_off()
    fig.suptitle("Categorical feature counts", y=1.02)
    fig.tight_layout()
    return _save(fig, outdir, "04_categorical_counts.png")


# --------------------------------------------------------------------------- #
# 6. Correlation heatmap
# --------------------------------------------------------------------------- #
def plot_correlation_heatmap(df: pd.DataFrame, numeric_features: list, target: str, outdir: str) -> str:
    """Correlation heatmap across numeric features + target.

    """
    cols = [c for c in numeric_features if c in df.columns] + [target]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(0.7 * len(cols) + 2, 0.7 * len(cols) + 2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation matrix (numeric features + target)")
    fig.tight_layout()
    return _save(fig, outdir, "05_correlation_heatmap.png")


# --------------------------------------------------------------------------- #
# 7. Feature vs target scatter plots
# --------------------------------------------------------------------------- #
def plot_feature_vs_target(df: pd.DataFrame, numeric_features: list, target: str, outdir: str) -> str:
    """Grid of scatter plots (with regression line) for each numeric
    feature against the target — the single most useful EDA view for a
    regression problem: which features actually move the target?

    """
    cols = [c for c in numeric_features if c in df.columns and c != target]
    n_cols = 3
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(cols):
        sample = df[[col, target]].dropna()
        if len(sample) > 2000:
            sample = sample.sample(2000, random_state=42)
        sns.regplot(
            data=sample, x=col, y=target, ax=axes[i],
            scatter_kws={"alpha": 0.3, "s": 12, "color": "#457B9D"},
            line_kws={"color": "#E63946"},
        )
        r = df[[col, target]].corr(numeric_only=True).iloc[0, 1]
        axes[i].set_title(f"{col}  (r={r:.2f})")
    for j in range(len(cols), len(axes)):
        axes[j].set_axis_off()
    fig.suptitle(f"Feature vs {target}", y=1.02)
    fig.tight_layout()
    return _save(fig, outdir, "06_feature_vs_target.png")


# --------------------------------------------------------------------------- #
# 8. Rate by equipment type
# --------------------------------------------------------------------------- #
def plot_rate_by_equipment(df: pd.DataFrame, target: str, outdir: str, equipment_col: str = "equipment") -> str:
    """Boxplot + violin of target by equipment type.

    """
    if equipment_col not in df.columns:
        log.warning("plot_rate_by_equipment: '%s' not in columns, skipping", equipment_col)
        return ""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    order = df.groupby(equipment_col)[target].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x=equipment_col, y=target, order=order, hue=equipment_col, palette="Set2", legend=False, ax=axes[0])
    axes[0].set_title(f"{target} by {equipment_col} (boxplot)")
    sns.violinplot(data=df, x=equipment_col, y=target, order=order, hue=equipment_col, palette="Set2", legend=False, ax=axes[1])
    axes[1].set_title(f"{target} by {equipment_col} (violin)")
    fig.tight_layout()
    return _save(fig, outdir, "07_rate_by_equipment.png")


# --------------------------------------------------------------------------- #
# 9. Rate over time
# --------------------------------------------------------------------------- #
def plot_rate_over_time(df: pd.DataFrame, target: str, outdir: str, date_col: str = "date") -> str:
    """Line plot of mean target by day, plus a day-of-week boxplot, to
    surface any temporal / seasonality signal.

    """
    if date_col not in df.columns:
        log.warning("plot_rate_over_time: '%s' not in columns, skipping", date_col)
        return ""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    daily = df.set_index(date_col)[target].resample("D").mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    daily.plot(ax=axes[0], color="#2E86AB", marker="o", markersize=3)
    axes[0].set_title(f"Mean {target} by day")
    axes[0].set_ylabel(target)

    if "day_of_week" in df.columns:
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        present = [d for d in dow_order if d in df["day_of_week"].unique()]
        sns.boxplot(data=df, x="day_of_week", y=target, order=present, ax=axes[1], color="#A8DADC")
        axes[1].set_title(f"{target} by day of week")
        axes[1].tick_params(axis="x", rotation=30)
    else:
        axes[1].set_axis_off()

    fig.tight_layout()
    return _save(fig, outdir, "08_rate_over_time.png")


# --------------------------------------------------------------------------- #
# 10. Lane geography
# --------------------------------------------------------------------------- #
def plot_lane_geography(df: pd.DataFrame, target: str, outdir: str) -> str:
    """Scatter of pickup locations colored by rate, and top origin/destination
    region frequency — a geographic sanity check on lane coverage.
    """
    needed = {"pickup_lat", "pickup_lon"}
    if not needed.issubset(df.columns):
        log.warning("plot_lane_geography: missing lat/lon columns, skipping")
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sc = axes[0].scatter(
        df["pickup_lon"], df["pickup_lat"], c=df[target], cmap="viridis", alpha=0.6, s=20
    )
    axes[0].set_title(f"Pickup locations colored by {target}")
    axes[0].set_xlabel("longitude")
    axes[0].set_ylabel("latitude")
    fig.colorbar(sc, ax=axes[0], label=target, shrink=0.8)

    if "origin_region" in df.columns:
        top_regions = df["origin_region"].value_counts().head(10)
        sns.barplot(x=top_regions.values, y=top_regions.index, ax=axes[1], color="#1D3557")
        axes[1].set_title("Top origin regions by load count")
        axes[1].set_xlabel("count")
    else:
        axes[1].set_axis_off()

    fig.tight_layout()
    return _save(fig, outdir, "09_lane_geography.png")


# --------------------------------------------------------------------------- #
# 11. Route circuity vs rate
# --------------------------------------------------------------------------- #
def plot_circuity_vs_rate(df: pd.DataFrame, target: str, outdir: str) -> str:
    """Scatter of route_circuity vs target, colored by distance_bucket —
    checks whether indirect routes (circuity > 1) carry a rate premium.

    """
    if "route_circuity" not in df.columns:
        log.warning("plot_circuity_vs_rate: 'route_circuity' not in columns, skipping")
        return ""
    fig, ax = plt.subplots(figsize=(7, 5))
    sample = df.dropna(subset=["route_circuity", target])
    hue = "distance_bucket" if "distance_bucket" in df.columns else None
    sns.scatterplot(data=sample, x="route_circuity", y=target, hue=hue, alpha=0.6, s=25, ax=ax, palette="Set2")
    ax.set_title(f"Route circuity vs {target}")
    ax.set_xlabel("route_circuity (route distance / straight-line distance)")
    fig.tight_layout()
    return _save(fig, outdir, "10_circuity_vs_rate.png")






# --------------------------------------------------------------------------- #
# 12. Orchestrator
# --------------------------------------------------------------------------- #
def run_eda(data_path: str, target: str = "posted_rate", outdir: str = "Images") -> list:
    """Run every EDA plot function above in sequence, saving each to
    outdir. Returns the list of saved file paths.

    This function does no analysis itself — it only calls the standalone
    functions above in order, so you can equally well call any one of
    them yourself on your own DataFrame slice.
    """
    df = load_and_engineer(data_path, target)
    numeric_features, categorical_features = get_feature_lists()

    saved = []
    saved.append(plot_target_distribution(df, target, outdir))
    saved.append(plot_missingness(df, outdir))
    saved.append(plot_numeric_distributions(df, numeric_features, outdir))
    saved.append(plot_categorical_counts(df, categorical_features, outdir))
    saved.append(plot_correlation_heatmap(df, numeric_features, target, outdir))
    saved.append(plot_feature_vs_target(df, numeric_features, target, outdir))
    saved.append(plot_rate_by_equipment(df, target, outdir))
    saved.append(plot_rate_over_time(df, target, outdir))
    saved.append(plot_lane_geography(df, target, outdir))
    saved.append(plot_circuity_vs_rate(df, target, outdir))

    saved = [p for p in saved if p]  # drop skipped ("") entries
    log.info("run_eda: saved %d plots to %s/", len(saved), outdir)
    return saved

run_eda("./data/train-test.csv", "posted_rate", "./Images")