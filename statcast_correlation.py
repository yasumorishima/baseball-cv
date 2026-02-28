"""Step 4: Skeleton features × Statcast/OBP metrics correlation analysis.

Uses Driveline OBP metadata (pitch speed, exit velocity) and skeleton features
(joint angles, angular velocities) to find correlations between body mechanics
and performance outcomes.

Usage:
    python statcast_correlation.py                      # Default: pitching
    python statcast_correlation.py --mode hitting        # Hitting analysis
    python statcast_correlation.py --mode both            # Both

Output:
    data/output/correlation_{mode}.png
    data/output/correlation_{mode}.csv
    data/output/scatter_{mode}.png
"""

import argparse
from pathlib import Path

import ezc3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from skeleton_analysis import (
    compute_angular_velocity,
    compute_elbow_flexion,
    compute_knee_flexion,
    compute_lead_leg_block_features,
    compute_shoulder_abduction,
    compute_trunk_rotation,
    load_c3d,
)

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/output")


def parse_pitching_filename(filename):
    """Extract metadata from pitching C3D filename.

    Format: USERid_SESSIONid_HEIGHT_WEIGHT_PITCHNUMBER_PITCHTYPE_PITCHSPEED.c3d
    Speed: last 3 digits with implied decimal (e.g., 809 = 80.9 mph)
    """
    parts = Path(filename).stem.split("_")
    if len(parts) < 7:
        return None
    return {
        "user_id": parts[0],
        "session_id": parts[1],
        "height_in": int(parts[2]),
        "weight_lb": int(parts[3]),
        "pitch_number": int(parts[4]),
        "pitch_type": parts[5],
        "pitch_speed_mph": int(parts[6]) / 10.0,
    }


def parse_hitting_filename(filename):
    """Extract metadata from hitting C3D filename.

    Format: USERid_SESSIONid_HEIGHT_WEIGHT_SIDE_SWINGNUMBER_EXITVELOCITY.c3d
    """
    parts = Path(filename).stem.split("_")
    if len(parts) < 7:
        return None
    return {
        "user_id": parts[0],
        "session_id": parts[1],
        "height_in": int(parts[2]),
        "weight_lb": int(parts[3]),
        "side": parts[4],
        "swing_number": int(parts[5]),
        "exit_velocity_mph": int(parts[6]) / 10.0,
    }


def extract_peak_features(markers, rate, mode="pitching"):
    """Extract peak joint angle and angular velocity features from a C3D file."""
    n_frames = next(iter(markers.values())).shape[1]
    features = {}

    # Throwing/hitting arm
    side = "R"  # default right-handed

    # Elbow flexion
    elbow = compute_elbow_flexion(markers, side)
    if elbow is not None:
        features["peak_elbow_flexion"] = np.nanmax(elbow)
        features["min_elbow_flexion"] = np.nanmin(elbow)
        features["elbow_rom"] = np.nanmax(elbow) - np.nanmin(elbow)
        vel = compute_angular_velocity(elbow, rate)
        features["peak_elbow_velocity"] = np.nanmax(np.abs(vel))

    # Shoulder abduction
    shoulder = compute_shoulder_abduction(markers, side)
    if shoulder is not None:
        features["peak_shoulder_abduction"] = np.nanmax(shoulder)
        vel = compute_angular_velocity(shoulder, rate)
        features["peak_shoulder_velocity"] = np.nanmax(np.abs(vel))

    # Trunk rotation
    trunk = compute_trunk_rotation(markers)
    if trunk is not None:
        features["peak_trunk_rotation"] = np.nanmax(trunk)
        features["trunk_rotation_range"] = np.nanmax(trunk) - np.nanmin(trunk)
        vel = compute_angular_velocity(trunk, rate)
        features["peak_trunk_velocity"] = np.nanmax(np.abs(vel))

    # Knee flexion (lead leg)
    lead_side = "L" if side == "R" else "R"  # lead leg is opposite of throwing arm
    knee = compute_knee_flexion(markers, lead_side)
    if knee is not None:
        features["peak_knee_flexion"] = np.nanmax(knee)
        features["min_knee_flexion"] = np.nanmin(knee)

    # Lead leg block features
    llb = compute_lead_leg_block_features(markers, rate, side=lead_side, verbose=True)
    features.update(llb)

    return features


def process_multiple_files(mode="pitching"):
    """Process all available C3D files and extract features + metadata."""
    c3d_dir = RAW_DIR
    rows = []

    # Find all C3D files matching the mode
    pattern = f"{mode}_sample*.c3d" if mode == "pitching" else f"{mode}_sample*.c3d"
    c3d_files = sorted(c3d_dir.glob("*.c3d"))

    # Filter to mode-relevant files
    parse_fn = parse_pitching_filename if mode == "pitching" else parse_hitting_filename
    speed_key = "pitch_speed_mph" if mode == "pitching" else "exit_velocity_mph"

    for fpath in c3d_files:
        meta = parse_fn(fpath.name)
        if meta is None:
            # Try as sample file
            if mode in fpath.name:
                meta = {speed_key: None}
            else:
                continue

        try:
            markers, rate, n_frames = load_c3d(str(fpath))
            features = extract_peak_features(markers, rate, mode)
            row = {**meta, **features, "filename": fpath.name}
            rows.append(row)
            print(f"  Processed: {fpath.name}")
        except Exception as e:
            print(f"  Error processing {fpath.name}: {e}")

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def download_additional_samples(mode="pitching", n_samples=15):
    """Download additional C3D files from Driveline OBP for correlation analysis."""
    import json
    import urllib.request

    base_url = "https://api.github.com/repos/drivelineresearch/openbiomechanics/contents"
    data_path = f"baseball_{mode}/data/c3d"
    api_url = f"{base_url}/{data_path}?per_page=100"

    print(f"  Fetching folder list from Driveline OBP ({mode})...")
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "baseball-cv"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            folders = json.loads(resp.read())
    except Exception as e:
        print(f"  Error fetching folder list: {e}")
        return []

    if not isinstance(folders, list):
        print(f"  Unexpected API response: {folders}")
        return []

    downloaded = []
    count = 0

    for folder in folders:
        if count >= n_samples:
            break
        folder_name = folder["name"]
        folder_url = f"{base_url}/{data_path}/{folder_name}"

        try:
            req = urllib.request.Request(folder_url, headers={"User-Agent": "baseball-cv"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                files = json.loads(resp.read())
        except Exception:
            continue

        if not isinstance(files, list):
            continue

        # Pick the first non-model C3D file
        for f in files:
            fname = f["name"]
            if fname.endswith(".c3d") and "model" not in fname:
                raw_url = f"https://github.com/drivelineresearch/openbiomechanics/raw/main/{data_path}/{folder_name}/{fname}"
                local_path = RAW_DIR / fname
                if local_path.exists():
                    downloaded.append(local_path)
                    count += 1
                    break

                print(f"  Downloading: {fname}")
                try:
                    urllib.request.urlretrieve(raw_url, str(local_path))
                    # Verify it's a valid C3D
                    ezc3d.c3d(str(local_path))
                    downloaded.append(local_path)
                    count += 1
                except Exception as e:
                    print(f"    Failed: {e}")
                    if local_path.exists():
                        local_path.unlink()
                break

    print(f"  Downloaded {len(downloaded)} C3D files for {mode}")
    return downloaded


def plot_correlation_matrix(df, mode, output_dir):
    """Plot correlation heatmap of skeleton features vs performance."""
    speed_key = "pitch_speed_mph" if mode == "pitching" else "exit_velocity_mph"
    feature_cols = [c for c in df.columns if c not in (
        "user_id", "session_id", "height_in", "weight_lb",
        "pitch_number", "pitch_type", "side", "swing_number",
        "filename", speed_key,
    )]

    if speed_key not in df.columns or df[speed_key].isna().all():
        print("  No speed data available for correlation matrix")
        return

    cols = [speed_key] + feature_cols
    corr_df = df[cols].dropna(axis=1, how="all").corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_df, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.columns)))
    labels = [c.replace("_", "\n") for c in corr_df.columns]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # Add correlation values
    for i in range(len(corr_df)):
        for j in range(len(corr_df)):
            val = corr_df.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(val) > 0.5 else "black")

    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_title(f"Skeleton Features × {'Pitch Speed' if mode == 'pitching' else 'Exit Velocity'} Correlation")
    plt.tight_layout()

    path = output_dir / f"correlation_{mode}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Correlation matrix: {path}")

    # Save correlation CSV
    csv_path = output_dir / f"correlation_{mode}.csv"
    corr_df.to_csv(str(csv_path))
    print(f"  Correlation CSV: {csv_path}")


def plot_scatter(df, mode, output_dir):
    """Plot scatter plots of key skeleton features vs speed."""
    speed_key = "pitch_speed_mph" if mode == "pitching" else "exit_velocity_mph"
    speed_label = "Pitch Speed (mph)" if mode == "pitching" else "Exit Velocity (mph)"

    if speed_key not in df.columns or df[speed_key].isna().all():
        print("  No speed data available for scatter plots")
        return

    feature_pairs = [
        ("peak_trunk_velocity", "Peak Trunk Angular Velocity (deg/s)"),
        ("peak_elbow_velocity", "Peak Elbow Angular Velocity (deg/s)"),
        ("peak_shoulder_abduction", "Peak Shoulder Abduction (deg)"),
        ("trunk_rotation_range", "Trunk Rotation Range (deg)"),
    ]

    available = [(col, label) for col, label in feature_pairs if col in df.columns]
    if not available:
        print("  No feature columns available for scatter plots")
        return

    n_plots = len(available)
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    for ax, (col, label) in zip(axes, available):
        valid = df[[speed_key, col]].dropna()
        if len(valid) < 3:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
            ax.set_title(label)
            continue

        x = valid[col].values
        y = valid[speed_key].values
        ax.scatter(x, y, alpha=0.7, edgecolors="black", linewidths=0.5, s=60)

        # Trend line
        slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 50)
        ax.plot(x_line, slope * x_line + intercept, "r--", alpha=0.7)

        ax.set_xlabel(label)
        ax.set_ylabel(speed_label)
        ax.set_title(f"r={r_val:.3f}, p={p_val:.3f}")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Skeleton Features vs {speed_label} — Driveline OBP", fontsize=13)
    plt.tight_layout()

    path = output_dir / f"scatter_{mode}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scatter plots: {path}")


def plot_lead_leg_block_profile(df, mode, output_dir):
    """Scatter plots: Lead Leg Block metrics vs Arm Speed metrics.

    Tests the biomechanical hypothesis: stronger leg block → faster arm action.
    """
    # LLB features (X axis) vs arm speed features (Y axis)
    scatter_pairs = [
        ("llb_knee_extension_range", "Knee Extension After Strike (\u00b0)",
         "peak_elbow_velocity", "Peak Elbow Angular Velocity (deg/s)"),
        ("llb_ankle_velocity_delta", "Ankle Braking at Foot Strike (mm/s)",
         "peak_elbow_velocity", "Peak Elbow Angular Velocity (deg/s)"),
        ("llb_knee_extension_range", "Knee Extension After Strike (\u00b0)",
         "peak_shoulder_velocity", "Peak Shoulder Angular Velocity (deg/s)"),
    ]

    available = [(xc, xl, yc, yl) for xc, xl, yc, yl in scatter_pairs
                 if xc in df.columns and yc in df.columns
                 and df[[xc, yc]].dropna().shape[0] >= 5]
    if not available:
        print("  Not enough data for LLB vs arm speed scatter plots")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (xc, xl, yc, yl) in zip(axes, available):
        valid = df[[xc, yc]].dropna()
        x, y = valid[xc].values, valid[yc].values

        ax.scatter(x, y, alpha=0.7, edgecolors="black", linewidths=0.5, s=70,
                   color="#2980b9")

        # Trend line + correlation
        slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 50)
        ax.plot(x_line, slope * x_line + intercept, "r--", alpha=0.7, linewidth=2)

        ax.set_xlabel(xl, fontsize=12, fontweight="bold")
        ax.set_ylabel(yl, fontsize=12, fontweight="bold")
        sig = "*" if p_val < 0.05 else ""
        ax.set_title(f"r = {r_val:+.3f}{sig}  (n={len(valid)})",
                     fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Lead Leg Block \u2192 Arm Speed\n"
                 "Does a stronger leg block produce faster arm action?",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    path = output_dir / f"llb_profile_{mode}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  LLB profile: {path}")


def main():
    parser = argparse.ArgumentParser(description="Skeleton features × Statcast correlation")
    parser.add_argument("--mode", choices=["pitching", "hitting", "both"], default="pitching")
    parser.add_argument("--download", type=int, default=40,
                        help="Number of additional C3D files to download for correlation")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    modes = ["pitching", "hitting"] if args.mode == "both" else [args.mode]

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  Correlation Analysis: {mode.upper()}")
        print(f"{'='*60}")

        # Download additional samples
        if args.download > 0:
            download_additional_samples(mode, args.download)

        # Process all C3D files
        df = process_multiple_files(mode)

        if df.empty:
            print(f"  No data available for {mode}")
            continue

        speed_key = "pitch_speed_mph" if mode == "pitching" else "exit_velocity_mph"
        print(f"\n  Samples: {len(df)}")
        if speed_key in df.columns:
            valid_speed = df[speed_key].dropna()
            if len(valid_speed) > 0:
                print(f"  Speed range: {valid_speed.min():.1f} - {valid_speed.max():.1f} mph")

        # Save features
        csv_path = OUTPUT_DIR / f"features_{mode}.csv"
        df.to_csv(str(csv_path), index=False)
        print(f"  Features CSV: {csv_path}")

        # Correlation analysis
        plot_correlation_matrix(df, mode, OUTPUT_DIR)
        plot_scatter(df, mode, OUTPUT_DIR)
        plot_lead_leg_block_profile(df, mode, OUTPUT_DIR)

    print("\nDone!")


if __name__ == "__main__":
    main()
