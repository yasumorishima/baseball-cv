"""Step 3: Joint angle & angular velocity extraction from skeleton data.

Computes biomechanical features from C3D motion capture data:
- Joint angles: elbow flexion, shoulder external rotation, trunk rotation, knee flexion
- Angular velocities (frame-to-frame differences)
- Kinematic sequence time-series plots

Usage:
    python skeleton_analysis.py                             # Default: pitching sample
    python skeleton_analysis.py --input data/raw/hitting_sample.c3d --mode hitting

Output:
    data/output/joint_angles_{mode}.csv
    data/output/kinematic_sequence_{mode}.png
    data/output/angular_velocity_{mode}.png
"""

import argparse
from pathlib import Path

import ezc3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_c3d(filepath):
    """Load C3D file and return marker dict and frame rate."""
    c = ezc3d.c3d(str(filepath))
    labels = c["parameters"]["POINT"]["LABELS"]["value"]
    points = c["data"]["points"]  # (4, n_markers, n_frames)
    rate = c["parameters"]["POINT"]["RATE"]["value"][0]
    marker_dict = {}
    for i, label in enumerate(labels):
        marker_dict[label] = points[:3, i, :]  # (3, n_frames) — x, y, z
    return marker_dict, rate, points.shape[2]


def angle_between_vectors(v1, v2):
    """Compute angle (degrees) between two 3D vectors per frame."""
    # v1, v2: (3, n_frames)
    dot = np.sum(v1 * v2, axis=0)
    norm1 = np.linalg.norm(v1, axis=0)
    norm2 = np.linalg.norm(v2, axis=0)
    # Avoid division by zero
    denom = norm1 * norm2
    denom[denom == 0] = 1e-10
    cos_angle = np.clip(dot / denom, -1, 1)
    return np.degrees(np.arccos(cos_angle))


def compute_elbow_flexion(markers, side="R"):
    """Elbow flexion angle: angle at elbow between upper arm and forearm."""
    sho = markers.get(f"{side}SHO")
    elb = markers.get(f"{side}ELB")
    wra = markers.get(f"{side}WRA")
    if sho is None or elb is None or wra is None:
        return None
    upper_arm = sho - elb  # shoulder to elbow
    forearm = wra - elb    # wrist to elbow
    return angle_between_vectors(upper_arm, forearm)


def compute_shoulder_abduction(markers, side="R"):
    """Shoulder abduction: angle between upper arm and trunk midline."""
    sho = markers.get(f"{side}SHO")
    elb = markers.get(f"{side}ELB")
    c7 = markers.get("C7")
    strn = markers.get("STRN")
    if sho is None or elb is None or c7 is None or strn is None:
        return None
    upper_arm = elb - sho
    trunk = strn - c7
    return angle_between_vectors(upper_arm, trunk)


def compute_trunk_rotation(markers):
    """Trunk rotation: angle between shoulder line and pelvis line in horizontal plane."""
    lsho = markers.get("LSHO")
    rsho = markers.get("RSHO")
    lasi = markers.get("LASI")
    rasi = markers.get("RASI")
    if lsho is None or rsho is None or lasi is None or rasi is None:
        return None
    shoulder_line = rsho - lsho  # (3, n_frames)
    pelvis_line = rasi - lasi
    # Project onto horizontal plane (XY)
    shoulder_xy = shoulder_line[:2, :]
    pelvis_xy = pelvis_line[:2, :]
    # Angle between projections
    dot = np.sum(shoulder_xy * pelvis_xy, axis=0)
    norm1 = np.linalg.norm(shoulder_xy, axis=0)
    norm2 = np.linalg.norm(pelvis_xy, axis=0)
    denom = norm1 * norm2
    denom[denom == 0] = 1e-10
    cos_angle = np.clip(dot / denom, -1, 1)
    return np.degrees(np.arccos(cos_angle))


def compute_knee_flexion(markers, side="R"):
    """Knee flexion: angle at knee between thigh and shank."""
    hip_marker = f"{side}ASI"  # Use ASIS as hip proxy
    kne = markers.get(f"{side}KNE")
    ank = markers.get(f"{side}ANK")
    hip = markers.get(hip_marker)
    if hip is None or kne is None or ank is None:
        return None
    thigh = hip - kne
    shank = ank - kne
    return angle_between_vectors(thigh, shank)


def compute_angular_velocity(angles, rate):
    """Compute angular velocity (deg/s) from angle time series."""
    dt = 1.0 / rate
    vel = np.gradient(angles) / dt
    return vel


def analyze(filepath, mode="pitching"):
    """Run full joint angle analysis on a C3D file."""
    print(f"\nAnalyzing {mode}: {filepath}")
    markers, rate, n_frames = load_c3d(filepath)
    print(f"  Markers: {len(markers)}, Frames: {n_frames}, Rate: {rate} Hz")

    time = np.arange(n_frames) / rate

    # Determine throwing/hitting arm
    # Pitching: right-handed default; Hitting: use both sides
    sides = ["R", "L"] if mode == "hitting" else ["R"]

    results = {"frame": np.arange(n_frames), "time_s": time}

    for side in sides:
        side_label = "right" if side == "R" else "left"

        elbow = compute_elbow_flexion(markers, side)
        if elbow is not None:
            results[f"elbow_flexion_{side_label}"] = elbow

        shoulder = compute_shoulder_abduction(markers, side)
        if shoulder is not None:
            results[f"shoulder_abduction_{side_label}"] = shoulder

        knee = compute_knee_flexion(markers, side)
        if knee is not None:
            results[f"knee_flexion_{side_label}"] = knee

    trunk = compute_trunk_rotation(markers)
    if trunk is not None:
        results["trunk_rotation"] = trunk

    df = pd.DataFrame(results)
    return df, rate, time


def plot_kinematic_sequence(df, time, mode, output_dir):
    """Plot kinematic sequence: joint angles over time."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Kinematic Sequence — {mode.title()}", fontsize=14)

    angle_cols = [c for c in df.columns if c not in ("frame", "time_s")]

    # Elbow flexion
    ax = axes[0, 0]
    for col in [c for c in angle_cols if "elbow" in c]:
        ax.plot(time, df[col], label=col.replace("_", " ").title())
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Elbow Flexion")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Shoulder abduction
    ax = axes[0, 1]
    for col in [c for c in angle_cols if "shoulder" in c]:
        ax.plot(time, df[col], label=col.replace("_", " ").title())
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Shoulder Abduction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Trunk rotation
    ax = axes[1, 0]
    if "trunk_rotation" in df.columns:
        ax.plot(time, df["trunk_rotation"], label="Trunk Rotation", color="#e74c3c")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Trunk Rotation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Knee flexion
    ax = axes[1, 1]
    for col in [c for c in angle_cols if "knee" in c]:
        ax.plot(time, df[col], label=col.replace("_", " ").title())
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Knee Flexion")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / f"kinematic_sequence_{mode}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kinematic sequence plot: {path}")


def plot_angular_velocity(df, rate, mode, output_dir):
    """Plot angular velocities for key joints."""
    time = df["time_s"].values
    angle_cols = [c for c in df.columns if c not in ("frame", "time_s")]

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in angle_cols:
        vel = compute_angular_velocity(df[col].values, rate)
        ax.plot(time, vel, label=col.replace("_", " ").title(), alpha=0.8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular Velocity (deg/s)")
    ax.set_title(f"Angular Velocities — {mode.title()}")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    path = output_dir / f"angular_velocity_{mode}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Angular velocity plot: {path}")


def main():
    parser = argparse.ArgumentParser(description="Joint angle & angular velocity extraction")
    parser.add_argument("--input", type=str, default=None, help="C3D file path")
    parser.add_argument("--mode", choices=["pitching", "hitting"], default="pitching")
    args = parser.parse_args()

    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        filepath = args.input
    else:
        filepath = f"data/raw/{args.mode}_sample.c3d"

    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        return

    df, rate, time = analyze(filepath, args.mode)

    # Save CSV
    csv_path = output_dir / f"joint_angles_{args.mode}.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"  Joint angles CSV: {csv_path} ({len(df)} rows)")

    # Plots
    plot_kinematic_sequence(df, time, args.mode, output_dir)
    plot_angular_velocity(df, rate, args.mode, output_dir)

    # Summary statistics
    angle_cols = [c for c in df.columns if c not in ("frame", "time_s")]
    print(f"\n  Summary ({args.mode}):")
    for col in angle_cols:
        vals = df[col].dropna()
        print(f"    {col}: min={vals.min():.1f}°, max={vals.max():.1f}°, "
              f"range={vals.max()-vals.min():.1f}°")

    print("\nDone!")


if __name__ == "__main__":
    main()
