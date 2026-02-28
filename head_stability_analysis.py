"""Head Stability Analysis — the core of good pitching mechanics.

Hypothesis: Braking -> Head Stabilizes -> Better body mechanics
Pitch speed is IGNORED entirely.

Metrics:
- head_braking_ratio: fraction of head motion AFTER foot strike (lower = better)
- head_forward_disp_post_norm: post-FS head displacement / height (lower = better)

Usage:
    python head_stability_analysis.py

Output:
    data/output/head_stability_analysis.png
    Console: correlation summary
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DIR = Path("data/output")


def load_and_enrich(csv_path):
    """Load features and compute derived head stability metrics."""
    df = pd.read_csv(csv_path)

    height_m = df["height_in"] * 0.0254

    # Head braking ratio: what fraction of head motion happens AFTER foot strike
    # Lower = head stopped well after brake = good
    valid_disp = df["llb_head_forward_disp"] > 0.05  # filter tiny values
    df["head_braking_ratio"] = np.nan
    df.loc[valid_disp, "head_braking_ratio"] = (
        df.loc[valid_disp, "llb_head_forward_disp_post"]
        / df.loc[valid_disp, "llb_head_forward_disp"]
    )

    # Post-FS head displacement normalized by height (lower = more stable)
    df["head_post_norm"] = df["llb_head_forward_disp_post"] / height_m

    # Head pre-FS displacement (how much head moves BEFORE foot strike)
    df["head_pre_disp"] = (
        df["llb_head_forward_disp"] - df["llb_head_forward_disp_post"]
    )

    # Head stability score (inverted: higher = MORE stable)
    # = 1 - braking_ratio, clipped to [0, 1]
    df["head_stability_score"] = (1 - df["head_braking_ratio"]).clip(0, 1)

    return df


def analyze_braking_to_stability(df):
    """Part 1: Does braking stabilize the head?"""
    print("=" * 70)
    print("PART 1: BRAKING -> HEAD STABILITY")
    print("head_braking_ratio = post_disp / total_disp (lower = more stable)")
    print("head_stability_score = 1 - ratio (higher = more stable)")
    print("=" * 70)

    braking_features = [
        ("llb_ankle_braking_decel", "Ankle Brake Decel"),
        ("llb_knee_forward_decel", "Knee Forward Decel"),
        ("llb_pelvis_decel", "Pelvis Decel"),
    ]

    stability_features = [
        ("head_braking_ratio", "Head Braking Ratio (lower=stable)"),
        ("head_post_norm", "Head Post-FS Disp / Height (lower=stable)"),
        ("head_stability_score", "Head Stability Score (higher=stable)"),
    ]

    results = []
    for bf, bf_label in braking_features:
        for sf, sf_label in stability_features:
            valid = df[[bf, sf]].dropna()
            if len(valid) < 10:
                continue
            r, p = stats.pearsonr(valid[bf], valid[sf])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {bf_label:>25s} x {sf_label:<45s} r={r:+.3f} p={p:.4f} {sig}")
            results.append({"brake": bf, "stability": sf, "r": r, "p": p})

    return results


def analyze_stability_benefits(df):
    """Part 2: What does head stability bring? (NOT pitch speed)"""
    print("\n" + "=" * 70)
    print("PART 2: HEAD STABILITY -> BODY MECHANICS BENEFITS")
    print("What improves when the head is stable?")
    print("=" * 70)

    stability_metrics = [
        ("head_stability_score", "Head Stability Score"),
        ("head_braking_ratio", "Head Braking Ratio (inv)"),
    ]

    # Benefits to check (body mechanics only, NO pitch speed)
    benefit_features = {
        "Trunk Transfer": [
            ("llb_trunk_rot_vel_at_strike", "Trunk Rot Vel at FS"),
            ("peak_trunk_velocity", "Peak Trunk Velocity"),
            ("trunk_rotation_range", "Trunk Rotation Range"),
            ("llb_time_fs_to_peak_trunk_vel", "Time FS->Peak Trunk (s)"),
        ],
        "Smoothness": [
            ("elbow_rms_jerk", "Elbow RMS Jerk"),
            ("trunk_rms_jerk", "Trunk RMS Jerk"),
            ("knee_pos_rms_jerk", "Knee Positional Jerk"),
            ("kinematic_seq_score", "Kinematic Seq Score"),
        ],
        "Timing": [
            ("seq_gap_pelvis_to_trunk", "Pelvis->Trunk Gap"),
            ("seq_gap_trunk_to_shoulder", "Trunk->Shoulder Gap"),
            ("seq_gap_shoulder_to_elbow", "Shoulder->Elbow Gap"),
            ("seq_total_duration", "Sequence Total Duration"),
        ],
        "Whip / Distal Speed": [
            ("whip_finger_elbow", "Whip Ratio (Finger/Elbow)"),
            ("whip_delay_elbow_to_finger", "Whip Delay Elbow->Finger"),
            ("peak_wrist_linear_speed", "Wrist Linear Speed"),
            ("peak_finger_linear_speed", "Finger Linear Speed"),
        ],
        "Arm Coordination": [
            ("peak_elbow_velocity", "Elbow Angular Velocity"),
            ("peak_shoulder_velocity", "Shoulder Angular Velocity"),
            ("elbow_rom", "Elbow ROM"),
        ],
        "Lower Body": [
            ("llb_knee_ext_peak_velocity", "Knee Extension Velocity"),
            ("llb_knee_extension_range", "Knee Extension Range"),
            ("llb_stride_length", "Stride Length"),
        ],
    }

    significant_benefits = []

    for category, features in benefit_features.items():
        print(f"\n  --- {category} ---")
        for feat, feat_label in features:
            if feat not in df.columns:
                continue
            # Use head_stability_score (higher = more stable)
            valid = df[["head_stability_score", feat]].dropna()
            if len(valid) < 10:
                continue
            r, p = stats.pearsonr(valid["head_stability_score"], valid[feat])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if abs(r) >= 0.20 or p < 0.05:
                direction = "UP" if r > 0 else "DOWN"
                print(f"    Head Stable -> {feat_label:<35s} {direction:>4s}  "
                      f"r={r:+.3f} p={p:.4f} {sig}")
                if p < 0.05:
                    significant_benefits.append({
                        "category": category,
                        "feature": feat,
                        "label": feat_label,
                        "r": r,
                        "p": p,
                        "direction": direction,
                    })

    return significant_benefits


def plot_head_stability_dashboard(df, braking_results, benefits, output_path):
    """Create a summary dashboard for head stability."""
    fig = plt.figure(figsize=(16, 14))

    # Layout: 2 rows
    # Top row: 3 scatter plots (braking -> stability)
    # Bottom row: benefits summary bar chart + scatter of best benefit

    # --- Top row: Braking -> Head Stability ---
    braking_pairs = [
        ("llb_ankle_braking_decel", "Ankle Brake Decel"),
        ("llb_knee_forward_decel", "Knee Forward Decel"),
        ("llb_pelvis_decel", "Pelvis Decel"),
    ]

    for i, (feat, label) in enumerate(braking_pairs):
        ax = fig.add_subplot(3, 3, i + 1)
        valid = df[[feat, "head_stability_score"]].dropna()
        if len(valid) < 5:
            continue
        r, p = stats.pearsonr(valid[feat], valid["head_stability_score"])
        ax.scatter(valid[feat], valid["head_stability_score"],
                   alpha=0.6, c="#3498db", edgecolors="white", s=50)

        # Regression line
        z = np.polyfit(valid[feat], valid["head_stability_score"], 1)
        xline = np.linspace(valid[feat].min(), valid[feat].max(), 50)
        ax.plot(xline, np.polyval(z, xline), "r--", alpha=0.7, linewidth=2)

        sig = "*" if p < 0.05 else ""
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Head Stability Score", fontsize=9)
        ax.set_title(f"r={r:+.3f} (p={p:.3f}){sig}", fontsize=10)

    # --- Middle: head_braking_ratio distribution ---
    ax_hist = fig.add_subplot(3, 3, 4)
    valid_ratio = df["head_braking_ratio"].dropna()
    valid_ratio = valid_ratio[(valid_ratio > -1) & (valid_ratio < 2)]
    ax_hist.hist(valid_ratio, bins=20, color="#2ecc71", edgecolor="white", alpha=0.8)
    ax_hist.axvline(valid_ratio.median(), color="red", linestyle="--",
                    label=f"Median: {valid_ratio.median():.2f}")
    ax_hist.set_xlabel("Head Braking Ratio\n(post_disp / total_disp)", fontsize=9)
    ax_hist.set_ylabel("Count", fontsize=9)
    ax_hist.set_title("Head Braking Ratio Distribution", fontsize=10)
    ax_hist.legend(fontsize=8)

    # --- Middle: Top benefits bar chart ---
    ax_bar = fig.add_subplot(3, 3, 5)
    if benefits:
        sorted_benefits = sorted(benefits, key=lambda x: abs(x["r"]), reverse=True)[:8]
        labels_bar = [b["label"] for b in sorted_benefits]
        values = [b["r"] for b in sorted_benefits]
        colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]
        y_pos = range(len(labels_bar))
        ax_bar.barh(y_pos, values, color=colors, edgecolor="white", alpha=0.8)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(labels_bar, fontsize=7)
        ax_bar.set_xlabel("Correlation with Head Stability", fontsize=9)
        ax_bar.set_title("Benefits of Head Stability\n(p < 0.05 only)", fontsize=10)
        ax_bar.axvline(0, color="black", linewidth=0.5)
        ax_bar.invert_yaxis()

    # --- Middle: head stability vs best trunk feature ---
    ax_trunk = fig.add_subplot(3, 3, 6)
    trunk_feat = "llb_trunk_rot_vel_at_strike"
    if trunk_feat in df.columns:
        valid = df[["head_stability_score", trunk_feat]].dropna()
        if len(valid) >= 5:
            r, p = stats.pearsonr(valid["head_stability_score"], valid[trunk_feat])
            ax_trunk.scatter(valid["head_stability_score"], valid[trunk_feat],
                             alpha=0.6, c="#e74c3c", edgecolors="white", s=50)
            z = np.polyfit(valid["head_stability_score"], valid[trunk_feat], 1)
            xline = np.linspace(valid["head_stability_score"].min(),
                                valid["head_stability_score"].max(), 50)
            ax_trunk.plot(xline, np.polyval(z, xline), "b--", alpha=0.7, linewidth=2)
            sig = "*" if p < 0.05 else ""
            ax_trunk.set_xlabel("Head Stability Score", fontsize=9)
            ax_trunk.set_ylabel("Trunk Rot Vel at FS", fontsize=9)
            ax_trunk.set_title(f"Head Stable -> Trunk Rotation\nr={r:+.3f} (p={p:.3f}){sig}",
                               fontsize=10)

    # --- Bottom row: key scatter plots for benefits ---
    benefit_scatters = [
        ("elbow_rms_jerk", "Elbow RMS Jerk"),
        ("knee_pos_rms_jerk", "Knee Positional Jerk"),
        ("peak_trunk_velocity", "Peak Trunk Velocity"),
    ]

    for i, (feat, label) in enumerate(benefit_scatters):
        ax = fig.add_subplot(3, 3, 7 + i)
        if feat not in df.columns:
            continue
        valid = df[["head_stability_score", feat]].dropna()
        if len(valid) < 5:
            continue
        r, p = stats.pearsonr(valid["head_stability_score"], valid[feat])
        ax.scatter(valid["head_stability_score"], valid[feat],
                   alpha=0.6, c="#9b59b6", edgecolors="white", s=50)
        z = np.polyfit(valid["head_stability_score"], valid[feat], 1)
        xline = np.linspace(valid["head_stability_score"].min(),
                            valid["head_stability_score"].max(), 50)
        ax.plot(xline, np.polyval(z, xline), "r--", alpha=0.7, linewidth=2)
        sig = "*" if p < 0.05 else ""
        ax.set_xlabel("Head Stability Score", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(f"r={r:+.3f} (p={p:.3f}){sig}", fontsize=10)

    fig.suptitle("Head Stability: The Core of Good Pitching Mechanics\n"
                 "Driveline OBP (N=49, pitch speed NOT used)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDashboard saved: {output_path}")


def print_narrative(braking_results, benefits):
    """Print the story."""
    print("\n" + "=" * 70)
    print("NARRATIVE: HEAD STABILITY IN PITCHING MECHANICS")
    print("=" * 70)

    print("\n[1] BRAKING -> HEAD STABILITY")
    # Check braking -> head_stability_score
    sig_braking = [r for r in braking_results
                   if r["stability"] == "head_stability_score" and r["p"] < 0.10]
    if sig_braking:
        for r in sig_braking:
            brake_name = r["brake"].replace("llb_", "")
            direction = "stabilizes" if r["r"] > 0 else "destabilizes"
            print(f"  - {brake_name} {direction} the head (r={r['r']:+.3f}, p={r['p']:.3f})")
    else:
        print("  - Direct braking -> stability link is weak")
        print("  - May need to control for overall effort level")

    print("\n[2] HEAD STABILITY -> BENEFITS")
    if benefits:
        for b in sorted(benefits, key=lambda x: abs(x["r"]), reverse=True):
            direction = "increases" if b["r"] > 0 else "decreases"
            print(f"  - Head stable -> {b['label']} {direction} "
                  f"(r={b['r']:+.3f}, p={b['p']:.3f})")
    else:
        print("  - No significant benefits found (p < 0.05)")

    print("\n[3] INTERPRETATION")
    print("  'Head stability' = head stops moving forward after foot strike")
    print("  This is NOT about keeping the head perfectly still during windup")
    print("  It's about the head DECELERATING after the lead foot brakes")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_enrich(OUTPUT_DIR / "features_pitching.csv")
    print(f"Loaded {len(df)} samples")
    print(f"  head_stability_score: mean={df['head_stability_score'].mean():.3f}, "
          f"std={df['head_stability_score'].std():.3f}")

    braking_results = analyze_braking_to_stability(df)
    benefits = analyze_stability_benefits(df)

    plot_head_stability_dashboard(df, braking_results, benefits,
                                  OUTPUT_DIR / "head_stability_analysis.png")

    print_narrative(braking_results, benefits)


if __name__ == "__main__":
    main()
