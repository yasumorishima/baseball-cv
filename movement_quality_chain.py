"""Movement Quality Chain Analysis — body mechanics correlations.

Analyzes the chain: braking → head stability → smoothness → whip → release
WITHOUT using pitch speed as the target. Defines "good mechanics" by
how body segments transfer energy through the kinetic chain.

Usage:
    python movement_quality_chain.py

Output:
    data/output/movement_quality_chain.png
    data/output/movement_quality_matrix.png
    Console: correlation chain summary
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DIR = Path("data/output")


def load_features():
    """Load pitching features CSV."""
    df = pd.read_csv(OUTPUT_DIR / "features_pitching.csv")
    return df


def chain_analysis(df):
    """Analyze the movement quality chain.

    Chain hypothesis:
    1. Braking (foot strike → ground reaction) →
    2. Head stability (head doesn't fly forward) →
    3. Smoothness (low jerk, clean sequence) →
    4. Whip (distal segment acceleration) →
    5. Release quality (finger decel = grip)

    We check correlations BETWEEN these categories, not against pitch speed.
    """
    # Define feature groups
    groups = {
        "Braking": [
            "llb_ankle_braking_decel",
            "llb_knee_forward_decel",
            "llb_pelvis_decel",
        ],
        "Head Stability": [
            "llb_head_forward_disp",       # less = more stable
            "llb_head_forward_disp_post",   # less = more stable
        ],
        "Trunk Transfer": [
            "llb_trunk_rot_vel_at_strike",
            "llb_time_fs_to_peak_trunk_vel",
            "peak_trunk_velocity",
            "trunk_rotation_range",
        ],
        "Smoothness": [
            "elbow_rms_jerk",
            "trunk_rms_jerk",
            "knee_pos_rms_jerk",
            "kinematic_seq_score",
        ],
        "Whip": [
            "peak_wrist_linear_speed",
            "peak_finger_linear_speed",
            "whip_finger_elbow",
            "whip_delay_elbow_to_finger",
        ],
        "Arm Speed": [
            "peak_elbow_velocity",
            "peak_elbow_linear_speed",
        ],
    }

    print("=" * 70)
    print("MOVEMENT QUALITY CHAIN ANALYSIS")
    print("Correlations between body mechanics features (NOT vs pitch speed)")
    print("=" * 70)

    # Chain links to test
    chain_links = [
        ("Braking", "Head Stability"),
        ("Braking", "Trunk Transfer"),
        ("Head Stability", "Trunk Transfer"),
        ("Head Stability", "Smoothness"),
        ("Trunk Transfer", "Smoothness"),
        ("Trunk Transfer", "Whip"),
        ("Smoothness", "Whip"),
        ("Whip", "Arm Speed"),
        ("Braking", "Arm Speed"),
        ("Head Stability", "Arm Speed"),
        ("Smoothness", "Arm Speed"),
    ]

    all_results = []

    for cat_a, cat_b in chain_links:
        print(f"\n--- {cat_a} → {cat_b} ---")
        feats_a = groups[cat_a]
        feats_b = groups[cat_b]

        for fa in feats_a:
            if fa not in df.columns:
                continue
            for fb in feats_b:
                if fb not in df.columns:
                    continue
                valid = df[[fa, fb]].dropna()
                if len(valid) < 10:
                    continue
                r, p = stats.pearsonr(valid[fa], valid[fb])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                if abs(r) >= 0.25:
                    fa_short = fa.replace("llb_", "").replace("peak_", "")
                    fb_short = fb.replace("llb_", "").replace("peak_", "")
                    print(f"  {fa_short:>35s} × {fb_short:<30s}  r={r:+.3f}  p={p:.4f} {sig}")
                    all_results.append({
                        "from_cat": cat_a,
                        "to_cat": cat_b,
                        "feature_a": fa,
                        "feature_b": fb,
                        "r": r,
                        "p": p,
                        "significant": p < 0.05,
                    })

    return all_results


def plot_chain_diagram(results, output_path):
    """Visualize the movement quality chain as a flow diagram."""
    categories = ["Braking", "Head Stability", "Trunk Transfer",
                   "Smoothness", "Whip", "Arm Speed"]

    # Aggregate: average |r| between each pair of categories
    pair_strength = {}
    for res in results:
        key = (res["from_cat"], res["to_cat"])
        if key not in pair_strength:
            pair_strength[key] = []
        pair_strength[key].append(res["r"])

    fig, ax = plt.subplots(figsize=(14, 8))

    # Position categories in a flow
    positions = {
        "Braking": (1, 3),
        "Head Stability": (3, 4),
        "Trunk Transfer": (3, 2),
        "Smoothness": (5, 4),
        "Whip": (5, 2),
        "Arm Speed": (7, 3),
    }

    # Draw category boxes
    for cat, (x, y) in positions.items():
        ax.add_patch(plt.Rectangle((x - 0.8, y - 0.35), 1.6, 0.7,
                                    facecolor="#ecf0f1", edgecolor="#2c3e50",
                                    linewidth=2, zorder=2))
        ax.text(x, y, cat, ha="center", va="center", fontsize=10,
                fontweight="bold", zorder=3)

    # Draw connections
    for (cat_a, cat_b), r_vals in pair_strength.items():
        mean_r = np.mean(r_vals)
        max_abs_r = max(abs(r) for r in r_vals)
        n_sig = sum(1 for res in results
                    if res["from_cat"] == cat_a and res["to_cat"] == cat_b
                    and res["significant"])

        x1, y1 = positions[cat_a]
        x2, y2 = positions[cat_b]

        # Arrow thickness and color based on strength
        lw = max(1, abs(mean_r) * 8)
        color = "#e74c3c" if mean_r > 0 else "#3498db"
        alpha = min(1.0, abs(mean_r) + 0.3)

        label = f"r={mean_r:+.2f}"
        if n_sig > 0:
            label += f" ({n_sig}*)"

        ax.annotate("", xy=(x2 - 0.8, y2), xytext=(x1 + 0.8, y1),
                     arrowprops=dict(arrowstyle="->", lw=lw, color=color,
                                     alpha=alpha))

        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.15
        ax.text(mx, my, label, ha="center", va="center", fontsize=8,
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.8))

    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Movement Quality Chain — Driveline OBP\n"
                 "Red = positive correlation, Blue = negative\n"
                 "Arrow thickness = correlation strength",
                 fontsize=12, pad=20)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nChain diagram saved: {output_path}")


def plot_correlation_matrix(df, output_path):
    """Heatmap of selected features (between-category only)."""
    features = [
        # Braking
        "llb_ankle_braking_decel",
        "llb_knee_forward_decel",
        "llb_pelvis_decel",
        # Head stability
        "llb_head_forward_disp",
        "llb_head_forward_disp_post",
        # Trunk transfer
        "peak_trunk_velocity",
        "trunk_rotation_range",
        "llb_time_fs_to_peak_trunk_vel",
        # Smoothness
        "elbow_rms_jerk",
        "trunk_rms_jerk",
        "kinematic_seq_score",
        # Whip
        "peak_wrist_linear_speed",
        "peak_finger_linear_speed",
        "whip_finger_elbow",
        # Arm speed
        "peak_elbow_velocity",
        "peak_elbow_linear_speed",
    ]

    available = [f for f in features if f in df.columns]
    corr = df[available].corr()

    # Short labels
    label_map = {
        "llb_ankle_braking_decel": "Ankle Brake",
        "llb_knee_forward_decel": "Knee Decel",
        "llb_pelvis_decel": "Pelvis Decel",
        "llb_head_forward_disp": "Head Disp",
        "llb_head_forward_disp_post": "Head Disp Post",
        "peak_trunk_velocity": "Trunk Vel",
        "trunk_rotation_range": "Trunk ROM",
        "llb_time_fs_to_peak_trunk_vel": "FS→TrunkPeak",
        "elbow_rms_jerk": "Elbow Jerk",
        "trunk_rms_jerk": "Trunk Jerk",
        "kinematic_seq_score": "KinSeq Score",
        "peak_wrist_linear_speed": "Wrist Speed",
        "peak_finger_linear_speed": "Finger Speed",
        "whip_finger_elbow": "Whip Ratio",
        "peak_elbow_velocity": "Elbow AngVel",
        "peak_elbow_linear_speed": "Elbow LinSpeed",
    }

    short_labels = [label_map.get(f, f) for f in available]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(short_labels)))
    ax.set_yticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)

    # Annotate significant cells
    n = len(available)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = corr.values[i, j]
            if abs(val) >= 0.3:
                color = "white" if abs(val) >= 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

    # Category separators
    sep_positions = [3, 5, 8, 11, 14]  # after Braking, Head, Trunk, Smooth, Whip
    for pos in sep_positions:
        ax.axhline(pos - 0.5, color="black", linewidth=2)
        ax.axvline(pos - 0.5, color="black", linewidth=2)

    # Category labels on right side
    cat_labels = [
        (1.5, "Braking"),
        (4, "Head"),
        (6.5, "Trunk"),
        (9.5, "Smooth"),
        (12.5, "Whip"),
        (14.5, "Arm"),
    ]
    for pos, label in cat_labels:
        ax.text(n + 0.3, pos, label, ha="left", va="center", fontsize=9,
                fontweight="bold", color="#2c3e50")

    plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    ax.set_title("Movement Quality Feature Correlations — Driveline OBP\n"
                 "Values shown for |r| ≥ 0.30", fontsize=12)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Correlation matrix saved: {output_path}")


def summarize_chain(results):
    """Print the strongest chain links."""
    print("\n" + "=" * 70)
    print("STRONGEST CHAIN LINKS (|r| >= 0.30, p < 0.05)")
    print("=" * 70)

    sig_strong = [r for r in results if r["significant"] and abs(r["r"]) >= 0.30]
    sig_strong.sort(key=lambda x: abs(x["r"]), reverse=True)

    for res in sig_strong:
        fa = res["feature_a"].replace("llb_", "").replace("peak_", "")
        fb = res["feature_b"].replace("llb_", "").replace("peak_", "")
        direction = "→(+)" if res["r"] > 0 else "→(-)"
        print(f"  {res['from_cat']:>15s} {direction} {res['to_cat']:<15s}  "
              f"{fa} × {fb}  r={res['r']:+.3f}  p={res['p']:.4f}")

    # Summarize the chain narrative
    print("\n" + "=" * 70)
    print("CHAIN NARRATIVE")
    print("=" * 70)

    chain_steps = [
        ("Braking", "Head Stability"),
        ("Braking", "Trunk Transfer"),
        ("Head Stability", "Trunk Transfer"),
        ("Trunk Transfer", "Whip"),
        ("Smoothness", "Whip"),
        ("Whip", "Arm Speed"),
    ]

    for cat_a, cat_b in chain_steps:
        relevant = [r for r in sig_strong
                    if r["from_cat"] == cat_a and r["to_cat"] == cat_b]
        if relevant:
            best = max(relevant, key=lambda x: abs(x["r"]))
            status = "CONNECTED"
            detail = f"r={best['r']:+.3f}"
        else:
            # Check if there were any results at all (even non-significant)
            any_result = [r for r in results
                          if r["from_cat"] == cat_a and r["to_cat"] == cat_b]
            if any_result:
                best = max(any_result, key=lambda x: abs(x["r"]))
                status = "WEAK/NS"
                detail = f"r={best['r']:+.3f} (p={best['p']:.3f})"
            else:
                status = "NO DATA"
                detail = ""
        print(f"  {cat_a:>15s} → {cat_b:<15s}  [{status}] {detail}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_features()
    print(f"Loaded {len(df)} samples, {len(df.columns)} features")

    results = chain_analysis(df)

    plot_chain_diagram(results, OUTPUT_DIR / "movement_quality_chain.png")
    plot_correlation_matrix(df, OUTPUT_DIR / "movement_quality_matrix.png")

    summarize_chain(results)

    print(f"\nTotal correlations found (|r| >= 0.25): {len(results)}")
    print(f"Significant (p < 0.05): {sum(1 for r in results if r['significant'])}")


if __name__ == "__main__":
    main()
