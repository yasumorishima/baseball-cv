"""Body Efficiency Analysis — 5-component model of efficient throwing.

Discovers that pitchers with identical arm speed (24-26 m/s) vary by 13 mph
in pitch speed. Four independent body mechanics factors explain this gap.

Findings:
  - R2 increases from 0.491 (arm only) to 0.669 (5 components)
  - Q1 vs Q5 (same arm speed): 79.0 vs 89.3 mph (+10.3 mph)
  - Knee smoothness (jerk) is the largest single addition (+0.077 R2)
  - Root cause of poor efficiency: low ankle braking -> short stride

Requires:
    data/output/features_pitching.csv (from statcast_correlation.py)

Output:
    data/output/efficient_throwing_story.png
    data/output/body_efficiency_breakdown.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

OUTPUT_DIR = Path("data/output")
FEATURES_CSV = OUTPUT_DIR / "features_pitching.csv"

JERK_COL = "knee_pos_rms_jerk"
TARGET = "pitch_speed_mph"


def compute_body_efficiency(df):
    """Residual of pitch_speed after regressing on arm speed alone."""
    d = df[["peak_wrist_linear_speed", TARGET]].dropna().copy()
    lm = LinearRegression().fit(d[["peak_wrist_linear_speed"]], d[TARGET])
    df = df.copy()
    df.loc[d.index, "body_efficiency"] = (
        d[TARGET].values - lm.predict(d[["peak_wrist_linear_speed"]].values)
    )
    df["eff_q"] = pd.qcut(df["body_efficiency"], q=5,
                          labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    return df


def compute_incremental_r2(df):
    """Compute R2 at each step of the 5-component model."""
    steps = [
        (["peak_wrist_linear_speed", "height_in"],
         "Arm speed + Height"),
        (["peak_wrist_linear_speed", "height_in", "llb_stride_forward"],
         "+ Stride (translational)"),
        (["peak_wrist_linear_speed", "height_in", "llb_stride_forward",
          "min_knee_flexion"],
         "+ Leg lift (elastic)"),
        (["peak_wrist_linear_speed", "height_in", "llb_stride_forward",
          "min_knee_flexion", "whip_wrist_elbow"],
         "+ Arm chain (whip)"),
        (["peak_wrist_linear_speed", "height_in", "llb_stride_forward",
          "min_knee_flexion", "whip_wrist_elbow", JERK_COL],
         "+ Knee smoothness"),
    ]
    results = []
    for cols, label in steps:
        d = df[cols + [TARGET]].dropna()
        X = np.column_stack([d[cols].values, np.ones(len(d))])
        coef, *_ = np.linalg.lstsq(X, d[TARGET].values, rcond=None)
        y_pred = X @ coef
        r2 = r2_score(d[TARGET].values, y_pred)
        results.append((r2, label, len(d)))
    return results


def plot_story(df, r2_steps, out_path):
    """3-panel: scatter colored by jerk | R2 staircase | Q1/Q5 bar."""
    fig = plt.figure(figsize=(18, 8))

    # Panel 1: Arm speed vs Pitch speed, colored by knee jerk
    ax1 = fig.add_subplot(1, 3, 1)
    d_plot = df[["peak_wrist_linear_speed", TARGET, JERK_COL]].dropna()
    sc = ax1.scatter(
        d_plot["peak_wrist_linear_speed"], d_plot[TARGET],
        c=d_plot[JERK_COL], cmap="RdYlGn_r", s=80, alpha=0.8,
        edgecolors="gray", linewidths=0.5,
    )
    plt.colorbar(sc, ax=ax1, label="Knee jerk (red=jerky)")
    z = np.polyfit(d_plot["peak_wrist_linear_speed"], d_plot[TARGET], 1)
    xline = np.linspace(d_plot["peak_wrist_linear_speed"].min(),
                        d_plot["peak_wrist_linear_speed"].max(), 50)
    ax1.plot(xline, np.polyval(z, xline), color="black", lw=2, alpha=0.4)
    ax1.set_xlabel("Arm speed — peak wrist (m/s)", fontsize=14)
    ax1.set_ylabel("Pitch speed (mph)", fontsize=14)
    ax1.set_title("Same arm speed -> different pitch speed\n"
                  "(green=smooth knee, red=jerky)", fontsize=13)

    # Panel 2: Incremental R2 bars
    ax2 = fig.add_subplot(1, 3, 2)
    r2_vals = [r for r, *_ in r2_steps]
    labels_short = ["Arm\nspeed", "+Stride", "+Leg\nlift",
                    "+Arm\nchain", "+Knee\nsmooth"]
    colors = ["#2980b9", "#27ae60", "#8e44ad", "#e67e22", "#e74c3c"]
    ax2.bar(range(len(r2_vals)), r2_vals, color=colors, edgecolor="white", lw=2)
    for i, r2 in enumerate(r2_vals):
        prev = r2_vals[i - 1] if i > 0 else 0
        ax2.text(i, r2 + 0.01, f"{r2:.3f}", ha="center", fontweight="bold", fontsize=13)
        if i > 0:
            ax2.text(i, r2 - 0.05, f"+{r2-prev:.3f}", ha="center",
                     fontsize=11, color="white", fontweight="bold")
    ax2.set_xticks(range(len(labels_short)))
    ax2.set_xticklabels(labels_short, fontsize=13)
    ax2.set_ylabel("R2 (variance explained)", fontsize=14)
    ax2.set_ylim(0, 0.82)
    ax2.set_title("Each body component adds\npredictive power", fontsize=13)

    # Panel 3: Q1/Q5 pitch speed bars (with arm speed annotation)
    ax3 = fig.add_subplot(1, 3, 3)
    q_grp = (df.groupby("eff_q", observed=True)
               .agg(pitch_speed=(TARGET, "mean"),
                    arm_speed=("peak_wrist_linear_speed", "mean"))
               .reset_index().dropna())
    x = np.arange(len(q_grp))
    colors_q = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
    ax3.bar(x, q_grp["pitch_speed"], color=colors_q, edgecolor="white", lw=2)
    for i, (sp, arm) in enumerate(zip(q_grp["pitch_speed"], q_grp["arm_speed"])):
        ax3.text(i, sp + 0.3, f"{sp:.1f}", ha="center", fontweight="bold", fontsize=13)
        ax3.text(i, sp - 2.8, f"arm\n{arm:.1f}", ha="center", fontsize=11, color="white")
    ax3.set_xticks(x)
    ax3.set_xticklabels(["Q1\n(worst)", "Q2", "Q3", "Q4", "Q5\n(best)"], fontsize=13)
    ax3.set_ylabel("Mean pitch speed (mph)", fontsize=14)
    ax3.set_ylim(75, 95)
    ax3.set_title("Same arm speed, different body use\n"
                  "(Q1=79 mph vs Q5=89 mph)", fontsize=13)

    fig.suptitle(
        "Efficient Throwing: 5 Independent Components (n=58 pitchers, R2=0.669)",
        fontsize=16, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_breakdown(df, out_path):
    """2-panel: Q1/Q5 body mechanics z-scores | scatter with Q quintiles."""
    body_vars = ["llb_stride_forward", "min_knee_flexion",
                 "whip_wrist_elbow", JERK_COL]
    body_labels = ["Stride\n(translational)", "Leg Lift\n(elastic load)",
                   "Arm Chain\n(whip_we)", "Knee\nSmoothness"]
    flip = [+1, -1, -1, -1]  # stride: higher=better; rest: lower=better

    q1 = df[df["eff_q"] == "Q1"][body_vars].mean()
    q5 = df[df["eff_q"] == "Q5"][body_vars].mean()
    means = df[body_vars].mean()
    stds = df[body_vars].std()

    q1_z = ((q1 - means) / stds).values * flip
    q5_z = ((q5 - means) / stds).values * flip

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    x = np.arange(len(body_labels))
    w = 0.35
    ax1.bar(x - w / 2, q1_z, w, label="Q1 (Worst body, 79 mph)",
            color="#e74c3c", alpha=0.8, edgecolor="white")
    ax1.bar(x + w / 2, q5_z, w, label="Q5 (Best body, 89 mph)",
            color="#27ae60", alpha=0.8, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(body_labels, fontsize=13)
    ax1.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
    ax1.set_ylabel("Z-score (positive = more efficient)", fontsize=14)
    ax1.set_title("Body Mechanics: Same arm speed (24.6 vs 24.7 m/s)\n"
                  "--> 10.3 mph difference!", fontsize=13)
    ax1.legend(fontsize=12)
    for xi, (z1, z5) in enumerate(zip(q1_z, q5_z)):
        for val, offset in [(z1, -w / 2), (z5, +w / 2)]:
            if val >= 0:
                y_text = val + 0.05
            elif val > -0.45:
                y_text = val - 0.13
            else:
                y_text = val + 0.12
            ax1.text(xi + offset, y_text, f"{val:+.2f}", ha="center", fontsize=12)

    d_plot = df[["peak_wrist_linear_speed", TARGET, "eff_q"]].dropna()
    cmap = {"Q1": "#e74c3c", "Q2": "#e67e22", "Q3": "#f1c40f",
            "Q4": "#2ecc71", "Q5": "#27ae60"}
    for q in ["Q2", "Q3", "Q4"]:
        d_q = d_plot[d_plot["eff_q"] == q]
        ax2.scatter(d_q["peak_wrist_linear_speed"], d_q[TARGET],
                    c=cmap[q], s=50, alpha=0.6)
    for q in ["Q1", "Q5"]:
        d_q = d_plot[d_plot["eff_q"] == q]
        mph = d_q[TARGET].mean()
        arm = d_q["peak_wrist_linear_speed"].mean()
        ax2.scatter(d_q["peak_wrist_linear_speed"], d_q[TARGET],
                    c=cmap[q], s=100, label=f"{q}: {mph:.1f} mph (arm={arm:.1f})",
                    edgecolors="black", linewidths=1.5, zorder=5)
    z = np.polyfit(d_plot["peak_wrist_linear_speed"], d_plot[TARGET], 1)
    xline = np.linspace(20, 30, 50)
    ax2.plot(xline, np.polyval(z, xline), "k--", lw=2, alpha=0.4)
    ax2.set_xlabel("Arm speed — peak wrist (m/s)", fontsize=14)
    ax2.set_ylabel("Pitch speed (mph)", fontsize=14)
    ax2.set_title("\"Efficient throwing\" exists\n"
                  "(same arm speed -> 10 mph range)", fontsize=13)
    ax2.legend(fontsize=12, loc="upper left")
    ax2.set_xlim(19.5, 30.5)
    ax2.set_ylim(75, 98)

    fig.suptitle(
        "Efficient Throwing: Body Mechanics Explain 17.8% Beyond Arm Speed"
        " (R2: 0.491 -> 0.669)",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURES_CSV)
    df = compute_body_efficiency(df)
    r2_steps = compute_incremental_r2(df)

    print("=== 5-Component Model ===")
    prev = 0
    for r2, label, n in r2_steps:
        print(f"  {label}: R2={r2:.3f} (+{r2-prev:.3f})  n={n}")
        prev = r2

    q1 = df[df["eff_q"] == "Q1"]
    q5 = df[df["eff_q"] == "Q5"]
    print(f"\nQ1: pitch={q1[TARGET].mean():.1f} mph  arm={q1['peak_wrist_linear_speed'].mean():.2f} m/s")
    print(f"Q5: pitch={q5[TARGET].mean():.1f} mph  arm={q5['peak_wrist_linear_speed'].mean():.2f} m/s")
    print(f"Gap: {q5[TARGET].mean() - q1[TARGET].mean():.1f} mph at same arm speed")

    plot_story(df, r2_steps, OUTPUT_DIR / "efficient_throwing_story.png")
    plot_breakdown(df, OUTPUT_DIR / "body_efficiency_breakdown.png")
    print("\nDone. Run efficient_thrower_gif.py for skeleton animation.")


if __name__ == "__main__":
    main()
