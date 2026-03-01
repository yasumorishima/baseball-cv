"""Front Leg Brace Analysis — Hitting

Key finding:
  At foot strike, both Q1 and Q5 hitters have similar knee forward velocity.
  By 50ms after foot strike, Q5 knee has STOPPED (velocity ≈ 0).
  Q1 knee is still drifting forward at 50ms.
  This rapid stop-and-extend = the brace. Q5 does it 78ms faster.

  Knee ext peak velocity: Q1=355 deg/s vs Q5=468 deg/s (+32%)
  Time to peak extension: Q1=0.185s vs Q5=0.107s (78ms faster)

Output:
    data/output/llb_brace_hitting.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

OUTPUT_DIR = Path("data/output")
FEATURES_CSV = OUTPUT_DIR / "features_hitting.csv"

TARGET = "exit_velocity_mph"
BAT_SPEED = "peak_wrist_linear_speed"

VEL_COLS = [
    "llb_knee_forward_vel_at_strike",
    "llb_knee_forward_vel_25ms",
    "llb_knee_forward_vel_50ms",
    "llb_knee_forward_vel_100ms",
    "llb_knee_forward_vel_150ms",
]
VEL_TIMES = [0, 25, 50, 100, 150]

Q1_COLOR = "#e67e22"
Q5_COLOR = "#2980b9"


def compute_quintiles(df):
    size_cols = [c for c in ["height_in", "weight_lb"] if c in df.columns]
    base = [BAT_SPEED] + size_cols
    d = df[base + [TARGET]].dropna()
    lm = LinearRegression().fit(d[base], d[TARGET])
    df = df.copy()
    df.loc[d.index, "body_efficiency"] = (
        d[TARGET].values - lm.predict(d[base].values)
    )
    df["eff_q"] = pd.qcut(df["body_efficiency"], q=5,
                          labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    return df


def plot_brace(df, out_path):
    q1 = df[df["eff_q"] == "Q1"]
    q5 = df[df["eff_q"] == "Q5"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # --- Panel 1: Knee forward velocity time series ---
    ax = axes[0]
    avail = [(t, c) for t, c in zip(VEL_TIMES, VEL_COLS) if c in df.columns]
    times = [t for t, _ in avail]
    q1_vel = [q1[c].mean() for _, c in avail]
    q5_vel = [q5[c].mean() for _, c in avail]

    ax.axhline(0, color="gray", lw=1.5, ls="--", alpha=0.6, label="Zero (stopped)")
    ax.axvline(0, color="#e74c3c", lw=2, ls="--", alpha=0.7, label="Foot strike")
    ax.fill_between(times, q1_vel, q5_vel, alpha=0.12, color="#8e44ad",
                    label="Q1 vs Q5 gap")
    ax.plot(times, q1_vel, "o-", color=Q1_COLOR, lw=2.5, ms=8,
            label=f"Q1 ({q1[TARGET].mean():.0f} mph exit)")
    ax.plot(times, q5_vel, "o-", color=Q5_COLOR, lw=2.5, ms=8,
            label=f"Q5 ({q5[TARGET].mean():.0f} mph exit)")

    # Annotate zero crossing
    ax.annotate("Q5 knee stops\nat ~50ms",
                xy=(50, 0), xytext=(70, 0.15),
                fontsize=12, color=Q5_COLOR, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=Q5_COLOR, lw=1.5))
    ax.annotate("Q1 still\nmoving forward",
                xy=(50, q1_vel[2]), xytext=(65, 0.35),
                fontsize=12, color=Q1_COLOR, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=Q1_COLOR, lw=1.5))

    ax.set_xlabel("Time after foot strike (ms)", fontsize=14)
    ax.set_ylabel("Front knee forward velocity (m/s)\n+ = toward pitcher   - = extending", fontsize=13)
    ax.set_title("Front knee stops faster in Q5\n= brace locks in earlier", fontsize=14)
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(times)

    # --- Panel 2: Time to peak extension + peak velocity bars ---
    ax2 = axes[1]
    metrics = {
        "Time to peak\nextension (s)\n↓ lower = faster": (
            "llb_time_to_peak_extension", -1, "s"
        ),
        "Peak knee ext\nvelocity (deg/s)\n↑ higher = stronger": (
            "llb_knee_ext_peak_velocity", +1, "deg/s"
        ),
        "Knee forward\ndecel (m/s²)\n↑ higher = sharper": (
            "llb_knee_forward_decel", +1, "m/s²"
        ),
    }

    x = np.arange(len(metrics))
    w = 0.3
    q1_vals, q5_vals, labels_m = [], [], []
    for label, (col, sign, unit) in metrics.items():
        if col in df.columns:
            v1 = q1[col].mean() * sign
            v5 = q5[col].mean() * sign
            q1_vals.append(v1)
            q5_vals.append(v5)
            labels_m.append(label)

    x = np.arange(len(q1_vals))
    ax2.bar(x - w / 2, q1_vals, w, color=Q1_COLOR, alpha=0.85,
            label=f"Q1 ({q1[TARGET].mean():.0f} mph)", edgecolor="white")
    ax2.bar(x + w / 2, q5_vals, w, color=Q5_COLOR, alpha=0.85,
            label=f"Q5 ({q5[TARGET].mean():.0f} mph)", edgecolor="white")

    for i, (v1, v5) in enumerate(zip(q1_vals, q5_vals)):
        ax2.text(i - w / 2, v1 + max(q1_vals) * 0.02, f"{v1:.2f}",
                 ha="center", fontsize=11, fontweight="bold", color=Q1_COLOR)
        ax2.text(i + w / 2, v5 + max(q5_vals) * 0.02, f"{v5:.2f}",
                 ha="center", fontsize=11, fontweight="bold", color=Q5_COLOR)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_m, fontsize=11)
    ax2.set_title("Brace quality metrics\nQ5 braces faster and stronger", fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylabel("Value (sign-flipped so higher = better brace)", fontsize=11)

    # --- Panel 3: Scatter stride vs time_to_peak_extension, colored by exit vel ---
    ax3 = axes[2]
    cols3 = ["llb_stride_forward", "llb_time_to_peak_extension", TARGET, "eff_q"]
    d3 = df[cols3].dropna()
    sc = ax3.scatter(
        d3["llb_stride_forward"],
        d3["llb_time_to_peak_extension"],
        c=d3[TARGET], cmap="RdYlGn", s=80, alpha=0.85,
        edgecolors="gray", linewidths=0.5, vmin=70, vmax=110,
    )
    plt.colorbar(sc, ax=ax3, label="Exit velocity (mph)")

    # Highlight Q1 and Q5
    for q, color, label in [("Q1", Q1_COLOR, "Q1"), ("Q5", Q5_COLOR, "Q5")]:
        dq = d3[d3["eff_q"] == q]
        ax3.scatter(dq["llb_stride_forward"],
                    dq["llb_time_to_peak_extension"],
                    c=color, s=120, edgecolors="black",
                    linewidths=2, zorder=5, label=label)

    ax3.set_xlabel("Stride length (m)\nlonger = more forward momentum", fontsize=13)
    ax3.set_ylabel("Time to peak knee extension (s)\nshorter = faster brace", fontsize=13)
    ax3.set_title("Stride + fast brace = high exit velocity\n(green = high exit vel)", fontsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        "Front Leg Brace (Hitting): stride creates momentum, rapid knee extension converts it to rotation",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Print summary
    print(f"\nQ1 (n={len(q1)}): exit={q1[TARGET].mean():.1f} mph")
    print(f"Q5 (n={len(q5)}): exit={q5[TARGET].mean():.1f} mph")
    print(f"\nKnee forward velocity after foot strike:")
    for t, c in avail:
        print(f"  {t:4d}ms: Q1={q1[c].mean():+.3f}  Q5={q5[c].mean():+.3f}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not FEATURES_CSV.exists():
        print(f"Error: {FEATURES_CSV} not found.")
        return

    df = pd.read_csv(FEATURES_CSV)
    df = compute_quintiles(df)

    out = OUTPUT_DIR / "llb_brace_hitting.png"
    plot_brace(df, out)
    print("Done.")


if __name__ == "__main__":
    main()
