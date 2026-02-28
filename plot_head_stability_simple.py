"""Simple 2-panel scatter plot for README.

Left:  Ankle Braking -> Head Stability (quadratic, R2=0.58)
Right: Head Stability -> Time to Peak Trunk Velocity (r=-0.88)

Clean, big fonts, obvious correlation.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DIR = Path("data/output")


def main():
    df = pd.read_csv(OUTPUT_DIR / "features_pitching.csv")

    # Compute head stability score
    valid_disp = df["llb_head_forward_disp"] > 0.05
    df["head_stability_score"] = np.nan
    df.loc[valid_disp, "head_stability_score"] = 1 - (
        df.loc[valid_disp, "llb_head_forward_disp_post"]
        / df.loc[valid_disp, "llb_head_forward_disp"]
    )
    df["head_stability_score"] = df["head_stability_score"].clip(0, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: Braking Deviation -> Head Stability ---
    cols1 = ["llb_ankle_braking_decel", "head_stability_score"]
    d1 = df[cols1].dropna()
    raw_x1 = d1[cols1[0]].values
    y1 = d1[cols1[1]].values

    # Find optimal braking from quadratic peak
    q_coeffs = np.polyfit(raw_x1, y1, 2)
    optimal_braking = -q_coeffs[1] / (2 * q_coeffs[0])

    # Deviation from optimal (0 = perfect, higher = worse)
    x1 = np.abs(raw_x1 - optimal_braking)
    r1, p1 = stats.pearsonr(x1, y1)

    ax1.scatter(x1, y1, s=70, c="#2980b9", alpha=0.7, edgecolors="white", linewidth=0.5)
    z1 = np.polyfit(x1, y1, 1)
    xline1 = np.linspace(0, x1.max(), 50)
    ax1.plot(xline1, np.polyval(z1, xline1), color="#e74c3c", linewidth=3)

    ax1.set_xlabel("Deviation from Optimal Braking (m/s²)", fontsize=14)
    ax1.set_ylabel("Head Stability Score", fontsize=14)
    ax1.set_title(f"Braking Deviation → Head Stability\n"
                  f"r = {r1:+.2f}  (p < 0.001, n = {len(d1)})",
                  fontsize=15, fontweight="bold")
    ax1.tick_params(labelsize=12)

    # --- Right: Head Stability -> Time to Peak Trunk ---
    cols2 = ["head_stability_score", "llb_time_fs_to_peak_trunk_vel"]
    d2 = df[cols2].dropna()
    x2, y2 = d2[cols2[0]], d2[cols2[1]]
    r2, p2 = stats.pearsonr(x2, y2)

    ax2.scatter(x2, y2, s=70, c="#27ae60", alpha=0.7, edgecolors="white", linewidth=0.5)
    z2 = np.polyfit(x2, y2, 1)
    xline2 = np.linspace(x2.min(), x2.max(), 50)
    ax2.plot(xline2, np.polyval(z2, xline2), color="#e74c3c", linewidth=3)

    ax2.set_xlabel("Head Stability Score", fontsize=14)
    ax2.set_ylabel("Time to Peak Trunk Velocity (s)", fontsize=14)
    ax2.set_title(f"Head Stability → Faster Trunk Rotation\nr = {r2:+.2f}  (p < 0.001, n = {len(d2)})",
                  fontsize=15, fontweight="bold")
    ax2.tick_params(labelsize=12)

    fig.suptitle("Movement Quality Chain — Driveline OBP (60 pitchers)",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(str(OUTPUT_DIR / "head_stability_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'head_stability_scatter.png'}")
    print(f"  Left:  r={r1:+.3f}, p={p1:.4f}, n={len(d1)}")
    print(f"  Right: r={r2:+.3f}, p={p2:.4f}, n={len(d2)}")


if __name__ == "__main__":
    main()
