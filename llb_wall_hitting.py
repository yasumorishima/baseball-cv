"""Lead Leg Block — Hitting: The Wall Effect

Key story:
  After foot strike, the front ankle plants and the knee extends.
  Q5 (efficient): hip stops moving FORWARD and starts rotating around the ankle
                  -> the ankle becomes a fixed pivot = "the wall"
                  -> rotational energy is maximized -> high exit velocity
  Q1 (poor):      hip keeps drifting forward past the ankle -> no wall
                  -> rotation leaks into linear motion -> low exit velocity

The "wall" signal = hip-ankle gap in the forward direction (toward pitcher).
  - Gap stabilizing after foot strike = wall created (Q5)
  - Gap keeps growing after foot strike = no wall (Q1)

Layout:
  Top:    Q1 | Q5 3D skeletons, lead leg in red
  Bottom: Hip and ankle forward positions over time
          Hip-ankle gap (solid fill) = the "wall" indicator

Usage:
    python llb_wall_hitting.py

Output:
    data/output/llb_wall_hitting.gif
"""

from pathlib import Path

import ezc3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from skeleton_c3d import BODY_CONNECTIONS, get_marker_index

OUTPUT_DIR = Path("data/output")
RAW_DIR = Path("data/raw")

Q1_FILE = RAW_DIR / "000009_000123_63_140_R_001_746.c3d"
Q5_FILE = RAW_DIR / "000004_000103_75_236_R_003_972.c3d"

Q1_META = {
    "exit_velocity_mph": 74.6, "bat_speed": 7.23,
    "stride_m": 0.718, "body_efficiency": -11.88,
    "llb_foot_strike_frame": 374,
    "label": "Q1: No Wall — Hip Drifts Forward",
    "color": "#e67e22",
}
Q5_META = {
    "exit_velocity_mph": 97.2, "bat_speed": 7.80,
    "stride_m": 0.993, "body_efficiency": +9.70,
    "llb_foot_strike_frame": 574,
    "label": "Q5: Wall Created — Hip Rotates",
    "color": "#2980b9",
}

LEAD_LEG_MARKERS = {
    "LASI", "LKNE", "LMKNE", "LANK", "LMANK", "LHEE", "LTOE", "LTHI", "LTIB",
}
LEAD_LEG_CONNECTIONS = [
    ("LASI", "LKNE"), ("LKNE", "LMKNE"), ("LKNE", "LANK"),
    ("LANK", "LMANK"), ("LANK", "LHEE"), ("LANK", "LTOE"),
    ("LASI", "LTHI"), ("LKNE", "LTIB"),
]


def is_lead_leg(m1, m2):
    return (m1, m2) in LEAD_LEG_CONNECTIONS or (m2, m1) in LEAD_LEG_CONNECTIONS


def compute_forward_dir(labels, points, fs_frame):
    """Forward direction = from back heel to front heel at foot strike."""
    ri = get_marker_index(labels, "RHEE")
    li = get_marker_index(labels, "LHEE")
    if ri < 0 or li < 0:
        return np.array([0.0, 1.0, 0.0])
    fwd = points[:3, li, fs_frame] - points[:3, ri, fs_frame]
    fwd[2] = 0.0
    norm = np.linalg.norm(fwd)
    return fwd / norm if norm > 1e-6 else np.array([0.0, 1.0, 0.0])


def get_marker_fwd_series(labels, points, name, fwd):
    """Project marker position onto forward direction for all frames."""
    idx = get_marker_index(labels, name)
    if idx < 0:
        return np.full(points.shape[2], np.nan)
    series = np.empty(points.shape[2])
    for f in range(points.shape[2]):
        x, y, z = points[0, idx, f], points[1, idx, f], points[2, idx, f]
        if np.isnan(x) or (x == 0 and y == 0 and z == 0):
            series[f] = np.nan
        else:
            series[f] = np.dot([x, y, z], fwd)
    return series


def get_bounds(pts):
    valid = ~np.isnan(pts[0]) & (pts[0] != 0)
    gx, gy, gz = pts[0][valid], pts[1][valid], pts[2][valid]
    cx, cy, cz = np.mean(gx), np.mean(gy), np.mean(gz)
    sp = max(np.ptp(gx), np.ptp(gy), np.ptp(gz)) * 0.60
    return cx, cy, cz, sp


def draw_skeleton(ax, labels, points, fi, bnd, meta, is_fs):
    ax.cla()
    cx, cy, cz, sp = bnd

    for m1, m2 in BODY_CONNECTIONS:
        i1 = get_marker_index(labels, m1)
        i2 = get_marker_index(labels, m2)
        if i1 < 0 or i2 < 0:
            continue
        x = [points[0, i1, fi], points[0, i2, fi]]
        y = [points[1, i1, fi], points[1, i2, fi]]
        z = [points[2, i1, fi], points[2, i2, fi]]
        if any(np.isnan(x + y + z)) or (x[0] == 0 and y[0] == 0 and z[0] == 0):
            continue
        if is_lead_leg(m1, m2):
            ax.plot(x, y, z, color="#e74c3c", linewidth=5)
        else:
            ax.plot(x, y, z, color="#2c3e50", linewidth=1.5)

    for i, label in enumerate(labels):
        x, y, z = points[0, i, fi], points[1, i, fi], points[2, i, fi]
        if np.isnan(x) or (x == 0 and y == 0 and z == 0):
            continue
        if label in LEAD_LEG_MARKERS:
            ax.scatter(x, y, z, c="#e74c3c", s=35, alpha=0.9, zorder=5)
        else:
            ax.scatter(x, y, z, c="#3498db", s=8, alpha=0.5)

    ax.set_xlim(cx - sp, cx + sp)
    ax.set_ylim(cy - sp, cy + sp)
    ax.set_zlim(cz - sp, cz + sp)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(elev=10, azim=90)  # side view: look along X, see YZ plane

    fs_tag = "  ** FOOT STRIKE **" if is_fs else ""
    eff_sign = "+" if meta["body_efficiency"] > 0 else ""
    ax.set_title(
        f"{meta['label']}{fs_tag}\n"
        f"exit {meta['exit_velocity_mph']:.1f} mph  |  "
        f"stride {meta['stride_m']:.2f} m  |  "
        f"body eff {eff_sign}{meta['body_efficiency']:.2f} mph",
        fontsize=10, fontweight="bold", color=meta["color"],
    )


def draw_wall_graph(ax, common_time, hip_q1, hip_q5, ank_q1, ank_q5, frame_num):
    ax.clear()

    # Gap = hip forward position minus ankle forward position
    # Positive = hip is ahead of ankle in the forward direction
    gap_q1 = (hip_q1 - ank_q1) * 100   # convert to cm
    gap_q5 = (hip_q5 - ank_q5) * 100

    # Fill between lines to show the gap
    ax.fill_between(common_time, gap_q1, gap_q5,
                    where=common_time >= 0,
                    alpha=0.12, color="#8e44ad",
                    label="Gap difference (post-FS)")

    ax.plot(common_time, gap_q1, color=Q1_META["color"], linewidth=2.5,
            label=f"Q1: hip-ankle gap  (exit={Q1_META['exit_velocity_mph']:.0f} mph)")
    ax.plot(common_time, gap_q5, color=Q5_META["color"], linewidth=2.5,
            label=f"Q5: hip-ankle gap  (exit={Q5_META['exit_velocity_mph']:.0f} mph)")

    ax.axvline(0, color="#e74c3c", linewidth=2.5, linestyle="--",
               alpha=0.85, label="Foot strike")

    # Shade post-FS zone
    ymin, ymax = ax.get_ylim() if ax.get_ylim() != (0, 1) else (-5, 60)
    ax.axvspan(0, common_time[-1], alpha=0.04, color="#2980b9")

    t_now = common_time[frame_num]
    ax.scatter(t_now, gap_q1[frame_num], s=180, c=Q1_META["color"],
               zorder=6, edgecolors="black", linewidths=2)
    ax.scatter(t_now, gap_q5[frame_num], s=180, c=Q5_META["color"],
               zorder=6, edgecolors="black", linewidths=2)

    ax.set_xlabel(
        "<-- before foot strike   |   Time (s)   |   after foot strike -->",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylabel("Hip ahead of ankle — forward direction (cm)\n"
                  "smaller gap after FS = wall created", fontsize=10)
    ax.set_title(
        "The Wall: after foot strike, Q5 hip stops going forward (gap stabilizes) -> pivot created -> hip rotates\n"
        "Q1 hip keeps moving forward (gap grows) -> no pivot -> rotation lost as linear drift",
        fontsize=10, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(common_time[0] - 0.01, common_time[-1] + 0.01)


def create_gif(output_path):
    if not Q1_FILE.exists() or not Q5_FILE.exists():
        raise FileNotFoundError(f"Missing C3D files: {Q1_FILE} or {Q5_FILE}")

    c1 = ezc3d.c3d(str(Q1_FILE))
    labels1 = c1["parameters"]["POINT"]["LABELS"]["value"]
    pts1 = c1["data"]["points"]
    rate1 = c1["parameters"]["POINT"]["RATE"]["value"][0]

    c5 = ezc3d.c3d(str(Q5_FILE))
    labels5 = c5["parameters"]["POINT"]["LABELS"]["value"]
    pts5 = c5["data"]["points"]
    rate5 = c5["parameters"]["POINT"]["RATE"]["value"][0]

    fs1 = Q1_META["llb_foot_strike_frame"]
    fs5 = Q5_META["llb_foot_strike_frame"]

    fwd1 = compute_forward_dir(labels1, pts1, fs1)
    fwd5 = compute_forward_dir(labels5, pts5, fs5)
    print(f"  Q1 forward dir: {fwd1}")
    print(f"  Q5 forward dir: {fwd5}")

    hip1 = get_marker_fwd_series(labels1, pts1, "LASI", fwd1)
    ank1 = get_marker_fwd_series(labels1, pts1, "LANK", fwd1)
    hip5 = get_marker_fwd_series(labels5, pts5, "LASI", fwd5)
    ank5 = get_marker_fwd_series(labels5, pts5, "LANK", fwd5)

    pre_sec, post_sec = 0.35, 0.45
    total_sec = pre_sec + post_sec
    n_anim = 120

    start1 = max(0, fs1 - int(pre_sec * rate1))
    end1 = min(pts1.shape[2], fs1 + int(post_sec * rate1))
    start5 = max(0, fs5 - int(pre_sec * rate5))
    end5 = min(pts5.shape[2], fs5 + int(post_sec * rate5))

    idx1 = np.linspace(start1, end1 - 1, n_anim).astype(int)
    idx5 = np.linspace(start5, end5 - 1, n_anim).astype(int)
    fs_gif = int(pre_sec / total_sec * n_anim)

    common_time = np.linspace(-pre_sec, post_sec, n_anim)

    gt1 = (np.arange(start1, end1) - fs1) / rate1
    gt5 = (np.arange(start5, end5) - fs5) / rate5

    def interp_zeroed(series, start, end, gt):
        """Interpolate to common_time and zero at foot strike."""
        s = np.interp(common_time, gt, series[start:end])
        fs_val = np.interp(0.0, gt, series[start:end])
        return s - fs_val

    hip1_i = interp_zeroed(hip1, start1, end1, gt1)
    ank1_i = interp_zeroed(ank1, start1, end1, gt1)
    hip5_i = interp_zeroed(hip5, start5, end5, gt5)
    ank5_i = interp_zeroed(ank5, start5, end5, gt5)

    bnd1 = get_bounds(pts1)
    bnd5 = get_bounds(pts5)

    fig = plt.figure(figsize=(18, 12))
    ax_q1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax_q5 = fig.add_subplot(2, 2, 2, projection="3d")
    ax_g = fig.add_axes([0.08, 0.05, 0.86, 0.24])

    overlay_texts = []

    def update(frame_num):
        for t in overlay_texts:
            t.remove()
        overlay_texts.clear()

        fi1 = idx1[frame_num]
        fi5 = idx5[frame_num]
        is_fs = abs(frame_num - fs_gif) <= 1
        is_post = frame_num > fs_gif + 3

        draw_skeleton(ax_q1, labels1, pts1, fi1, bnd1, Q1_META, is_fs)
        draw_skeleton(ax_q5, labels5, pts5, fi5, bnd5, Q5_META, is_fs)
        draw_wall_graph(ax_g, common_time, hip1_i, hip5_i, ank1_i, ank5_i, frame_num)

        if is_fs:
            for xpos in (0.23, 0.71):
                t = fig.text(
                    xpos, 0.31, "FOOT STRIKE",
                    fontsize=13, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor="#e74c3c", alpha=0.95),
                )
                overlay_texts.append(t)

        fig.suptitle(
            "The Wall: hip-ankle gap in forward direction after foot strike (Driveline OBP)\n"
            "Red = lead leg  |  Stable gap after foot strike = wall created",
            fontsize=12, fontweight="bold", y=0.99,
        )

    print(f"  Rendering {n_anim} frames...")
    anim = animation.FuncAnimation(fig, update, frames=n_anim, interval=80)
    anim.save(str(output_path), writer="pillow", fps=12)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "llb_wall_hitting.gif"
    print(f"Q1: {Q1_FILE.name}")
    print(f"Q5: {Q5_FILE.name}")
    create_gif(out)
    print("Done.")


if __name__ == "__main__":
    main()
