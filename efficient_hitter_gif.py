"""Efficient Hitter comparison GIF.

Q1 (worst body efficiency) vs Q5 (best body efficiency).
Similar bat speed (~7-8 m/s) but 22.6 mph difference in exit velocity.

Key story:
  Q1: bat=7.23 m/s, exit=74.6 mph, stride=0.72m (weak body use)
  Q5: bat=7.80 m/s, exit=97.2 mph, stride=0.99m (strong body use)

Red = lead leg (stride + front knee highlighted).
Foot strike anchor dot shows stride endpoint.

Usage:
    python efficient_hitter_gif.py

Output:
    data/output/efficient_hitter_comparison.gif
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

# Q1: short stride, low exit velocity (R-handed batter)
# Q5: long stride, high exit velocity (R-handed batter)
Q1_FILE = "000009_000123_63_140_R_001_746.c3d"
Q5_FILE = "000004_000103_75_236_R_003_972.c3d"

Q1_META = {
    "exit_velocity_mph": 74.6, "bat_speed": 7.23,
    "stride_m": 0.718, "body_efficiency": -11.88,
    "llb_foot_strike_frame": 374,
}
Q5_META = {
    "exit_velocity_mph": 97.2, "bat_speed": 7.80,
    "stride_m": 0.993, "body_efficiency": +9.70,
    "llb_foot_strike_frame": 574,
}

# Lead leg for right-handed batter = left leg (steps toward pitcher)
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


def compute_bounds(points):
    valid = ~np.isnan(points[0]) & (points[0] != 0)
    gx, gy, gz = points[0][valid], points[1][valid], points[2][valid]
    cx, cy, cz = np.mean(gx), np.mean(gy), np.mean(gz)
    span = max(np.ptp(gx), np.ptp(gy), np.ptp(gz)) * 0.62
    return cx, cy, cz, span


def get_marker_pos(labels, points, fi, marker_name):
    idx = get_marker_index(labels, marker_name)
    if idx < 0:
        return None
    x, y, z = points[0, idx, fi], points[1, idx, fi], points[2, idx, fi]
    if np.isnan(x) or (x == 0 and y == 0 and z == 0):
        return None
    return np.array([x, y, z])


def draw_skeleton(ax, labels, points, fi, bounds, is_fs, title, meta, foot_anchor):
    ax.cla()
    cx, cy, cz, span = bounds

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
            ax.plot(x, y, z, color="#e74c3c", linewidth=3.5)
        else:
            ax.plot(x, y, z, color="#2c3e50", linewidth=1.5)

    for i, label in enumerate(labels):
        x, y, z = points[0, i, fi], points[1, i, fi], points[2, i, fi]
        if np.isnan(x) or (x == 0 and y == 0 and z == 0):
            continue
        if label in LEAD_LEG_MARKERS:
            ax.scatter(x, y, z, c="#e74c3c", s=30, alpha=0.85, zorder=5)
        else:
            ax.scatter(x, y, z, c="#3498db", s=8, alpha=0.6)

    if foot_anchor is not None:
        ax.scatter(*foot_anchor, c="#f39c12", s=120, marker="*",
                   zorder=10, label="Foot strike")

    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_zlim(cz - span, cz + span)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    fs_tag = "  ** FOOT STRIKE **" if is_fs else ""
    eff_sign = "+" if meta["body_efficiency"] > 0 else ""
    ax.set_title(
        f"{title}{fs_tag}\n"
        f"{meta['exit_velocity_mph']:.1f} mph exit  |  bat {meta['bat_speed']:.2f} m/s  |  "
        f"stride {meta['stride_m']:.2f}m  |  eff {eff_sign}{meta['body_efficiency']:.2f}",
        fontsize=10, fontweight="bold",
    )
    ax.view_init(elev=15, azim=50)


def create_gif(output_path):
    q1_path = RAW_DIR / Q1_FILE
    q5_path = RAW_DIR / Q5_FILE

    if not q1_path.exists() or not q5_path.exists():
        raise FileNotFoundError(f"Missing C3D files: {q1_path} or {q5_path}")

    c1 = ezc3d.c3d(str(q1_path))
    labels1 = c1["parameters"]["POINT"]["LABELS"]["value"]
    pts1 = c1["data"]["points"]
    rate1 = c1["parameters"]["POINT"]["RATE"]["value"][0]

    c5 = ezc3d.c3d(str(q5_path))
    labels5 = c5["parameters"]["POINT"]["LABELS"]["value"]
    pts5 = c5["data"]["points"]
    rate5 = c5["parameters"]["POINT"]["RATE"]["value"][0]

    fs1 = Q1_META["llb_foot_strike_frame"]
    fs5 = Q5_META["llb_foot_strike_frame"]

    pre_sec = 0.25
    n_total1 = pts1.shape[2]
    n_total5 = pts5.shape[2]

    pre1 = min(fs1, int(pre_sec * rate1))
    pre5 = min(fs5, int(pre_sec * rate5))
    post1_avail = n_total1 - fs1
    post5_avail = n_total5 - fs5
    common_post = min(post1_avail, post5_avail)

    print(f"  File lengths: Q1={n_total1} frames ({n_total1/rate1:.2f}s), "
          f"Q5={n_total5} frames ({n_total5/rate5:.2f}s)")
    print(f"  Post-FS available: Q1={post1_avail/rate1:.2f}s, "
          f"Q5={post5_avail/rate5:.2f}s -> using {common_post/rate1:.2f}s")

    total1 = pre1 + common_post
    n_anim = min(total1, 240)

    frames1 = np.linspace(fs1 - pre1, fs1 + common_post - 1, n_anim).astype(int)
    frames5 = np.linspace(fs5 - pre5, fs5 + common_post - 1, n_anim).astype(int)
    fs_gif = int((pre1 / total1) * n_anim)

    bounds1 = compute_bounds(pts1)
    bounds5 = compute_bounds(pts5)

    def get_anchor(labels, points, fs):
        idx = get_marker_index(labels, "LHEE")
        if idx < 0:
            return None
        window_end = min(fs + 120, points.shape[2])
        hz = points[2, idx, fs:window_end]
        valid = ~np.isnan(hz) & (hz != 0)
        if not valid.any():
            return None
        fi = fs + int(np.argmin(np.where(valid, hz, np.inf)))
        x, y, z = points[0, idx, fi], points[1, idx, fi], points[2, idx, fi]
        if np.isnan(x) or (x == 0 and y == 0 and z == 0):
            return None
        print(f"    anchor frame={fi} XYZ=({x:.3f}, {y:.3f}, {z:.3f})")
        return np.array([x, y, z])

    anchor1 = get_anchor(labels1, pts1, fs1)
    anchor5 = get_anchor(labels5, pts5, fs5)

    fig, (ax1, ax5) = plt.subplots(1, 2, figsize=(16, 8),
                                    subplot_kw={"projection": "3d"})

    def update(frame_num):
        fi1 = frames1[frame_num]
        fi5 = frames5[frame_num]
        is_fs = abs(frame_num - fs_gif) <= 1

        draw_skeleton(ax1, labels1, pts1, fi1, bounds1, is_fs,
                      "Q1: Short stride (poor body use)", Q1_META, anchor1)
        draw_skeleton(ax5, labels5, pts5, fi5, bounds5, is_fs,
                      "Q5: Long stride (efficient body use)", Q5_META, anchor5)

        fig.suptitle(
            "Efficient Hitting: Same bat speed -> 22.6 mph exit velocity difference (Driveline OBP)\n"
            "Red = Lead leg (stride foot)  |  Orange star = Foot strike landing",
            fontsize=12, fontweight="bold", y=0.99,
        )

    print(f"  Generating {n_anim} frames...")
    anim = animation.FuncAnimation(fig, update, frames=n_anim, interval=80)
    anim.save(str(output_path), writer="pillow", fps=12)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "efficient_hitter_comparison.gif"
    print(f"Q1: {Q1_FILE}")
    print(f"  exit={Q1_META['exit_velocity_mph']} mph, bat={Q1_META['bat_speed']} m/s, "
          f"stride={Q1_META['stride_m']}m, eff={Q1_META['body_efficiency']:.2f}")
    print(f"Q5: {Q5_FILE}")
    print(f"  exit={Q5_META['exit_velocity_mph']} mph, bat={Q5_META['bat_speed']} m/s, "
          f"stride={Q5_META['stride_m']}m, eff={Q5_META['body_efficiency']:.2f}")
    create_gif(out)
    print("Done.")


if __name__ == "__main__":
    main()
