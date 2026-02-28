"""Head Stability comparison GIF — side-by-side skeleton animation.

Compares the most head-stable vs most head-unstable pitcher,
aligned on foot strike timing. Head markers highlighted in red
with position trail to visualize head drift.

Usage:
    python head_stability_comparison.py

Output:
    data/output/head_stability_comparison.gif
"""

from pathlib import Path

import ezc3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from skeleton_analysis import compute_lead_leg_block_features, load_c3d
from skeleton_c3d import BODY_CONNECTIONS, get_marker_index
from statcast_correlation import parse_pitching_filename

OUTPUT_DIR = Path("data/output")
RAW_DIR = Path("data/raw")

# Head markers
HEAD_MARKERS = {"LFHD", "RFHD", "LBHD", "RBHD"}

HEAD_CONNECTIONS = [
    ("LFHD", "RFHD"), ("RFHD", "RBHD"), ("RBHD", "LBHD"), ("LBHD", "LFHD"),
]


def find_best_worst():
    """Find most stable and most unstable pitcher from local C3D files.

    Picks a pair with similar foot strike timing (fs=522 vs 524)
    so the visual comparison is fair.
    """
    import pandas as pd

    df = pd.read_csv(OUTPUT_DIR / "features_pitching.csv")
    valid = df[df["llb_head_forward_disp"] > 0.10].copy()
    valid["head_stability_score"] = (
        1 - valid["llb_head_forward_disp_post"] / valid["llb_head_forward_disp"]
    ).clip(0, 1)

    valid["exists"] = valid["filename"].apply(
        lambda f: (RAW_DIR / f).exists()
    )
    available = valid[valid["exists"]].copy()

    # Similar foot strike timing, different head stability
    stable = available[available["filename"].str.contains("000538")].iloc[0]
    unstable = available[available["filename"].str.contains("000359")].iloc[0]
    return stable, unstable


def is_head_connection(m1, m2):
    return (m1, m2) in HEAD_CONNECTIONS or (m2, m1) in HEAD_CONNECTIONS


def is_head_marker(label):
    return label in HEAD_MARKERS


def create_comparison_gif(stable_meta, unstable_meta, output_path):
    """Create side-by-side GIF comparing head-stable vs head-unstable."""
    stable_file = RAW_DIR / stable_meta["filename"]
    unstable_file = RAW_DIR / unstable_meta["filename"]

    c_s = ezc3d.c3d(str(stable_file))
    labels_s = c_s["parameters"]["POINT"]["LABELS"]["value"]
    points_s = c_s["data"]["points"]
    rate_s = c_s["parameters"]["POINT"]["RATE"]["value"][0]

    c_u = ezc3d.c3d(str(unstable_file))
    labels_u = c_u["parameters"]["POINT"]["LABELS"]["value"]
    points_u = c_u["data"]["points"]
    rate_u = c_u["parameters"]["POINT"]["RATE"]["value"][0]

    n_s = points_s.shape[2]
    n_u = points_u.shape[2]

    fs_s = int(stable_meta["llb_foot_strike_frame"])
    fs_u = int(unstable_meta["llb_foot_strike_frame"])

    # Align on foot strike: 0.5s before, 0.4s after
    pre_sec, post_sec = 0.5, 0.4

    pre_s = int(pre_sec * rate_s)
    post_s = int(post_sec * rate_s)
    start_s = max(0, fs_s - pre_s)
    end_s = min(n_s, fs_s + post_s)

    pre_u = int(pre_sec * rate_u)
    post_u = int(post_sec * rate_u)
    start_u = max(0, fs_u - pre_u)
    end_u = min(n_u, fs_u + post_u)

    n_anim = 108
    frames_s = np.linspace(start_s, end_s - 1, n_anim).astype(int)
    frames_u = np.linspace(start_u, end_u - 1, n_anim).astype(int)

    fs_gif_frame = int(pre_sec / (pre_sec + post_sec) * n_anim)

    fig, (ax_stable, ax_unstable) = plt.subplots(
        1, 2, figsize=(16, 8), subplot_kw={"projection": "3d"}
    )

    def compute_bounds(points):
        valid = ~np.isnan(points[0]) & (points[0] != 0)
        gx, gy, gz = points[0][valid], points[1][valid], points[2][valid]
        cx, cy, cz = np.mean(gx), np.mean(gy), np.mean(gz)
        span = max(np.ptp(gx), np.ptp(gy), np.ptp(gz)) * 0.6
        return cx, cy, cz, span

    bounds_s = compute_bounds(points_s)
    bounds_u = compute_bounds(points_u)

    # Head trail storage
    head_trail_s = []
    head_trail_u = []

    def get_head_center(labels, points, fi):
        """Average position of head markers."""
        positions = []
        for hm in ["LFHD", "RFHD", "LBHD", "RBHD"]:
            idx = get_marker_index(labels, hm)
            if idx >= 0:
                x, y, z = points[0, idx, fi], points[1, idx, fi], points[2, idx, fi]
                if not np.isnan(x) and not (x == 0 and y == 0 and z == 0):
                    positions.append([x, y, z])
        if positions:
            return np.mean(positions, axis=0)
        return None

    def draw_skeleton(ax, labels, points, fi, bounds, is_foot_strike,
                      title, head_trail):
        ax.cla()
        cx, cy, cz, span = bounds

        # Draw body connections
        for m1_name, m2_name in BODY_CONNECTIONS:
            i1 = get_marker_index(labels, m1_name)
            i2 = get_marker_index(labels, m2_name)
            if i1 < 0 or i2 < 0:
                continue
            x = [points[0, i1, fi], points[0, i2, fi]]
            y = [points[1, i1, fi], points[1, i2, fi]]
            z = [points[2, i1, fi], points[2, i2, fi]]
            if any(np.isnan(x + y + z)) or (x[0] == 0 and y[0] == 0 and z[0] == 0):
                continue

            if is_head_connection(m1_name, m2_name):
                color = "#e74c3c"
                lw = 4
            else:
                color = "#2c3e50"
                lw = 1.5
            ax.plot(x, y, z, color=color, linewidth=lw)

        # Draw markers
        for i, label in enumerate(labels):
            x, y, z = points[0, i, fi], points[1, i, fi], points[2, i, fi]
            if np.isnan(x) or (x == 0 and y == 0 and z == 0):
                continue
            if is_head_marker(label):
                color = "#e74c3c"
                size = 35
            else:
                color = "#3498db"
                size = 10
            ax.scatter(x, y, z, c=color, s=size, alpha=0.8)

        # Head trail (shows drift)
        head_pos = get_head_center(labels, points, fi)
        if head_pos is not None:
            head_trail.append(head_pos)
            if len(head_trail) > 1:
                trail = np.array(head_trail[-30:])  # last 30 frames
                ax.plot(trail[:, 0], trail[:, 1], trail[:, 2],
                        color="#e74c3c", linewidth=2, alpha=0.5)

        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_zlim(cz - span, cz + span)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        fs_marker = "  ** FOOT STRIKE **" if is_foot_strike else ""
        ax.set_title(f"{title}{fs_marker}", fontsize=12, fontweight="bold")
        ax.view_init(elev=20, azim=45)

    score_s = stable_meta["head_stability_score"]
    score_u = unstable_meta["head_stability_score"]

    def update(frame_num):
        fi_s = frames_s[frame_num]
        fi_u = frames_u[frame_num]
        is_fs = abs(frame_num - fs_gif_frame) <= 1

        draw_skeleton(ax_stable, labels_s, points_s, fi_s, bounds_s,
                      is_fs, f"Head Stable (score={score_s:.2f})",
                      head_trail_s)
        draw_skeleton(ax_unstable, labels_u, points_u, fi_u, bounds_u,
                      is_fs, f"Head Unstable (score={score_u:.2f})",
                      head_trail_u)

        fig.suptitle("Head Stability Comparison — Driveline OBP\n"
                     "Red = Head markers & trail",
                     fontsize=13, y=0.98)

    print(f"  Generating {n_anim} frames...")
    anim = animation.FuncAnimation(fig, update, frames=n_anim, interval=80)
    anim.save(str(output_path), writer="pillow", fps=12)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stable, unstable = find_best_worst()
    print(f"Stable:   {stable['filename']} (score={stable['head_stability_score']:.3f}, "
          f"post_disp={stable['llb_head_forward_disp_post']:.3f}m)")
    print(f"Unstable: {unstable['filename']} (score={unstable['head_stability_score']:.3f}, "
          f"post_disp={unstable['llb_head_forward_disp_post']:.3f}m)")

    output_path = OUTPUT_DIR / "head_stability_comparison.gif"
    create_comparison_gif(stable, unstable, output_path)
    print("Done!")


if __name__ == "__main__":
    main()
