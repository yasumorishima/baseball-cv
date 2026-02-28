"""Head Stability static comparison — foot strike vs 100ms after.

Shows 2x2 grid: Stable/Unstable x FootStrike/100ms After
with head position circled and drift arrow for unstable pitcher.

Usage:
    python head_stability_static.py

Output:
    data/output/head_stability_static.png
"""

from pathlib import Path

import ezc3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from skeleton_c3d import BODY_CONNECTIONS, get_marker_index

OUTPUT_DIR = Path("data/output")
RAW_DIR = Path("data/raw")

HEAD_MARKERS = {"LFHD", "RFHD", "LBHD", "RBHD"}
HEAD_CONNECTIONS = [
    ("LFHD", "RFHD"), ("RFHD", "RBHD"), ("RBHD", "LBHD"), ("LBHD", "LFHD"),
]


def find_best_worst():
    import pandas as pd
    df = pd.read_csv(OUTPUT_DIR / "features_pitching.csv")
    valid = df[df["llb_head_forward_disp"] > 0.10].copy()
    valid["head_stability_score"] = (
        1 - valid["llb_head_forward_disp_post"] / valid["llb_head_forward_disp"]
    ).clip(0, 1)
    valid["exists"] = valid["filename"].apply(lambda f: (RAW_DIR / f).exists())
    available = valid[valid["exists"]].sort_values("head_stability_score")
    return available.iloc[-1], available.iloc[0]  # stable, unstable


def get_head_center(labels, points, fi):
    positions = []
    for hm in HEAD_MARKERS:
        idx = get_marker_index(labels, hm)
        if idx >= 0:
            x, y, z = points[0, idx, fi], points[1, idx, fi], points[2, idx, fi]
            if not np.isnan(x) and not (x == 0 and y == 0 and z == 0):
                positions.append([x, y, z])
    return np.mean(positions, axis=0) if positions else None


def is_head_connection(m1, m2):
    return (m1, m2) in HEAD_CONNECTIONS or (m2, m1) in HEAD_CONNECTIONS


def draw_skeleton_2d(ax, labels, points, fi, head_center=None, head_circle=True):
    """Draw skeleton in XZ plane (side view) — X=forward, Z=up."""
    for m1_name, m2_name in BODY_CONNECTIONS:
        i1 = get_marker_index(labels, m1_name)
        i2 = get_marker_index(labels, m2_name)
        if i1 < 0 or i2 < 0:
            continue
        x = [points[0, i1, fi], points[0, i2, fi]]
        z = [points[2, i1, fi], points[2, i2, fi]]
        if any(np.isnan(x + z)) or (x[0] == 0 and z[0] == 0):
            continue
        if is_head_connection(m1_name, m2_name):
            ax.plot(x, z, color="#e74c3c", linewidth=3, zorder=3)
        else:
            ax.plot(x, z, color="#2c3e50", linewidth=1.5, zorder=2)

    for i, label in enumerate(labels):
        x, z = points[0, i, fi], points[2, i, fi]
        if np.isnan(x) or (x == 0 and z == 0):
            continue
        if label in HEAD_MARKERS:
            ax.scatter(x, z, c="#e74c3c", s=40, alpha=0.9, zorder=4)
        else:
            ax.scatter(x, z, c="#3498db", s=8, alpha=0.6, zorder=2)

    if head_center is not None and head_circle:
        circle = plt.Circle((head_center[0], head_center[2]), 0.06,
                             fill=False, color="#e74c3c", linewidth=2.5,
                             linestyle="--", zorder=5)
        ax.add_patch(circle)


def main():
    stable_meta, unstable_meta = find_best_worst()

    print(f"Stable:   {stable_meta['filename']} (score={stable_meta['head_stability_score']:.3f})")
    print(f"Unstable: {unstable_meta['filename']} (score={unstable_meta['head_stability_score']:.3f})")

    # Load C3D
    c_s = ezc3d.c3d(str(RAW_DIR / stable_meta["filename"]))
    labels_s = c_s["parameters"]["POINT"]["LABELS"]["value"]
    points_s = c_s["data"]["points"]
    rate_s = c_s["parameters"]["POINT"]["RATE"]["value"][0]

    c_u = ezc3d.c3d(str(RAW_DIR / unstable_meta["filename"]))
    labels_u = c_u["parameters"]["POINT"]["LABELS"]["value"]
    points_u = c_u["data"]["points"]
    rate_u = c_u["parameters"]["POINT"]["RATE"]["value"][0]

    fs_s = int(stable_meta["llb_foot_strike_frame"])
    fs_u = int(unstable_meta["llb_foot_strike_frame"])

    post_ms = 100  # 100ms after foot strike
    post_frames_s = int(post_ms / 1000 * rate_s)
    post_frames_u = int(post_ms / 1000 * rate_u)

    fi_s_fs = fs_s
    fi_s_post = min(fs_s + post_frames_s, points_s.shape[2] - 1)
    fi_u_fs = fs_u
    fi_u_post = min(fs_u + post_frames_u, points_u.shape[2] - 1)

    # Head centers
    head_s_fs = get_head_center(labels_s, points_s, fi_s_fs)
    head_s_post = get_head_center(labels_s, points_s, fi_s_post)
    head_u_fs = get_head_center(labels_u, points_u, fi_u_fs)
    head_u_post = get_head_center(labels_u, points_u, fi_u_post)

    # Compute bounds for each pitcher (shared between their two frames)
    def get_bounds(points):
        valid = ~np.isnan(points[0]) & (points[0] != 0)
        gx, gz = points[0][valid], points[2][valid]
        cx, cz = np.mean(gx), np.mean(gz)
        span = max(np.ptp(gx), np.ptp(gz)) * 0.55
        return cx, cz, span

    cx_s, cz_s, span_s = get_bounds(points_s)
    cx_u, cz_u, span_u = get_bounds(points_u)

    # --- Create 2x2 figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    panels = [
        (axes[0, 0], labels_s, points_s, fi_s_fs, head_s_fs,
         cx_s, cz_s, span_s, "Head Stable — Foot Strike"),
        (axes[0, 1], labels_s, points_s, fi_s_post, head_s_post,
         cx_s, cz_s, span_s, "Head Stable — 100ms After"),
        (axes[1, 0], labels_u, points_u, fi_u_fs, head_u_fs,
         cx_u, cz_u, span_u, "Head Unstable — Foot Strike"),
        (axes[1, 1], labels_u, points_u, fi_u_post, head_u_post,
         cx_u, cz_u, span_u, "Head Unstable — 100ms After"),
    ]

    for ax, labels, points, fi, head_pos, cx, cz, span, title in panels:
        draw_skeleton_2d(ax, labels, points, fi, head_pos)
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cz - span, cz + span)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("X (forward)")
        ax.set_ylabel("Z (up)")

    # Draw drift arrow for unstable pitcher (foot strike -> 100ms after)
    if head_u_fs is not None and head_u_post is not None:
        dx = head_u_post[0] - head_u_fs[0]
        dz = head_u_post[2] - head_u_fs[2]
        drift_cm = np.sqrt(dx**2 + dz**2) * 100

        # Arrow on the "100ms after" panel
        ax_arrow = axes[1, 1]
        ax_arrow.annotate(
            f"Drift: {drift_cm:.0f} cm",
            xy=(head_u_post[0], head_u_post[2]),
            xytext=(head_u_fs[0] - 0.05, head_u_fs[2] + 0.08),
            fontsize=12, fontweight="bold", color="#e74c3c",
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2.5),
            zorder=10,
        )

    # Draw stable annotation (minimal drift)
    if head_s_fs is not None and head_s_post is not None:
        dx = head_s_post[0] - head_s_fs[0]
        dz = head_s_post[2] - head_s_fs[2]
        drift_cm = np.sqrt(dx**2 + dz**2) * 100

        ax_arrow = axes[0, 1]
        ax_arrow.annotate(
            f"Drift: {drift_cm:.0f} cm",
            xy=(head_s_post[0], head_s_post[2]),
            xytext=(head_s_post[0] - 0.05, head_s_post[2] + 0.08),
            fontsize=12, fontweight="bold", color="#27ae60",
            arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2.5),
            zorder=10,
        )

    # Draw foot strike position marker on FS panels
    for ax_fs in [axes[0, 0], axes[1, 0]]:
        ax_fs.text(0.02, 0.02, "FOOT STRIKE", transform=ax_fs.transAxes,
                   fontsize=11, color="#e67e22", fontweight="bold",
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.suptitle(
        "Head Stability: Foot Strike vs 100ms After\n"
        "Driveline OBP — Red = Head markers",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(str(OUTPUT_DIR / "head_stability_static.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'head_stability_static.png'}")

    if head_s_fs is not None and head_s_post is not None:
        d = np.sqrt(sum((head_s_post[i] - head_s_fs[i])**2 for i in range(3)))
        print(f"  Stable drift: {d*100:.1f} cm")
    if head_u_fs is not None and head_u_post is not None:
        d = np.sqrt(sum((head_u_post[i] - head_u_fs[i])**2 for i in range(3)))
        print(f"  Unstable drift: {d*100:.1f} cm")


if __name__ == "__main__":
    main()
