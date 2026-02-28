"""Lead Leg Block — Full-body skeleton with knee extension annotations.

Shows full 3D skeleton (like llb_comparison.gif) with real-time knee
angle, extension velocity, and explanatory text overlaid. Side-by-side
strong vs weak comparison with time-series graph at bottom.

Usage:
    python llb_knee_detail.py --download 40

Output:
    data/output/llb_knee_detail.gif
"""

import argparse
from pathlib import Path

import ezc3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from skeleton_analysis import (
    compute_angular_velocity,
    compute_elbow_flexion,
    compute_knee_flexion,
    compute_lead_leg_block_features,
    load_c3d,
)
from skeleton_c3d import BODY_CONNECTIONS, get_marker_index
from statcast_correlation import download_additional_samples, parse_pitching_filename

OUTPUT_DIR = Path("data/output")
RAW_DIR = Path("data/raw")

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


def collect_llb_candidates(n_samples=40):
    """Download C3D files and find best/worst LLB by knee ext velocity."""
    download_additional_samples("pitching", n_samples)
    results = []
    for fpath in sorted(RAW_DIR.glob("*.c3d")):
        meta = parse_pitching_filename(fpath.name)
        if meta is None:
            continue
        try:
            markers, rate, _ = load_c3d(str(fpath))
            llb = compute_lead_leg_block_features(markers, rate, side="L")
            if not llb:
                continue
            results.append({
                "filename": fpath.name,
                "pitch_speed_mph": meta["pitch_speed_mph"],
                "llb_knee_ext_peak_velocity": llb.get("llb_knee_ext_peak_velocity", 0),
                "llb_foot_strike_frame": llb.get("llb_foot_strike_frame"),
            })
        except Exception as e:
            print(f"  Skip {fpath.name}: {e}")
    results.sort(key=lambda r: r["llb_knee_ext_peak_velocity"])
    return results


def create_knee_detail_gif(strong_file, weak_file, strong_meta, weak_meta, output_path):
    """Create full-body skeleton GIF with knee extension annotations."""
    # Load C3D
    c_s = ezc3d.c3d(str(strong_file))
    labels_s = c_s["parameters"]["POINT"]["LABELS"]["value"]
    points_s = c_s["data"]["points"]
    rate_s = c_s["parameters"]["POINT"]["RATE"]["value"][0]

    c_w = ezc3d.c3d(str(weak_file))
    labels_w = c_w["parameters"]["POINT"]["LABELS"]["value"]
    points_w = c_w["data"]["points"]
    rate_w = c_w["parameters"]["POINT"]["RATE"]["value"][0]

    n_s = points_s.shape[2]
    n_w = points_w.shape[2]

    fs_s = strong_meta["llb_foot_strike_frame"]
    fs_w = weak_meta["llb_foot_strike_frame"]

    # Knee angles & elbow velocities (from marker dict)
    markers_s, _, _ = load_c3d(str(strong_file))
    markers_w, _, _ = load_c3d(str(weak_file))
    knee_s = compute_knee_flexion(markers_s, "L")
    knee_w = compute_knee_flexion(markers_w, "L")
    vel_s = compute_angular_velocity(knee_s, rate_s)
    vel_w = compute_angular_velocity(knee_w, rate_w)

    # Elbow angular velocity for graph (arm speed)
    elbow_s = compute_elbow_flexion(markers_s, "R")
    elbow_w = compute_elbow_flexion(markers_w, "R")
    elbow_vel_s = compute_angular_velocity(elbow_s, rate_s)
    elbow_vel_w = compute_angular_velocity(elbow_w, rate_w)

    # Time window: 0.5s pre / 0.4s post foot strike
    pre_sec = 0.5
    post_sec = 0.4
    total_sec = pre_sec + post_sec

    start_s = max(0, fs_s - int(pre_sec * rate_s))
    end_s = min(n_s, fs_s + int(post_sec * rate_s))
    start_w = max(0, fs_w - int(pre_sec * rate_w))
    end_w = min(n_w, fs_w + int(post_sec * rate_w))

    n_anim = 108
    idx_s = np.linspace(start_s, end_s - 1, n_anim).astype(int)
    idx_w = np.linspace(start_w, end_w - 1, n_anim).astype(int)
    fs_gif = int(pre_sec / total_sec * n_anim)

    # Graph data — use common time axis so both dots move in sync
    common_time = np.linspace(-pre_sec, post_sec, n_anim)

    # Interpolate elbow angular velocity onto common time axis
    gt_s_raw = (np.arange(start_s, end_s) - fs_s) / rate_s
    gt_w_raw = (np.arange(start_w, end_w) - fs_w) / rate_w
    ev_s_interp = np.interp(common_time, gt_s_raw, elbow_vel_s[start_s:end_s])
    ev_w_interp = np.interp(common_time, gt_w_raw, elbow_vel_w[start_w:end_w])

    # Global bounds for 3D views
    def bounds(pts):
        v = ~np.isnan(pts[0]) & (pts[0] != 0)
        gx, gy, gz = pts[0][v], pts[1][v], pts[2][v]
        cx, cy, cz = np.mean(gx), np.mean(gy), np.mean(gz)
        sp = max(np.ptp(gx), np.ptp(gy), np.ptp(gz)) * 0.6
        return cx, cy, cz, sp

    b_s = bounds(points_s)
    b_w = bounds(points_w)

    # Figure: 3D skeletons on top, graph on bottom (enough room for labels)
    fig = plt.figure(figsize=(18, 12))
    ax_s = fig.add_subplot(2, 2, 1, projection="3d")
    ax_w = fig.add_subplot(2, 2, 2, projection="3d")
    ax_g = fig.add_axes([0.10, 0.06, 0.82, 0.22])

    def draw_skeleton(ax, labels, points, fi, bnd, color, title, speed,
                      knee_angle, knee_vel, is_fs, is_post_fs, is_strong):
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
                c = "#e74c3c"
                lw = 5
            else:
                c = "#2c3e50"
                lw = 1.5
            ax.plot(x, y, z, color=c, linewidth=lw)

        for i, label in enumerate(labels):
            x, y, z = points[0, i, fi], points[1, i, fi], points[2, i, fi]
            if np.isnan(x) or (x == 0 and y == 0 and z == 0):
                continue
            if label in LEAD_LEG_MARKERS:
                mc, ms = "#e74c3c", 30
            else:
                mc, ms = "#3498db", 10
            ax.scatter(x, y, z, c=mc, s=ms, alpha=0.8)

        ax.set_xlim(cx - sp, cx + sp)
        ax.set_ylim(cy - sp, cy + sp)
        ax.set_zlim(cz - sp, cz + sp)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.view_init(elev=15, azim=60)

        # Title with speed
        ax.set_title(f"{title}  ({speed:.1f} mph)", fontsize=14,
                     fontweight="bold", color=color, pad=0)

        # Overlay annotations using fig.text (positioned relative to axes)
        pos = ax.get_position()
        x0 = pos.x0
        y_top = pos.y1

        # Knee angle
        vel_color = "#27ae60" if knee_vel > 0 else "#c0392b"
        vel_label = "\u2191 extending" if knee_vel > 0 else "\u2193 flexing"

        # Clear previous overlay texts (handled by redrawing fig texts in update)

    def draw_graph(ax, frame_num):
        ax.clear()
        # Plot elbow angular velocity over time
        ax.plot(common_time, ev_s_interp, color="#2980b9", linewidth=2.5, label="Strong Block")
        ax.plot(common_time, ev_w_interp, color="#e67e22", linewidth=2.5, label="Weak Block")
        ax.axvline(0, color="#e74c3c", linewidth=2, linestyle="--",
                   alpha=0.7, label="Foot Strike")
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)

        # Dots on the lines at the same X position
        t_now = common_time[frame_num]
        ax.scatter(t_now, ev_s_interp[frame_num],
                   s=150, c="#2980b9", zorder=5, edgecolors="black", linewidths=2)
        ax.scatter(t_now, ev_w_interp[frame_num],
                   s=150, c="#e67e22", zorder=5, edgecolors="black", linewidths=2)

        ax.set_xlabel("Time (seconds)  \u2190 before foot strike | after foot strike \u2192",
                      fontsize=12, fontweight="bold")
        ax.set_ylabel("Elbow Angular Velocity (\u00b0/s)\n\u2191 faster arm action",
                      fontsize=11, fontweight="bold")
        ax.set_title("Elbow Speed Over Time \u2014 does leg block drive arm speed?",
                      fontsize=12, fontweight="bold", color="#2c3e50")
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-pre_sec - 0.02, post_sec + 0.02)

    # Persistent text objects for overlays
    overlay_texts = []

    def update(frame_num):
        # Remove old overlay texts
        for t in overlay_texts:
            t.remove()
        overlay_texts.clear()

        fi_s = idx_s[frame_num]
        fi_w = idx_w[frame_num]
        a_s = knee_s[fi_s]
        a_w = knee_w[fi_w]
        v_s = vel_s[fi_s]
        v_w = vel_w[fi_w]
        is_fs = abs(frame_num - fs_gif) <= 1
        is_post = frame_num > fs_gif + 3

        draw_skeleton(ax_s, labels_s, points_s, fi_s, b_s, "#2980b9",
                      "Strong Block", strong_meta["pitch_speed_mph"],
                      a_s, v_s, is_fs, is_post, True)
        draw_skeleton(ax_w, labels_w, points_w, fi_w, b_w, "#e67e22",
                      "Weak Block", weak_meta["pitch_speed_mph"],
                      a_w, v_w, is_fs, is_post, False)
        draw_graph(ax_g, frame_num)

        # Overlay annotations using fig.text
        # Strong block annotations (left side)
        t1 = fig.text(0.03, 0.56, f"Knee: {a_s:.0f}\u00b0", fontsize=15,
                      fontweight="bold", color="#2c3e50",
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                edgecolor="#2c3e50", alpha=0.95))
        overlay_texts.append(t1)

        vc_s = "#27ae60" if v_s > 0 else "#c0392b"
        vl_s = "\u2191 ext" if v_s > 0 else "\u2193 flex"
        t2 = fig.text(0.03, 0.52, f"{abs(v_s):.0f}\u00b0/s {vl_s}", fontsize=13,
                      fontweight="bold", color=vc_s,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                edgecolor=vc_s, alpha=0.95))
        overlay_texts.append(t2)

        # Weak block annotations (right side)
        t3 = fig.text(0.53, 0.56, f"Knee: {a_w:.0f}\u00b0", fontsize=15,
                      fontweight="bold", color="#2c3e50",
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                edgecolor="#2c3e50", alpha=0.95))
        overlay_texts.append(t3)

        vc_w = "#27ae60" if v_w > 0 else "#c0392b"
        vl_w = "\u2191 ext" if v_w > 0 else "\u2193 flex"
        t4 = fig.text(0.53, 0.52, f"{abs(v_w):.0f}\u00b0/s {vl_w}", fontsize=13,
                      fontweight="bold", color=vc_w,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                edgecolor=vc_w, alpha=0.95))
        overlay_texts.append(t4)

        # Phase label
        if is_fs:
            t5 = fig.text(0.25, 0.28, "\u26a1 FOOT STRIKE", fontsize=14,
                          fontweight="bold", color="white",
                          bbox=dict(boxstyle="round,pad=0.4", facecolor="#e74c3c", alpha=0.95))
            overlay_texts.append(t5)
            t6 = fig.text(0.68, 0.28, "\u26a1 FOOT STRIKE", fontsize=14,
                          fontweight="bold", color="white",
                          bbox=dict(boxstyle="round,pad=0.4", facecolor="#e74c3c", alpha=0.95))
            overlay_texts.append(t6)

        # Post-strike explanations
        if is_post:
            t7 = fig.text(0.35, 0.28, "Knee extends \u2192 faster arm \u2191",
                          fontsize=11, fontweight="bold", color="#27ae60",
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1",
                                    edgecolor="#27ae60", alpha=0.95))
            overlay_texts.append(t7)
            t8 = fig.text(0.78, 0.28, "Knee stays bent \u2192 slower arm \u2193",
                          fontsize=11, fontweight="bold", color="#c0392b",
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="#fdedec",
                                    edgecolor="#c0392b", alpha=0.95))
            overlay_texts.append(t8)

        fig.suptitle("Lead Leg Block \u2192 Arm Speed\n"
                     "Red = lead leg | Graph = elbow angular velocity",
                     fontsize=14, fontweight="bold", y=0.99)

    print("  Rendering animation...")
    anim = animation.FuncAnimation(fig, update, frames=n_anim, interval=80)
    anim.save(str(output_path), writer="pillow", fps=12)
    plt.close(fig)
    print(f"  Knee detail GIF saved: {output_path} ({n_anim} frames)")


def main():
    parser = argparse.ArgumentParser(description="LLB knee detail GIF")
    parser.add_argument("--download", type=int, default=40)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting LLB features...")
    results = collect_llb_candidates(args.download)

    if len(results) < 2:
        print(f"Need at least 2 samples, got {len(results)}")
        return

    weak = results[0]
    strong = results[-1]

    print(f"\n  Strong: {strong['filename']} "
          f"({strong['pitch_speed_mph']:.1f} mph, "
          f"knee ext vel={strong['llb_knee_ext_peak_velocity']:.0f} deg/s)")
    print(f"  Weak:   {weak['filename']} "
          f"({weak['pitch_speed_mph']:.1f} mph, "
          f"knee ext vel={weak['llb_knee_ext_peak_velocity']:.0f} deg/s)")

    print("\nGenerating knee detail GIF...")
    create_knee_detail_gif(
        RAW_DIR / strong["filename"],
        RAW_DIR / weak["filename"],
        strong, weak,
        OUTPUT_DIR / "llb_knee_detail.gif",
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
