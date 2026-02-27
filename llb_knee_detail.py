"""Lead Leg Block — Knee extension detail GIF with annotations.

Generates a zoomed-in 2D side-view GIF focusing on the lead leg,
showing real-time knee angle, angular velocity, and visual angle arc.
Side-by-side comparison of strong vs weak lead leg block.

Usage:
    python llb_knee_detail.py --download 40

Output:
    data/output/llb_knee_detail.gif
"""

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from skeleton_analysis import (
    compute_angular_velocity,
    compute_knee_flexion,
    compute_lead_leg_block_features,
    load_c3d,
)
from statcast_correlation import download_additional_samples, parse_pitching_filename

OUTPUT_DIR = Path("data/output")
RAW_DIR = Path("data/raw")


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


def to_side_view(pt3d):
    """Project 3D marker to 2D side view (Y=forward, Z=up)."""
    return np.array([pt3d[1], pt3d[2]])


def create_knee_detail_gif(strong_file, weak_file, strong_meta, weak_meta, output_path):
    """Create annotated knee extension detail GIF."""
    markers_s, rate_s, n_s = load_c3d(str(strong_file))
    markers_w, rate_w, n_w = load_c3d(str(weak_file))

    fs_s = strong_meta["llb_foot_strike_frame"]
    fs_w = weak_meta["llb_foot_strike_frame"]

    # Compute full knee angle & velocity time series
    knee_s = compute_knee_flexion(markers_s, "L")
    knee_w = compute_knee_flexion(markers_w, "L")
    vel_s = compute_angular_velocity(knee_s, rate_s)
    vel_w = compute_angular_velocity(knee_w, rate_w)

    # Time-aligned window: 0.3s before → 0.5s after foot strike
    pre_sec = 0.3
    post_sec = 0.5
    total_sec = pre_sec + post_sec

    start_s = max(0, fs_s - int(pre_sec * rate_s))
    end_s = min(n_s, fs_s + int(post_sec * rate_s))
    start_w = max(0, fs_w - int(pre_sec * rate_w))
    end_w = min(n_w, fs_w + int(post_sec * rate_w))

    n_anim = 96
    idx_s = np.linspace(start_s, end_s - 1, n_anim).astype(int)
    idx_w = np.linspace(start_w, end_w - 1, n_anim).astype(int)
    fs_gif = int(pre_sec / total_sec * n_anim)

    # Precompute graph data (time relative to foot strike)
    graph_time_s = (np.arange(start_s, end_s) - fs_s) / rate_s
    graph_time_w = (np.arange(start_w, end_w) - fs_w) / rate_w
    graph_knee_s = knee_s[start_s:end_s]
    graph_knee_w = knee_w[start_w:end_w]

    # Map GIF frame → graph index
    graph_idx_s = np.linspace(0, len(graph_knee_s) - 1, n_anim).astype(int)
    graph_idx_w = np.linspace(0, len(graph_knee_w) - 1, n_anim).astype(int)

    # Figure layout
    fig = plt.figure(figsize=(16, 9))
    ax_leg_s = fig.add_axes([0.02, 0.32, 0.46, 0.62])
    ax_leg_w = fig.add_axes([0.52, 0.32, 0.46, 0.62])
    ax_graph = fig.add_axes([0.08, 0.05, 0.84, 0.24])

    def get_leg_pts(markers, frame):
        """Get hip, knee, ankle 2D points for a frame."""
        pts = {}
        for name in ("LASI", "LKNE", "LANK", "LHEE", "LTOE"):
            m = markers.get(name)
            if m is not None:
                pts[name] = to_side_view(m[:, frame])
        return pts

    def draw_leg_panel(ax, markers, frame, knee_angle, knee_vel,
                       title, color, speed_mph, is_fs, is_post_fs, is_strong):
        ax.clear()
        pts = get_leg_pts(markers, frame)

        if "LASI" not in pts or "LKNE" not in pts or "LANK" not in pts:
            ax.text(0.5, 0.5, "Markers missing", transform=ax.transAxes, ha="center")
            return

        hip = pts["LASI"]
        knee = pts["LKNE"]
        ankle = pts["LANK"]

        # Thigh (hip → knee): dark gray
        ax.plot([hip[0], knee[0]], [hip[1], knee[1]],
                color="#555555", linewidth=8, solid_capstyle="round", zorder=2)
        # Shin (knee → ankle): red
        ax.plot([knee[0], ankle[0]], [knee[1], ankle[1]],
                color="#e74c3c", linewidth=8, solid_capstyle="round", zorder=2)

        # Foot segments
        if "LHEE" in pts:
            heel = pts["LHEE"]
            ax.plot([ankle[0], heel[0]], [ankle[1], heel[1]],
                    color="#e74c3c", linewidth=5, solid_capstyle="round", zorder=2)
        if "LTOE" in pts:
            toe = pts["LTOE"]
            ax.plot([ankle[0], toe[0]], [ankle[1], toe[1]],
                    color="#e74c3c", linewidth=5, solid_capstyle="round", zorder=2)

        # Joint dots
        ax.scatter(*hip, s=120, c="#555555", zorder=3, edgecolors="black", linewidths=1.5)
        ax.scatter(*knee, s=160, c=color, zorder=3, edgecolors="black", linewidths=1.5)
        ax.scatter(*ankle, s=120, c="#e74c3c", zorder=3, edgecolors="black", linewidths=1.5)

        # Joint labels
        ax.annotate("Hip", hip, textcoords="offset points", xytext=(-25, 10),
                     fontsize=9, color="#555555", fontweight="bold")
        ax.annotate("Knee", knee, textcoords="offset points", xytext=(10, -15),
                     fontsize=9, color=color, fontweight="bold")
        ax.annotate("Ankle", ankle, textcoords="offset points", xytext=(10, -15),
                     fontsize=9, color="#e74c3c", fontweight="bold")

        # Angle arc at knee — radius proportional to thigh length
        thigh_len = np.linalg.norm(hip - knee)
        arc_radius = thigh_len * 0.3
        v_thigh = hip - knee
        v_shin = ankle - knee
        ang_thigh = np.degrees(np.arctan2(v_thigh[1], v_thigh[0]))
        ang_shin = np.degrees(np.arctan2(v_shin[1], v_shin[0]))
        theta1, theta2 = sorted([ang_thigh, ang_shin])
        arc = patches.Arc(knee, arc_radius * 2, arc_radius * 2,
                          angle=0, theta1=theta1, theta2=theta2,
                          color=color, linewidth=2.5, linestyle="--", zorder=4)
        ax.add_patch(arc)

        # Title
        ax.set_title(f"{title}  ({speed_mph:.1f} mph)", fontsize=15,
                     fontweight="bold", color=color, pad=10)

        # Knee angle box
        ax.text(0.05, 0.93, f"Knee Angle: {knee_angle:.1f}\u00b0",
                transform=ax.transAxes, fontsize=17, fontweight="bold",
                color="#2c3e50",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#2c3e50", alpha=0.95))

        # Velocity box
        vel_color = "#27ae60" if knee_vel > 0 else "#c0392b"
        vel_arrow = "\u2191 extending" if knee_vel > 0 else "\u2193 flexing"
        ax.text(0.05, 0.80, f"Velocity: {abs(knee_vel):.0f} \u00b0/s {vel_arrow}",
                transform=ax.transAxes, fontsize=14, fontweight="bold",
                color=vel_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=vel_color, alpha=0.95))

        # Phase label
        if is_fs:
            ax.text(0.5, 0.05, "\u26a1 FOOT STRIKE \u26a1", transform=ax.transAxes,
                    fontsize=18, fontweight="bold", color="white", ha="center",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#e74c3c", alpha=0.95))

        # Explanation (post foot strike only)
        if is_post_fs and is_strong:
            ax.text(0.95, 0.15,
                    "Knee snaps straight\n\u2192 energy transfers\n   to upper body",
                    transform=ax.transAxes, fontsize=11, ha="right",
                    color="#27ae60", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#eafaf1",
                              edgecolor="#27ae60", alpha=0.95))
        elif is_post_fs and not is_strong:
            ax.text(0.95, 0.15,
                    "Knee stays bent\n\u2192 energy absorbed\n   by the leg",
                    transform=ax.transAxes, fontsize=11, ha="right",
                    color="#c0392b", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fdedec",
                              edgecolor="#c0392b", alpha=0.95))

        # Auto-scale to leg area with padding
        all_pts = np.array([hip, knee, ankle])
        if "LHEE" in pts:
            all_pts = np.vstack([all_pts, pts["LHEE"]])
        if "LTOE" in pts:
            all_pts = np.vstack([all_pts, pts["LTOE"]])
        cx = np.mean(all_pts[:, 0])
        cy = np.mean(all_pts[:, 1])
        span = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.75 + 80
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_aspect("equal")
        ax.axis("off")

    def draw_graph(ax, anim_frame):
        ax.clear()

        # Plot both knee angle curves
        ax.plot(graph_time_s, graph_knee_s, color="#2980b9", linewidth=2.5,
                label="Strong Block", zorder=2)
        ax.plot(graph_time_w, graph_knee_w, color="#e67e22", linewidth=2.5,
                label="Weak Block", zorder=2)

        # Foot strike vertical line
        ax.axvline(0, color="#e74c3c", linewidth=2, linestyle="--",
                   alpha=0.7, label="Foot Strike", zorder=1)

        # Current position dots (synced to animation frame)
        gi_s = graph_idx_s[anim_frame]
        gi_w = graph_idx_w[anim_frame]
        ax.scatter(graph_time_s[gi_s], graph_knee_s[gi_s],
                   s=120, c="#2980b9", zorder=5, edgecolors="black", linewidths=1.5)
        ax.scatter(graph_time_w[gi_w], graph_knee_w[gi_w],
                   s=120, c="#e67e22", zorder=5, edgecolors="black", linewidths=1.5)

        ax.set_xlabel("Time from Foot Strike (s)", fontsize=11)
        ax.set_ylabel("Knee Angle (\u00b0)", fontsize=11)
        ax.set_title("Knee Angle Over Time \u2014 higher = straighter leg",
                      fontsize=11, color="#2c3e50", fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-pre_sec - 0.02, post_sec + 0.02)

    def update(frame_num):
        fi_s = idx_s[frame_num]
        fi_w = idx_w[frame_num]

        angle_s = knee_s[fi_s]
        angle_w = knee_w[fi_w]
        v_s = vel_s[fi_s]
        v_w = vel_w[fi_w]

        is_fs = abs(frame_num - fs_gif) <= 1
        is_post_fs = frame_num > fs_gif + 3

        draw_leg_panel(ax_leg_s, markers_s, fi_s, angle_s, v_s,
                       "Strong Block", "#2980b9", strong_meta["pitch_speed_mph"],
                       is_fs, is_post_fs, True)
        draw_leg_panel(ax_leg_w, markers_w, fi_w, angle_w, v_w,
                       "Weak Block", "#e67e22", weak_meta["pitch_speed_mph"],
                       is_fs, is_post_fs, False)
        draw_graph(ax_graph, frame_num)

        fig.suptitle("Lead Leg Block \u2014 Knee Extension Detail\n"
                     "Gray = thigh \u2502 Red = shin & foot",
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
