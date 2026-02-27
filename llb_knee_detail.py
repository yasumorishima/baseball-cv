"""Lead Leg Block — Knee extension detail GIF with annotations.

Draws a schematic leg diagram based on actual knee angle measurements,
avoiding coordinate-system issues with raw marker projection.
Side-by-side strong vs weak comparison with time-series graph.

Usage:
    python llb_knee_detail.py --download 40

Output:
    data/output/llb_knee_detail.gif
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Arc

from skeleton_analysis import (
    compute_angular_velocity,
    compute_knee_flexion,
    compute_lead_leg_block_features,
    load_c3d,
)
from statcast_correlation import download_additional_samples, parse_pitching_filename

OUTPUT_DIR = Path("data/output")
RAW_DIR = Path("data/raw")

# Schematic leg dimensions (arbitrary units for diagram)
THIGH_LEN = 1.0
SHIN_LEN = 0.95


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


def knee_angle_to_leg_points(angle_deg):
    """Convert knee angle to schematic leg coordinates.

    Returns hip, knee, ankle positions for a diagram where:
    - Hip is at top
    - Thigh goes down-right
    - Shin angle relative to thigh is the measured knee angle
    - 180° = fully straight, <180° = bent

    Returns (hip, knee, ankle) as 2D points.
    """
    # Thigh direction: slightly angled forward (down-right)
    thigh_angle_rad = np.radians(-80)  # near-vertical, slight forward lean
    hip = np.array([0.0, 2.0])
    knee = hip + THIGH_LEN * np.array([np.cos(thigh_angle_rad), np.sin(thigh_angle_rad)])

    # Shin direction: rotated from thigh by (180 - knee_angle) toward the front
    # When knee_angle=180, shin is straight extension of thigh
    # When knee_angle<180, shin bends forward
    bend_rad = np.radians(180 - angle_deg)
    shin_angle_rad = thigh_angle_rad + bend_rad
    ankle = knee + SHIN_LEN * np.array([np.cos(shin_angle_rad), np.sin(shin_angle_rad)])

    return hip, knee, ankle


def create_knee_detail_gif(strong_file, weak_file, strong_meta, weak_meta, output_path):
    """Create annotated knee extension detail GIF."""
    markers_s, rate_s, n_s = load_c3d(str(strong_file))
    markers_w, rate_w, n_w = load_c3d(str(weak_file))

    fs_s = strong_meta["llb_foot_strike_frame"]
    fs_w = weak_meta["llb_foot_strike_frame"]

    # Knee angles & velocities
    knee_s = compute_knee_flexion(markers_s, "L")
    knee_w = compute_knee_flexion(markers_w, "L")
    vel_s = compute_angular_velocity(knee_s, rate_s)
    vel_w = compute_angular_velocity(knee_w, rate_w)

    # Time window: 0.3s before → 0.5s after foot strike
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

    # Graph data
    graph_time_s = (np.arange(start_s, end_s) - fs_s) / rate_s
    graph_time_w = (np.arange(start_w, end_w) - fs_w) / rate_w
    graph_knee_s = knee_s[start_s:end_s]
    graph_knee_w = knee_w[start_w:end_w]
    graph_idx_s = np.linspace(0, len(graph_knee_s) - 1, n_anim).astype(int)
    graph_idx_w = np.linspace(0, len(graph_knee_w) - 1, n_anim).astype(int)

    # Figure
    fig = plt.figure(figsize=(16, 10))
    ax_s = fig.add_axes([0.02, 0.33, 0.46, 0.60])
    ax_w = fig.add_axes([0.52, 0.33, 0.46, 0.60])
    ax_g = fig.add_axes([0.08, 0.05, 0.84, 0.25])

    def draw_leg_diagram(ax, angle_deg, knee_vel, title, color, speed_mph,
                         is_fs, is_post_fs, is_strong):
        ax.clear()

        hip, knee, ankle = knee_angle_to_leg_points(angle_deg)

        # Ground line
        ground_y = ankle[1] - 0.15
        ax.axhline(ground_y, color="#bdc3c7", linewidth=2, linestyle="-", zorder=0)
        ax.fill_between([-1.5, 1.5], ground_y - 0.3, ground_y,
                         color="#ecf0f1", alpha=0.5, zorder=0)

        # Thigh (gray)
        ax.plot([hip[0], knee[0]], [hip[1], knee[1]],
                color="#555555", linewidth=12, solid_capstyle="round", zorder=2)
        # Shin (red)
        ax.plot([knee[0], ankle[0]], [knee[1], ankle[1]],
                color="#e74c3c", linewidth=12, solid_capstyle="round", zorder=2)

        # Foot stub
        foot_dir = np.array([0.3, -0.05])
        foot_end = ankle + foot_dir
        ax.plot([ankle[0], foot_end[0]], [ankle[1], foot_end[1]],
                color="#e74c3c", linewidth=8, solid_capstyle="round", zorder=2)

        # Joint circles
        ax.scatter(*hip, s=250, c="#555555", zorder=3, edgecolors="black", linewidths=2)
        ax.scatter(*knee, s=350, c=color, zorder=3, edgecolors="black", linewidths=2)
        ax.scatter(*ankle, s=250, c="#e74c3c", zorder=3, edgecolors="black", linewidths=2)

        # Labels
        ax.annotate("Hip", hip, xytext=(-0.35, 0.08), fontsize=11,
                     fontweight="bold", color="#555555")
        ax.annotate("Knee", knee, xytext=(0.12, 0.05), fontsize=11,
                     fontweight="bold", color=color)
        ax.annotate("Ankle", ankle, xytext=(0.12, -0.05), fontsize=11,
                     fontweight="bold", color="#e74c3c")

        # Angle arc at knee
        v_thigh = hip - knee
        v_shin = ankle - knee
        ang_thigh = np.degrees(np.arctan2(v_thigh[1], v_thigh[0]))
        ang_shin = np.degrees(np.arctan2(v_shin[1], v_shin[0]))
        theta1, theta2 = sorted([ang_thigh, ang_shin])
        arc = Arc(knee, 0.5, 0.5, angle=0, theta1=theta1, theta2=theta2,
                  color=color, linewidth=3, linestyle="--", zorder=4)
        ax.add_patch(arc)

        # Angle value near arc
        mid_angle = np.radians((theta1 + theta2) / 2)
        label_pos = knee + 0.35 * np.array([np.cos(mid_angle), np.sin(mid_angle)])
        ax.text(label_pos[0], label_pos[1], f"{angle_deg:.0f}\u00b0",
                fontsize=14, fontweight="bold", color=color, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=color, alpha=0.9))

        # Title
        ax.set_title(f"{title}  ({speed_mph:.1f} mph)", fontsize=16,
                     fontweight="bold", color=color, pad=12)

        # Knee angle box
        ax.text(0.05, 0.93, f"Knee Angle: {angle_deg:.1f}\u00b0",
                transform=ax.transAxes, fontsize=18, fontweight="bold",
                color="#2c3e50",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#2c3e50", alpha=0.95))

        # Velocity box
        vel_color = "#27ae60" if knee_vel > 0 else "#c0392b"
        vel_label = "\u2191 extending" if knee_vel > 0 else "\u2193 flexing"
        ax.text(0.05, 0.80, f"Velocity: {abs(knee_vel):.0f} \u00b0/s {vel_label}",
                transform=ax.transAxes, fontsize=15, fontweight="bold",
                color=vel_color,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=vel_color, alpha=0.95))

        # Phase label
        if is_fs:
            ax.text(0.5, 0.03, "\u26a1 FOOT STRIKE \u26a1", transform=ax.transAxes,
                    fontsize=18, fontweight="bold", color="white", ha="center",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#e74c3c", alpha=0.95))

        # Explanation
        if is_post_fs and is_strong:
            ax.text(0.95, 0.55,
                    "Knee snaps straight\n\u2192 energy transfers\n   to upper body \u2191",
                    transform=ax.transAxes, fontsize=12, ha="right",
                    color="#27ae60", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#eafaf1",
                              edgecolor="#27ae60", alpha=0.95))
        elif is_post_fs and not is_strong:
            ax.text(0.95, 0.55,
                    "Knee stays bent\n\u2192 energy absorbed\n   by the leg \u2193",
                    transform=ax.transAxes, fontsize=12, ha="right",
                    color="#c0392b", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fdedec",
                              edgecolor="#c0392b", alpha=0.95))

        ax.set_xlim(-1.2, 1.5)
        ax.set_ylim(-0.3, 2.3)
        ax.set_aspect("equal")
        ax.axis("off")

    def draw_graph(ax, frame_num):
        ax.clear()
        ax.plot(graph_time_s, graph_knee_s, color="#2980b9", linewidth=2.5,
                label="Strong Block", zorder=2)
        ax.plot(graph_time_w, graph_knee_w, color="#e67e22", linewidth=2.5,
                label="Weak Block", zorder=2)
        ax.axvline(0, color="#e74c3c", linewidth=2, linestyle="--",
                   alpha=0.7, label="Foot Strike", zorder=1)

        gi_s = graph_idx_s[frame_num]
        gi_w = graph_idx_w[frame_num]
        ax.scatter(graph_time_s[gi_s], graph_knee_s[gi_s],
                   s=150, c="#2980b9", zorder=5, edgecolors="black", linewidths=2)
        ax.scatter(graph_time_w[gi_w], graph_knee_w[gi_w],
                   s=150, c="#e67e22", zorder=5, edgecolors="black", linewidths=2)

        ax.set_xlabel("Time from Foot Strike (s)", fontsize=12)
        ax.set_ylabel("Knee Angle (\u00b0)", fontsize=12)
        ax.set_title("Knee Angle Over Time \u2014 higher = straighter leg",
                      fontsize=12, color="#2c3e50", fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
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

        draw_leg_diagram(ax_s, angle_s, v_s, "Strong Block", "#2980b9",
                         strong_meta["pitch_speed_mph"], is_fs, is_post_fs, True)
        draw_leg_diagram(ax_w, angle_w, v_w, "Weak Block", "#e67e22",
                         weak_meta["pitch_speed_mph"], is_fs, is_post_fs, False)
        draw_graph(ax_g, frame_num)

        fig.suptitle("Lead Leg Block \u2014 Knee Extension Detail\n"
                     "Gray = thigh \u2502 Red = shin & foot  "
                     "\u2502  180\u00b0 = fully straight",
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
