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
                "llb_knee_angle_at_strike": llb.get("llb_knee_angle_at_strike"),
            })
        except Exception as e:
            print(f"  Skip {fpath.name}: {e}")

    results.sort(key=lambda r: r["llb_knee_ext_peak_velocity"])
    return results


def create_knee_detail_gif(strong_file, weak_file, strong_meta, weak_meta, output_path):
    """Create annotated knee extension detail GIF."""
    markers_s, rate_s, n_s = load_c3d(str(strong_file))
    markers_w, rate_w, n_w = load_c3d(str(weak_file))

    fs_s = strong_meta["llb_foot_strike_frame"]
    fs_w = weak_meta["llb_foot_strike_frame"]

    # Compute knee angles for full timeline
    knee_s = compute_knee_flexion(markers_s, "L")
    knee_w = compute_knee_flexion(markers_w, "L")
    vel_s = compute_angular_velocity(knee_s, rate_s)
    vel_w = compute_angular_velocity(knee_w, rate_w)

    # Time-aligned window: 0.3s before → 0.5s after foot strike
    pre_sec = 0.3
    post_sec = 0.5

    start_s = max(0, fs_s - int(pre_sec * rate_s))
    end_s = min(n_s, fs_s + int(post_sec * rate_s))
    start_w = max(0, fs_w - int(pre_sec * rate_w))
    end_w = min(n_w, fs_w + int(post_sec * rate_w))

    n_anim = 96
    idx_s = np.linspace(start_s, end_s - 1, n_anim).astype(int)
    idx_w = np.linspace(start_w, end_w - 1, n_anim).astype(int)
    fs_gif = int(pre_sec / (pre_sec + post_sec) * n_anim)

    # Get leg marker positions (LASI=hip, LKNE=knee, LANK=ankle, LHEE=heel, LTOE=toe)
    def get_leg_markers(markers, frame):
        pts = {}
        for name in ("LASI", "LKNE", "LANK", "LHEE", "LTOE"):
            m = markers.get(name)
            if m is not None:
                pts[name] = m[:, frame]
        return pts

    # Project 3D to 2D side view (use X=forward, Z=up)
    def to_2d(pt3d):
        return np.array([pt3d[0], pt3d[2]])

    def draw_angle_arc(ax, hip_2d, knee_2d, ankle_2d, angle_deg, color):
        """Draw an arc at the knee showing the flexion angle."""
        v1 = hip_2d - knee_2d
        v2 = ankle_2d - knee_2d
        ang1 = np.degrees(np.arctan2(v1[1], v1[0]))
        ang2 = np.degrees(np.arctan2(v2[1], v2[0]))
        radius = 40
        arc = patches.Arc(knee_2d, radius * 2, radius * 2,
                          angle=0, theta1=min(ang1, ang2), theta2=max(ang1, ang2),
                          color=color, linewidth=2, linestyle="--")
        ax.add_patch(arc)

    fig = plt.figure(figsize=(16, 9))
    # Layout: top row = 2 leg views, bottom = knee angle time series
    ax_leg_s = fig.add_axes([0.02, 0.35, 0.46, 0.60])
    ax_leg_w = fig.add_axes([0.52, 0.35, 0.46, 0.60])
    ax_graph = fig.add_axes([0.08, 0.06, 0.84, 0.26])

    # Precompute time-series data for graph
    time_s = (np.arange(start_s, end_s) - fs_s) / rate_s
    time_w = (np.arange(start_w, end_w) - fs_w) / rate_w
    knee_slice_s = knee_s[start_s:end_s]
    knee_slice_w = knee_w[start_w:end_w]

    def draw_leg(ax, markers, frame, knee_angle, knee_vel, title, color, meta, is_fs):
        ax.clear()
        pts = get_leg_markers(markers, frame)
        if len(pts) < 3:
            return

        hip = to_2d(pts["LASI"])
        knee = to_2d(pts["LKNE"])
        ankle = to_2d(pts["LANK"])

        # Draw ground line
        ground_z = min(hip[1], knee[1], ankle[1]) - 30
        if "LHEE" in pts:
            ground_z = min(ground_z, to_2d(pts["LHEE"])[1] - 10)

        # Draw leg segments
        ax.plot([hip[0], knee[0]], [hip[1], knee[1]],
                color="#2c3e50", linewidth=6, solid_capstyle="round")
        ax.plot([knee[0], ankle[0]], [knee[1], ankle[1]],
                color="#e74c3c", linewidth=6, solid_capstyle="round")

        # Draw foot if available
        if "LHEE" in pts and "LTOE" in pts:
            heel = to_2d(pts["LHEE"])
            toe = to_2d(pts["LTOE"])
            ax.plot([ankle[0], heel[0]], [ankle[1], heel[1]],
                    color="#e74c3c", linewidth=4, solid_capstyle="round")
            ax.plot([ankle[0], toe[0]], [ankle[1], toe[1]],
                    color="#e74c3c", linewidth=4, solid_capstyle="round")

        # Joint markers
        for pt, sz in [(hip, 80), (knee, 100), (ankle, 80)]:
            ax.scatter(pt[0], pt[1], s=sz, c=color, zorder=5, edgecolors="black", linewidths=1)

        # Angle arc at knee
        draw_angle_arc(ax, hip, knee, ankle, knee_angle, color)

        # Annotations
        speed = meta["pitch_speed_mph"]
        ax.set_title(f"{title}  ({speed:.1f} mph)", fontsize=14, fontweight="bold",
                     color=color)

        # Knee angle and velocity text
        vel_color = "#27ae60" if knee_vel > 0 else "#e74c3c"
        vel_arrow = "\u2191" if knee_vel > 0 else "\u2193"

        info_text = f"Knee Angle: {knee_angle:.1f}\u00b0"
        ax.text(0.05, 0.92, info_text, transform=ax.transAxes,
                fontsize=16, fontweight="bold", color="#2c3e50",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

        vel_text = f"Ext Velocity: {knee_vel:.0f} \u00b0/s {vel_arrow}"
        ax.text(0.05, 0.78, vel_text, transform=ax.transAxes,
                fontsize=14, fontweight="bold", color=vel_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

        # Phase label
        if is_fs:
            phase = "FOOT STRIKE"
            phase_color = "#e74c3c"
        else:
            phase = ""
            phase_color = "#7f8c8d"

        if phase:
            ax.text(0.5, 0.05, phase, transform=ax.transAxes,
                    fontsize=16, fontweight="bold", color="white", ha="center",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=phase_color, alpha=0.9))

        # Explanation text (only after foot strike)
        if frame > fs_s and title == "Strong Block":
            ax.text(0.95, 0.78, "Knee extends\nrapidly \u2192 energy\ntransfers up",
                    transform=ax.transAxes, fontsize=10, ha="right", color="#27ae60",
                    style="italic",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1", alpha=0.9))
        elif frame > fs_w and title == "Weak Block":
            ax.text(0.95, 0.78, "Knee stays bent\n\u2192 energy\nabsorbed by leg",
                    transform=ax.transAxes, fontsize=10, ha="right", color="#e74c3c",
                    style="italic",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fdedec", alpha=0.9))

        ax.set_aspect("equal")
        ax.axis("off")

        # Set bounds around leg
        all_pts = np.array([hip, knee, ankle])
        cx, cy = np.mean(all_pts, axis=0)
        span = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.8 + 50
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)

    def draw_graph(ax, anim_frame):
        ax.clear()
        ax.plot(time_s, knee_slice_s, color="#2980b9", linewidth=2, label="Strong Block")
        ax.plot(time_w, knee_slice_w, color="#e67e22", linewidth=2, label="Weak Block")
        ax.axvline(0, color="#e74c3c", linewidth=1.5, linestyle="--", alpha=0.7, label="Foot Strike")

        # Current time marker
        t_cur = (anim_frame - fs_gif) / n_anim * (pre_sec + post_sec) - pre_sec
        cur_s_idx = min(max(0, int((anim_frame / n_anim) * len(knee_slice_s))), len(knee_slice_s) - 1)
        cur_w_idx = min(max(0, int((anim_frame / n_anim) * len(knee_slice_w))), len(knee_slice_w) - 1)

        if cur_s_idx < len(knee_slice_s):
            ax.scatter(time_s[cur_s_idx], knee_slice_s[cur_s_idx],
                       s=80, c="#2980b9", zorder=5, edgecolors="black")
        if cur_w_idx < len(knee_slice_w):
            ax.scatter(time_w[cur_w_idx], knee_slice_w[cur_w_idx],
                       s=80, c="#e67e22", zorder=5, edgecolors="black")

        ax.set_xlabel("Time from Foot Strike (s)", fontsize=11)
        ax.set_ylabel("Knee Angle (\u00b0)", fontsize=11)
        ax.set_title("Knee Angle Over Time \u2014 higher = more extended (straighter)",
                      fontsize=11, color="#2c3e50")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-pre_sec, post_sec)

    def update(frame_num):
        fi_s = idx_s[frame_num]
        fi_w = idx_w[frame_num]

        angle_s = knee_s[fi_s]
        angle_w = knee_w[fi_w]
        v_s = vel_s[fi_s]
        v_w = vel_w[fi_w]

        is_fs = abs(frame_num - fs_gif) <= 1

        draw_leg(ax_leg_s, markers_s, fi_s, angle_s, v_s,
                 "Strong Block", "#2980b9", strong_meta, is_fs)
        draw_leg(ax_leg_w, markers_w, fi_w, angle_w, v_w,
                 "Weak Block", "#e67e22", weak_meta, is_fs)
        draw_graph(ax_graph, frame_num)

        fig.suptitle("Lead Leg Block \u2014 Knee Extension Detail\n"
                     "Red segments = shin & foot (lead leg below knee)",
                     fontsize=13, fontweight="bold", y=0.99)

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
