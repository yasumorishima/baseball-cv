"""Lead Leg Block comparison GIF — side-by-side skeleton animation.

Downloads pitching C3D files from Driveline OBP, extracts LLB features,
then generates a side-by-side GIF comparing the strongest vs weakest
lead leg block with highlighted lead leg.

Usage:
    python llb_comparison.py                # Default: download 40, pick best/worst
    python llb_comparison.py --download 20  # Fewer samples

Output:
    data/output/llb_comparison.gif
"""

import argparse
from pathlib import Path

import ezc3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from skeleton_analysis import compute_lead_leg_block_features, load_c3d
from skeleton_c3d import BODY_CONNECTIONS, get_marker_index
from statcast_correlation import download_additional_samples, parse_pitching_filename

OUTPUT_DIR = Path("data/output")
RAW_DIR = Path("data/raw")

# Lead leg markers (left side = lead leg for right-handed pitcher)
LEAD_LEG_MARKERS = {
    "LASI", "LKNE", "LMKNE", "LANK", "LMANK", "LHEE", "LTOE", "LTHI", "LTIB",
}

LEAD_LEG_CONNECTIONS = [
    ("LASI", "LKNE"), ("LKNE", "LMKNE"), ("LKNE", "LANK"),
    ("LANK", "LMANK"), ("LANK", "LHEE"), ("LANK", "LTOE"),
    ("LASI", "LTHI"), ("LKNE", "LTIB"),
]


def is_lead_leg_connection(m1, m2):
    """Check if a connection involves the lead leg."""
    return (m1, m2) in LEAD_LEG_CONNECTIONS or (m2, m1) in LEAD_LEG_CONNECTIONS


def collect_llb_features(n_samples=40):
    """Download C3D files and extract LLB features, return sorted list."""
    download_additional_samples("pitching", n_samples)

    results = []
    for fpath in sorted(RAW_DIR.glob("*.c3d")):
        meta = parse_pitching_filename(fpath.name)
        if meta is None:
            continue

        try:
            markers, rate, n_frames = load_c3d(str(fpath))
            llb = compute_lead_leg_block_features(markers, rate, side="L")
            if not llb:
                continue

            results.append({
                "filename": fpath.name,
                "pitch_speed_mph": meta["pitch_speed_mph"],
                "llb_ankle_velocity_delta": llb.get("llb_ankle_velocity_delta", 0),
                "llb_knee_ext_peak_velocity": llb.get("llb_knee_ext_peak_velocity", 0),
                "llb_foot_strike_frame": llb.get("llb_foot_strike_frame"),
            })
        except Exception as e:
            print(f"  Skip {fpath.name}: {e}")

    return results


def create_comparison_gif(strong_file, weak_file, strong_meta, weak_meta, output_path):
    """Create side-by-side GIF comparing strong vs weak lead leg block."""
    # Load both C3D files
    c_strong = ezc3d.c3d(str(strong_file))
    labels_s = c_strong["parameters"]["POINT"]["LABELS"]["value"]
    points_s = c_strong["data"]["points"]
    rate_s = c_strong["parameters"]["POINT"]["RATE"]["value"][0]

    c_weak = ezc3d.c3d(str(weak_file))
    labels_w = c_weak["parameters"]["POINT"]["LABELS"]["value"]
    points_w = c_weak["data"]["points"]
    rate_w = c_weak["parameters"]["POINT"]["RATE"]["value"][0]

    n_s = points_s.shape[2]
    n_w = points_w.shape[2]

    fs_s = strong_meta["llb_foot_strike_frame"]
    fs_w = weak_meta["llb_foot_strike_frame"]

    # Align animations around foot strike: show from 30% before to 30% after
    def get_window(n_frames, fs_frame):
        pre = int(n_frames * 0.30)
        post = int(n_frames * 0.30)
        start = max(0, fs_frame - pre)
        end = min(n_frames, fs_frame + post)
        return start, end, fs_frame - start

    start_s, end_s, fs_rel_s = get_window(n_s, fs_s)
    start_w, end_w, fs_rel_w = get_window(n_w, fs_w)

    max_frames = 120
    frames_s = list(range(start_s, end_s))
    frames_w = list(range(start_w, end_w))

    # Subsample to equal length
    n_anim = min(len(frames_s), len(frames_w), max_frames)
    step_s = max(1, len(frames_s) // n_anim)
    step_w = max(1, len(frames_w) // n_anim)
    frames_s = frames_s[::step_s][:n_anim]
    frames_w = frames_w[::step_w][:n_anim]

    # Ensure same number of frames
    n_anim = min(len(frames_s), len(frames_w))
    frames_s = frames_s[:n_anim]
    frames_w = frames_w[:n_anim]

    fig, (ax_strong, ax_weak) = plt.subplots(1, 2, figsize=(16, 8),
                                              subplot_kw={"projection": "3d"})

    def compute_bounds(points):
        valid = ~np.isnan(points[0]) & (points[0] != 0)
        gx, gy, gz = points[0][valid], points[1][valid], points[2][valid]
        cx, cy, cz = np.mean(gx), np.mean(gy), np.mean(gz)
        span = max(np.ptp(gx), np.ptp(gy), np.ptp(gz)) * 0.6
        return cx, cy, cz, span

    bounds_s = compute_bounds(points_s)
    bounds_w = compute_bounds(points_w)

    def draw_skeleton(ax, labels, points, fi, bounds, is_foot_strike, title_prefix, meta):
        ax.cla()
        cx, cy, cz, span = bounds

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

            if is_lead_leg_connection(m1_name, m2_name):
                color = "#e74c3c"  # Red = lead leg
                lw = 4
            else:
                color = "#2c3e50"
                lw = 1.5
            ax.plot(x, y, z, color=color, linewidth=lw)

        for i, label in enumerate(labels):
            x, y, z = points[0, i, fi], points[1, i, fi], points[2, i, fi]
            if np.isnan(x) or (x == 0 and y == 0 and z == 0):
                continue
            if label in LEAD_LEG_MARKERS:
                color = "#e74c3c"
                size = 25
            else:
                color = "#3498db"
                size = 10
            ax.scatter(x, y, z, c=color, s=size, alpha=0.8)

        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_zlim(cz - span, cz + span)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")

        speed = meta["pitch_speed_mph"]
        fs_marker = " ** FOOT STRIKE **" if is_foot_strike else ""
        ax.set_title(f"{title_prefix} ({speed:.1f} mph){fs_marker}", fontsize=11)
        ax.view_init(elev=20, azim=45)

    # Map animation frame to foot strike detection
    fs_anim_s = min(range(n_anim), key=lambda i: abs(frames_s[i] - fs_s))
    fs_anim_w = min(range(n_anim), key=lambda i: abs(frames_w[i] - fs_w))

    def update(frame_num):
        fi_s = frames_s[frame_num]
        fi_w = frames_w[frame_num]

        is_fs_s = abs(frame_num - fs_anim_s) <= 1
        is_fs_w = abs(frame_num - fs_anim_w) <= 1

        draw_skeleton(ax_strong, labels_s, points_s, fi_s, bounds_s,
                      is_fs_s, "Strong Block", strong_meta)
        draw_skeleton(ax_weak, labels_w, points_w, fi_w, bounds_w,
                      is_fs_w, "Weak Block", weak_meta)

        fig.suptitle("Lead Leg Block Comparison — Driveline OBP\n"
                     "Red = Lead Leg", fontsize=13, y=0.98)

    anim = animation.FuncAnimation(fig, update, frames=n_anim, interval=80)
    anim.save(str(output_path), writer="pillow", fps=12)
    plt.close(fig)
    print(f"  Comparison GIF saved: {output_path} ({n_anim} frames)")


def main():
    parser = argparse.ArgumentParser(description="LLB comparison GIF")
    parser.add_argument("--download", type=int, default=40)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting LLB features...")
    results = collect_llb_features(args.download)

    if len(results) < 2:
        print(f"Need at least 2 samples with LLB features, got {len(results)}")
        return

    # Sort by knee extension peak velocity (higher = stronger block)
    results.sort(key=lambda r: r["llb_knee_ext_peak_velocity"])

    weak = results[0]
    strong = results[-1]

    print(f"\n  Strong block: {strong['filename']} "
          f"({strong['pitch_speed_mph']:.1f} mph, "
          f"knee ext vel={strong['llb_knee_ext_peak_velocity']:.0f} deg/s)")
    print(f"  Weak block:   {weak['filename']} "
          f"({weak['pitch_speed_mph']:.1f} mph, "
          f"knee ext vel={weak['llb_knee_ext_peak_velocity']:.0f} deg/s)")

    strong_file = RAW_DIR / strong["filename"]
    weak_file = RAW_DIR / weak["filename"]

    print("\nGenerating comparison GIF...")
    output_path = OUTPUT_DIR / "llb_comparison.gif"
    create_comparison_gif(strong_file, weak_file, strong, weak, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
