"""Step 1: Driveline OBP C3D → ezc3d skeleton visualization.

Reads C3D motion capture files from the Driveline OpenBiomechanics Project,
extracts 3D marker positions, and generates skeleton stick-figure visualizations.

Usage:
    python skeleton_c3d.py                          # Both pitching & hitting
    python skeleton_c3d.py --mode pitching           # Pitching only
    python skeleton_c3d.py --mode hitting             # Hitting only
    python skeleton_c3d.py --input data/raw/my.c3d   # Custom file

Output:
    data/output/skeleton_pitching_frame.png
    data/output/skeleton_pitching_anim.gif
    data/output/skeleton_hitting_frame.png
    data/output/skeleton_hitting_anim.gif
"""

import argparse
import os
from pathlib import Path

import ezc3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

# Driveline OBP marker set — anatomical connections for stick figure
# Based on Plug-in Gait marker placement
BODY_CONNECTIONS = [
    # Head
    ("LFHD", "RFHD"), ("LBHD", "RBHD"), ("LFHD", "LBHD"), ("RFHD", "RBHD"),
    # Spine
    ("C7", "CLAV"), ("CLAV", "STRN"), ("C7", "T10"), ("T10", "STRN"),
    # Left arm
    ("CLAV", "LSHO"), ("LSHO", "LUPA"), ("LUPA", "LELB"), ("LELB", "LFRM"),
    ("LFRM", "LWRA"), ("LWRA", "LWRB"), ("LWRA", "LFIN"),
    # Right arm
    ("CLAV", "RSHO"), ("RSHO", "RUPA"), ("RUPA", "RELB"), ("RELB", "RFRM"),
    ("RFRM", "RWRA"), ("RWRA", "RWRB"), ("RWRA", "RFIN"),
    # Left medial elbow
    ("LELB", "LMELB"),
    # Right medial elbow
    ("RELB", "RMELB"),
    # Pelvis
    ("LASI", "RASI"), ("LPSI", "RPSI"), ("LASI", "LPSI"), ("RASI", "RPSI"),
    # Left leg
    ("LASI", "LKNE"), ("LKNE", "LMKNE"), ("LKNE", "LANK"),
    ("LANK", "LMANK"), ("LANK", "LHEE"), ("LANK", "LTOE"),
    ("LASI", "LTHI"), ("LKNE", "LTIB"),
    # Right leg
    ("RASI", "RKNE"), ("RKNE", "RMKNE"), ("RKNE", "RANK"),
    ("RANK", "RMANK"), ("RANK", "RHEE"), ("RANK", "RTOE"),
    ("RASI", "RTHI"), ("RKNE", "RTIB"),
    # Back
    ("RBAK", "T10"),
]

# Bat markers (hitting only)
BAT_CONNECTIONS = [
    ("Marker1", "Marker2"), ("Marker2", "Marker3"), ("Marker3", "Marker4"),
    ("Marker4", "Marker5"), ("Marker5", "Marker6"), ("Marker6", "Marker7"),
    ("Marker7", "Marker8"), ("Marker8", "Marker9"), ("Marker9", "Marker10"),
]


def load_c3d(filepath):
    """Load C3D file and return marker labels, positions, and frame rate."""
    c = ezc3d.c3d(str(filepath))
    labels = c["parameters"]["POINT"]["LABELS"]["value"]
    points = c["data"]["points"]  # shape: (4, n_markers, n_frames) — x,y,z,1
    rate = c["parameters"]["POINT"]["RATE"]["value"][0]
    return labels, points, rate


def get_marker_index(labels, name):
    """Get index of a marker by name, or -1 if not found."""
    try:
        return labels.index(name)
    except ValueError:
        return -1


def plot_skeleton_frame(labels, points, frame_idx, title="", ax=None):
    """Plot a single frame of the skeleton as a 3D stick figure."""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    # Determine if this is a hitting file (has bat markers)
    has_bat = get_marker_index(labels, "Marker1") >= 0
    connections = BODY_CONNECTIONS + (BAT_CONNECTIONS if has_bat else [])

    # Plot connections
    for m1_name, m2_name in connections:
        i1 = get_marker_index(labels, m1_name)
        i2 = get_marker_index(labels, m2_name)
        if i1 < 0 or i2 < 0:
            continue
        x = [points[0, i1, frame_idx], points[0, i2, frame_idx]]
        y = [points[1, i1, frame_idx], points[1, i2, frame_idx]]
        z = [points[2, i1, frame_idx], points[2, i2, frame_idx]]
        # Skip if any coordinate is NaN or zero (missing marker)
        if any(np.isnan(x + y + z)) or (x[0] == 0 and y[0] == 0 and z[0] == 0):
            continue
        is_bat = m1_name.startswith("Marker")
        color = "#e74c3c" if is_bat else "#2c3e50"
        linewidth = 3 if is_bat else 1.5
        ax.plot(x, y, z, color=color, linewidth=linewidth)

    # Plot markers
    for i, label in enumerate(labels):
        x, y, z = points[0, i, frame_idx], points[1, i, frame_idx], points[2, i, frame_idx]
        if np.isnan(x) or (x == 0 and y == 0 and z == 0):
            continue
        is_bat = label.startswith("Marker")
        color = "#e74c3c" if is_bat else "#3498db"
        size = 20 if is_bat else 10
        ax.scatter(x, y, z, c=color, s=size, alpha=0.8)

    # Set axis properties
    all_x = points[0, :, frame_idx]
    all_y = points[1, :, frame_idx]
    all_z = points[2, :, frame_idx]
    valid = ~np.isnan(all_x) & (all_x != 0)
    if valid.any():
        cx, cy, cz = np.mean(all_x[valid]), np.mean(all_y[valid]), np.mean(all_z[valid])
        span = max(np.ptp(all_x[valid]), np.ptp(all_y[valid]), np.ptp(all_z[valid])) * 0.6
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_zlim(cz - span, cz + span)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(title or "Skeleton")
    ax.view_init(elev=20, azim=45)
    return fig, ax


def create_animation(labels, points, rate, output_path, title="", max_frames=200):
    """Create an animated GIF of the skeleton motion."""
    n_frames = points.shape[2]
    # Subsample if too many frames
    step = max(1, n_frames // max_frames)
    frame_indices = list(range(0, n_frames, step))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Compute global bounds
    all_valid = ~np.isnan(points[0]) & (points[0] != 0)
    gx = points[0][all_valid]
    gy = points[1][all_valid]
    gz = points[2][all_valid]
    cx, cy, cz = np.mean(gx), np.mean(gy), np.mean(gz)
    span = max(np.ptp(gx), np.ptp(gy), np.ptp(gz)) * 0.6

    has_bat = get_marker_index(labels, "Marker1") >= 0
    connections = BODY_CONNECTIONS + (BAT_CONNECTIONS if has_bat else [])

    def update(frame_num):
        ax.cla()
        fi = frame_indices[frame_num]
        time_s = fi / rate

        for m1_name, m2_name in connections:
            i1 = get_marker_index(labels, m1_name)
            i2 = get_marker_index(labels, m2_name)
            if i1 < 0 or i2 < 0:
                continue
            x = [points[0, i1, fi], points[0, i2, fi]]
            y = [points[1, i1, fi], points[1, i2, fi]]
            z = [points[2, i1, fi], points[2, i2, fi]]
            if any(np.isnan(x + y + z)) or (x[0] == 0 and y[0] == 0 and z[0] == 0):
                continue
            is_bat = m1_name.startswith("Marker")
            color = "#e74c3c" if is_bat else "#2c3e50"
            lw = 3 if is_bat else 1.5
            ax.plot(x, y, z, color=color, linewidth=lw)

        for i, label in enumerate(labels):
            x, y, z = points[0, i, fi], points[1, i, fi], points[2, i, fi]
            if np.isnan(x) or (x == 0 and y == 0 and z == 0):
                continue
            is_bat = label.startswith("Marker")
            ax.scatter(x, y, z, c="#e74c3c" if is_bat else "#3498db",
                       s=20 if is_bat else 10, alpha=0.8)

        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_zlim(cz - span, cz + span)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_title(f"{title}  t={time_s:.2f}s  (frame {fi}/{n_frames})")
        ax.view_init(elev=20, azim=45)

    anim = animation.FuncAnimation(fig, update, frames=len(frame_indices), interval=50)
    anim.save(str(output_path), writer="pillow", fps=20)
    plt.close(fig)
    print(f"  Animation saved: {output_path} ({len(frame_indices)} frames)")


def process_file(filepath, mode_label, output_dir):
    """Process a single C3D file: static frame + animation."""
    print(f"\nProcessing {mode_label}: {filepath}")
    labels, points, rate = load_c3d(filepath)
    print(f"  Markers: {len(labels)}, Frames: {points.shape[2]}, Rate: {rate} Hz")

    # Find mid-frame for static image
    mid = points.shape[2] // 2
    fig, ax = plot_skeleton_frame(
        labels, points, mid,
        title=f"Driveline OBP — {mode_label} (frame {mid})"
    )
    frame_path = output_dir / f"skeleton_{mode_label.lower()}_frame.png"
    fig.savefig(str(frame_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Static frame saved: {frame_path}")

    # Create animation
    anim_path = output_dir / f"skeleton_{mode_label.lower()}_anim.gif"
    create_animation(labels, points, rate, anim_path, title=f"Driveline OBP — {mode_label}")

    return labels, points, rate


def main():
    parser = argparse.ArgumentParser(description="C3D skeleton visualization with ezc3d")
    parser.add_argument("--mode", choices=["pitching", "hitting", "both"], default="both")
    parser.add_argument("--input", type=str, help="Custom C3D file path")
    args = parser.parse_args()

    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        process_file(args.input, "Custom", output_dir)
        return

    raw_dir = Path("data/raw")

    if args.mode in ("pitching", "both"):
        pitch_file = raw_dir / "pitching_sample.c3d"
        if pitch_file.exists():
            process_file(str(pitch_file), "Pitching", output_dir)
        else:
            print(f"Pitching file not found: {pitch_file}")

    if args.mode in ("hitting", "both"):
        hit_file = raw_dir / "hitting_sample.c3d"
        if hit_file.exists():
            process_file(str(hit_file), "Hitting", output_dir)
        else:
            print(f"Hitting file not found: {hit_file}")

    print("\nDone! Check data/output/ for results.")


if __name__ == "__main__":
    main()
