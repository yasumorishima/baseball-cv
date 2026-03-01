"""Lead Leg Block — Hitting: full-body skeleton with knee braking annotation.

Key story (hitting):
  After foot strike, the lead knee must rapidly EXTEND to create a fixed axis
  for hip rotation.  Q5 (efficient) knee extends fast from a bent position.
  Q1 (inefficient) knee stays bent or collapses forward.

Layout:
  Top: Q1 (poor) | Q5 (efficient) 3-D skeletons, lead leg in red
  Bottom: knee angle + knee extension velocity over time, dot tracking current frame

Usage:
    python llb_knee_detail_hitting.py

Output:
    data/output/llb_knee_detail_hitting.gif
"""

from pathlib import Path

import ezc3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import numpy as np
from matplotlib import animation

from skeleton_analysis import (
    compute_angular_velocity,
    compute_knee_flexion,
    load_c3d,
)
from skeleton_c3d import BODY_CONNECTIONS, get_marker_index

_jp = [f.name for f in _fm.fontManager.ttflist if "Noto" in f.name and "CJK" in f.name]
if _jp:
    plt.rcParams["font.family"] = _jp[0]

OUTPUT_DIR = Path("data/output")
RAW_DIR = Path("data/raw")

# Same files as efficient_hitter_gif.py
Q1_FILE = RAW_DIR / "000009_000123_63_140_R_001_746.c3d"
Q5_FILE = RAW_DIR / "000004_000103_75_236_R_003_972.c3d"

Q1_META = {
    "exit_velocity_mph": 74.6, "bat_speed": 7.23,
    "stride_m": 0.718, "body_efficiency": -11.88,
    "llb_foot_strike_frame": 374,
    "label": "Q1：前脚ブレーキ 弱",
    "color": "#e67e22",
}
Q5_META = {
    "exit_velocity_mph": 97.2, "bat_speed": 7.80,
    "stride_m": 0.993, "body_efficiency": +9.70,
    "llb_foot_strike_frame": 574,
    "label": "Q5：前脚ブレーキ 強",
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


def get_bounds(pts):
    v = ~np.isnan(pts[0]) & (pts[0] != 0)
    gx, gy, gz = pts[0][v], pts[1][v], pts[2][v]
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
    ax.view_init(elev=15, azim=50)

    fs_tag = "  ★ 踏み込み ★" if is_fs else ""
    eff_sign = "+" if meta["body_efficiency"] > 0 else ""
    ax.set_title(
        f"{meta['label']}{fs_tag}\n"
        f"打球速度 {meta['exit_velocity_mph']:.1f} mph  |  "
        f"ストライド {meta['stride_m']:.2f}m  |  "
        f"体効率 {eff_sign}{meta['body_efficiency']:.2f} mph",
        fontsize=10, fontweight="bold", color=meta["color"],
    )


def draw_graph(ax, common_time, knee_q1, knee_q5, vel_q1, vel_q5, frame_num):
    ax.clear()

    # Knee angle (solid line)
    ax.plot(common_time, knee_q1, color=Q1_META["color"], linewidth=2.5,
            label=f"Q1 膝角度", linestyle="-")
    ax.plot(common_time, knee_q5, color=Q5_META["color"], linewidth=2.5,
            label=f"Q5 膝角度", linestyle="-")

    # Knee extension velocity (dashed line, right axis)
    ax2 = ax.twinx()
    ax2.plot(common_time, vel_q1, color=Q1_META["color"], linewidth=1.5,
             linestyle="--", alpha=0.7)
    ax2.plot(common_time, vel_q5, color=Q5_META["color"], linewidth=1.5,
             linestyle="--", alpha=0.7, label="膝の伸展速度（破線）")
    ax2.set_ylabel("膝の伸展速度 (°/s)\n↑ 伸びる  ↓ 曲がる", fontsize=9, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.axhline(0, color="gray", linewidth=0.5, alpha=0.4)

    # Foot strike line
    ax.axvline(0, color="#e74c3c", linewidth=2.5, linestyle="--",
               alpha=0.85, label="踏み込み瞬間")

    # Current position dots
    t_now = common_time[frame_num]
    ax.scatter(t_now, knee_q1[frame_num], s=180, c=Q1_META["color"],
               zorder=6, edgecolors="black", linewidths=2)
    ax.scatter(t_now, knee_q5[frame_num], s=180, c=Q5_META["color"],
               zorder=6, edgecolors="black", linewidths=2)

    ax.set_xlabel(
        "踏み込み前 ← 時間 (秒) → 踏み込み後",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylabel("前膝の角度 (°)\n小さい = より曲がっている", fontsize=10)
    ax.set_title(
        "踏み込み後に前膝が素早く伸びる（Q5）→ 腰の回転軸が固定される → 打球速度↑\n"
        "実線 = 膝角度　破線 = 伸展速度",
        fontsize=10, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(common_time[0] - 0.01, common_time[-1] + 0.01)


def create_gif(output_path):
    if not Q1_FILE.exists() or not Q5_FILE.exists():
        raise FileNotFoundError(
            f"C3Dファイルが見つかりません: {Q1_FILE} / {Q5_FILE}"
        )

    # Load C3D
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

    # Knee angles & velocities
    markers1, _, _ = load_c3d(str(Q1_FILE))
    markers5, _, _ = load_c3d(str(Q5_FILE))
    knee1 = compute_knee_flexion(markers1, "L")
    knee5 = compute_knee_flexion(markers5, "L")
    vel1 = compute_angular_velocity(knee1, rate1)
    vel5 = compute_angular_velocity(knee5, rate5)

    # Time window: 0.35s pre / 0.45s post foot strike
    pre_sec, post_sec = 0.35, 0.45
    total_sec = pre_sec + post_sec

    start1 = max(0, fs1 - int(pre_sec * rate1))
    end1 = min(pts1.shape[2], fs1 + int(post_sec * rate1))
    start5 = max(0, fs5 - int(pre_sec * rate5))
    end5 = min(pts5.shape[2], fs5 + int(post_sec * rate5))

    n_anim = 120
    idx1 = np.linspace(start1, end1 - 1, n_anim).astype(int)
    idx5 = np.linspace(start5, end5 - 1, n_anim).astype(int)
    fs_gif = int(pre_sec / total_sec * n_anim)

    common_time = np.linspace(-pre_sec, post_sec, n_anim)

    # Interpolate knee angle & velocity onto common time axis
    gt1 = (np.arange(start1, end1) - fs1) / rate1
    gt5 = (np.arange(start5, end5) - fs5) / rate5
    knee1_i = np.interp(common_time, gt1, knee1[start1:end1])
    knee5_i = np.interp(common_time, gt5, knee5[start5:end5])
    vel1_i = np.interp(common_time, gt1, vel1[start1:end1])
    vel5_i = np.interp(common_time, gt5, vel5[start5:end5])

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
        draw_graph(ax_g, common_time, knee1_i, knee5_i, vel1_i, vel5_i, frame_num)

        # Knee angle overlays
        for ax_pos, kang, kvel, meta in [
            (0.03, knee1_i[frame_num], vel1_i[frame_num], Q1_META),
            (0.53, knee5_i[frame_num], vel5_i[frame_num], Q5_META),
        ]:
            t = fig.text(
                ax_pos, 0.56, f"膝角度: {kang:.0f}°",
                fontsize=14, fontweight="bold", color="#2c3e50",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#2c3e50", alpha=0.95),
            )
            overlay_texts.append(t)
            vc = "#27ae60" if kvel > 0 else "#c0392b"
            vl = "↑ 伸展中" if kvel > 0 else "↓ 屈曲中"
            t2 = fig.text(
                ax_pos, 0.52, f"{abs(kvel):.0f}°/s {vl}",
                fontsize=12, fontweight="bold", color=vc,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=vc, alpha=0.95),
            )
            overlay_texts.append(t2)

        if is_fs:
            for xpos in (0.23, 0.71):
                t = fig.text(
                    xpos, 0.29, "⚡ 踏み込み瞬間",
                    fontsize=13, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor="#e74c3c", alpha=0.95),
                )
                overlay_texts.append(t)

        if is_post:
            t = fig.text(
                0.35, 0.29,
                "膝が素早く伸びて\n腰の回転軸を固定 →",
                fontsize=10, fontweight="bold", color="#2980b9",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#eaf4fb", edgecolor="#2980b9", alpha=0.95),
            )
            overlay_texts.append(t)
            t2 = fig.text(
                0.78, 0.29,
                "膝の伸展が遅く\n前方に流れる →",
                fontsize=10, fontweight="bold", color="#e67e22",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#fef9e7", edgecolor="#e67e22", alpha=0.95),
            )
            overlay_texts.append(t2)

        fig.suptitle(
            "前脚ブレーキの仕組み：踏み込み後に前膝が素早く伸びることで腰の回転軸が固定される（Driveline OBP）\n"
            "赤 = 踏み込み脚（前脚）",
            fontsize=12, fontweight="bold", y=0.99,
        )

    print(f"  Rendering {n_anim} frames...")
    anim = animation.FuncAnimation(fig, update, frames=n_anim, interval=80)
    anim.save(str(output_path), writer="pillow", fps=12)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "llb_knee_detail_hitting.gif"
    print(f"Q1: {Q1_FILE.name}")
    print(f"Q5: {Q5_FILE.name}")
    create_gif(out)
    print("Done.")


if __name__ == "__main__":
    main()
