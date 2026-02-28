"""Step 3: Joint angle & angular velocity extraction from skeleton data.

Computes biomechanical features from C3D motion capture data:
- Joint angles: elbow flexion, shoulder external rotation, trunk rotation, knee flexion
- Angular velocities (frame-to-frame differences)
- Kinematic sequence time-series plots

Usage:
    python skeleton_analysis.py                             # Default: pitching sample
    python skeleton_analysis.py --input data/raw/hitting_sample.c3d --mode hitting

Output:
    data/output/joint_angles_{mode}.csv
    data/output/kinematic_sequence_{mode}.png
    data/output/angular_velocity_{mode}.png
"""

import argparse
from pathlib import Path

import ezc3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_c3d(filepath):
    """Load C3D file and return marker dict and frame rate."""
    c = ezc3d.c3d(str(filepath))
    labels = c["parameters"]["POINT"]["LABELS"]["value"]
    points = c["data"]["points"]  # (4, n_markers, n_frames)
    rate = c["parameters"]["POINT"]["RATE"]["value"][0]
    marker_dict = {}
    for i, label in enumerate(labels):
        marker_dict[label] = points[:3, i, :]  # (3, n_frames) — x, y, z
    return marker_dict, rate, points.shape[2]


def angle_between_vectors(v1, v2):
    """Compute angle (degrees) between two 3D vectors per frame."""
    # v1, v2: (3, n_frames)
    dot = np.sum(v1 * v2, axis=0)
    norm1 = np.linalg.norm(v1, axis=0)
    norm2 = np.linalg.norm(v2, axis=0)
    # Avoid division by zero
    denom = norm1 * norm2
    denom[denom == 0] = 1e-10
    cos_angle = np.clip(dot / denom, -1, 1)
    return np.degrees(np.arccos(cos_angle))


def compute_elbow_flexion(markers, side="R"):
    """Elbow flexion angle: angle at elbow between upper arm and forearm."""
    sho = markers.get(f"{side}SHO")
    elb = markers.get(f"{side}ELB")
    wra = markers.get(f"{side}WRA")
    if sho is None or elb is None or wra is None:
        return None
    upper_arm = sho - elb  # shoulder to elbow
    forearm = wra - elb    # wrist to elbow
    return angle_between_vectors(upper_arm, forearm)


def compute_shoulder_abduction(markers, side="R"):
    """Shoulder abduction: angle between upper arm and trunk midline."""
    sho = markers.get(f"{side}SHO")
    elb = markers.get(f"{side}ELB")
    c7 = markers.get("C7")
    strn = markers.get("STRN")
    if sho is None or elb is None or c7 is None or strn is None:
        return None
    upper_arm = elb - sho
    trunk = strn - c7
    return angle_between_vectors(upper_arm, trunk)


def compute_trunk_rotation(markers):
    """Trunk rotation: angle between shoulder line and pelvis line in horizontal plane."""
    lsho = markers.get("LSHO")
    rsho = markers.get("RSHO")
    lasi = markers.get("LASI")
    rasi = markers.get("RASI")
    if lsho is None or rsho is None or lasi is None or rasi is None:
        return None
    shoulder_line = rsho - lsho  # (3, n_frames)
    pelvis_line = rasi - lasi
    # Project onto horizontal plane (XY)
    shoulder_xy = shoulder_line[:2, :]
    pelvis_xy = pelvis_line[:2, :]
    # Angle between projections
    dot = np.sum(shoulder_xy * pelvis_xy, axis=0)
    norm1 = np.linalg.norm(shoulder_xy, axis=0)
    norm2 = np.linalg.norm(pelvis_xy, axis=0)
    denom = norm1 * norm2
    denom[denom == 0] = 1e-10
    cos_angle = np.clip(dot / denom, -1, 1)
    return np.degrees(np.arccos(cos_angle))


def compute_knee_flexion(markers, side="R"):
    """Knee flexion: angle at knee between thigh and shank."""
    hip_marker = f"{side}ASI"  # Use ASIS as hip proxy
    kne = markers.get(f"{side}KNE")
    ank = markers.get(f"{side}ANK")
    hip = markers.get(hip_marker)
    if hip is None or kne is None or ank is None:
        return None
    thigh = hip - kne
    shank = ank - kne
    return angle_between_vectors(thigh, shank)


def compute_angular_velocity(angles, rate):
    """Compute angular velocity (deg/s) from angle time series."""
    dt = 1.0 / rate
    vel = np.gradient(angles) / dt
    return vel


def detect_foot_strike(markers, side="L", rate=360.0, verbose=False):
    """Detect foot strike frame using heel Z coordinate position.

    Simple coordinate-based algorithm:
    1. Ground level = heel Z at recording start (pitcher is standing)
    2. Find leg lift peak (max heel Z)
    3. After lift peak, first frame where heel Z returns to ground level

    If side="auto", auto-detects the stride leg by comparing LHEE vs RHEE
    lift peak timing — the stride leg peaks earlier (30-50% of recording),
    the plant foot peaks late (90-100%) due to follow-through.

    Args:
        markers: dict of marker name -> (3, n_frames) array
        side: "L", "R", or "auto" (auto-detect stride leg)
        rate: sampling rate in Hz (unused, kept for API compat)
        verbose: print debug info

    Returns:
        Frame index of foot strike, or None if detection fails.
    """
    if side == "auto":
        lhee = markers.get("LHEE")
        rhee = markers.get("RHEE")
        if lhee is not None and rhee is not None:
            nf = lhee.shape[1]
            l_peak = 10 + np.argmax(lhee[2, 10:])
            r_peak = 10 + np.argmax(rhee[2, 10:])
            # Stride leg lifts and peaks earlier; plant foot peaks late
            side = "L" if l_peak < r_peak else "R"
            if verbose:
                print(f"    [foot_strike] auto-detect: LHEE peak={l_peak} "
                      f"({l_peak/nf*100:.0f}%), RHEE peak={r_peak} "
                      f"({r_peak/nf*100:.0f}%) -> side={side}")
        elif lhee is not None:
            side = "L"
        elif rhee is not None:
            side = "R"
        else:
            return None

    heel_key = f"{side}HEE"
    heel = markers.get(heel_key)
    if heel is None:
        if verbose:
            print(f"    [foot_strike] FAIL: marker '{heel_key}' not found")
        return None

    heel_z = heel[2, :]  # vertical position
    n_frames = len(heel_z)

    # Check for sufficient valid data
    valid_mask = ~np.isnan(heel_z) & (heel_z != 0)
    if valid_mask.sum() < n_frames * 0.5:
        if verbose:
            print(f"    [foot_strike] FAIL: '{heel_key}' has "
                  f"{valid_mask.sum()}/{n_frames} valid frames")
        return None

    # Ground level = average heel Z at recording start (pitcher standing)
    ground_z = np.mean(heel_z[:10])

    # Edge case: recording starts with foot already elevated (e.g. 000610)
    if ground_z > 0.30:
        ground_z = np.min(heel_z[valid_mask])
        if verbose:
            print(f"    [foot_strike] start_z high "
                  f"({np.mean(heel_z[:10]):.3f}m), using min: {ground_z:.3f}m")

    # Find leg lift peak (highest point after first 10 frames)
    lift_peak = 10 + np.argmax(heel_z[10:])
    lift_height = heel_z[lift_peak] - ground_z

    # Sanity: lift must be at least 15cm
    if lift_height < 0.15:
        if verbose:
            print(f"    [foot_strike] FAIL: lift height only "
                  f"{lift_height:.3f}m (need >= 0.15m)")
        return None

    # Threshold: ground level + 3cm tolerance
    threshold = ground_z + 0.03

    # After lift peak, find first frame where heel Z drops to ground
    for i in range(lift_peak + 1, n_frames):
        if heel_z[i] <= threshold:
            if verbose:
                print(f"    [foot_strike] FOUND: frame {i}/{n_frames} "
                      f"({i / n_frames * 100:.1f}%), "
                      f"heel_z={heel_z[i]:.4f}m, ground={ground_z:.4f}m")
            return i

    if verbose:
        print(f"    [foot_strike] FAIL: heel never returned to ground "
              f"(threshold={threshold:.4f}m)")
    return None


def infer_throwing_direction(markers):
    """Infer throwing/swinging direction as a 2D unit vector in the horizontal plane.

    Uses the RFIN (right fingertip) marker trajectory to estimate direction.
    Falls back to LFIN if RFIN is not available.

    Returns:
        (2,) unit vector in XY plane, or None.
    """
    for key in ("RFIN", "LFIN"):
        fin = markers.get(key)
        if fin is not None:
            break
    else:
        return None

    fin_xy = fin[:2, :]  # (2, n_frames)

    # Horizontal velocity
    vx = np.gradient(fin_xy[0, :])
    vy = np.gradient(fin_xy[1, :])
    speed = np.sqrt(vx**2 + vy**2)

    # Peak speed frame (near release/contact)
    peak_frame = np.argmax(speed)

    # Displacement vector around peak (±10 frames)
    window = 10
    f_start = max(0, peak_frame - window)
    f_end = min(fin_xy.shape[1] - 1, peak_frame + window)
    disp = fin_xy[:, f_end] - fin_xy[:, f_start]

    norm = np.linalg.norm(disp)
    if norm < 1e-6:
        return None
    return disp / norm


def compute_ankle_braking(markers, foot_strike_frame, rate, throwing_dir, side="L"):
    """Compute ankle braking metrics at foot strike.

    Measures the horizontal velocity change of the ankle marker
    around the foot strike event to quantify the lead leg block.

    Returns:
        dict with braking metrics, or None if markers missing.
    """
    ank_key = f"{side}ANK"
    ank = markers.get(ank_key)
    if ank is None:
        return None

    n_frames = ank.shape[1]
    ank_xy = ank[:2, :]  # horizontal plane

    # Horizontal velocity (mm/s)
    vx = np.gradient(ank_xy[0, :]) * rate
    vy = np.gradient(ank_xy[1, :]) * rate

    # Project velocity onto throwing direction if available
    if throwing_dir is not None:
        v_proj = vx * throwing_dir[0] + vy * throwing_dir[1]
    else:
        v_proj = np.sqrt(vx**2 + vy**2)

    # Velocity at foot strike
    vel_at_strike = v_proj[foot_strike_frame]

    # Velocity 50ms after foot strike
    post_frames = int(0.050 * rate)
    post_idx = min(foot_strike_frame + post_frames, n_frames - 1)
    vel_post_strike = v_proj[post_idx]

    # Braking metrics
    delta_v = vel_at_strike - vel_post_strike
    dt = post_frames / rate
    decel = delta_v / dt if dt > 0 else 0.0

    return {
        "ankle_velocity_at_strike": vel_at_strike,
        "ankle_velocity_post_strike": vel_post_strike,
        "ankle_velocity_delta": delta_v,
        "ankle_braking_decel": decel,
    }


def compute_lead_knee_extension_velocity(markers, foot_strike_frame, rate, side="L"):
    """Compute lead knee extension velocity after foot strike.

    Measures how quickly the lead knee extends (straightens) after ground
    contact, which is a key component of the lead leg block mechanism.

    Returns:
        dict with knee extension metrics, or None if markers missing.
    """
    knee_angles = compute_knee_flexion(markers, side)
    if knee_angles is None:
        return None

    n_frames = len(knee_angles)

    # Knee angle at foot strike
    angle_at_strike = knee_angles[foot_strike_frame]

    # Window: 200ms after foot strike
    window_frames = int(0.200 * rate)
    end_idx = min(foot_strike_frame + window_frames, n_frames)

    if end_idx <= foot_strike_frame:
        return None

    post_strike = knee_angles[foot_strike_frame:end_idx]

    # Angular velocity (deg/s)
    ang_vel = compute_angular_velocity(knee_angles, rate)
    post_vel = ang_vel[foot_strike_frame:end_idx]

    # Peak extension velocity (positive = extending/straightening)
    peak_ext_vel = np.nanmax(post_vel)

    # Maximum extension angle after strike
    max_ext_angle = np.nanmax(post_strike)
    ext_range = max_ext_angle - angle_at_strike

    # Time to peak extension
    peak_ext_frame = np.nanargmax(post_strike)
    time_to_peak = peak_ext_frame / rate

    return {
        "knee_angle_at_strike": angle_at_strike,
        "knee_ext_peak_velocity": peak_ext_vel,
        "knee_extension_range": ext_range,
        "time_to_peak_extension": time_to_peak,
    }


def compute_position_braking_features(markers, foot_strike_frame, rate, throwing_dir, side="L"):
    """Compute position-based braking features at foot strike.

    Measures joint POSITIONS relative to the throwing direction — not angles.
    Features that need throwing_dir projection are skipped if it's None.
    Stride length (Euclidean distance) is always computed.

    All directional values are projected onto the throwing direction (mm or mm/s).
    Positive = forward (in throwing direction), Negative = backward.
    """
    result = {}

    kne = markers.get(f"{side}KNE")
    ank = markers.get(f"{side}ANK")
    n_frames = next(iter(markers.values())).shape[1]
    c7 = markers.get("C7")
    lasi = markers.get("LASI")
    rasi = markers.get("RASI")

    # === Features that DON'T need throwing_dir ===

    # 6. Stride length at foot strike (plant foot to pivot foot) — Euclidean
    throw_side = "R" if side == "L" else "L"
    pivot_ank = markers.get(f"{throw_side}ANK")
    if ank is not None and pivot_ank is not None:
        plant_xy = ank[:2, foot_strike_frame]
        pivot_xy = pivot_ank[:2, foot_strike_frame]
        stride_vec = plant_xy - pivot_xy
        result["stride_length"] = float(np.linalg.norm(stride_vec))

    # === Features that NEED throwing_dir ===

    if throwing_dir is None:
        return result

    # 1. Knee-ankle offset in throwing direction at foot strike
    if kne is not None and ank is not None:
        knee_xy = kne[:2, foot_strike_frame]
        ankle_xy = ank[:2, foot_strike_frame]
        offset = knee_xy - ankle_xy
        result["knee_ankle_offset"] = float(np.dot(offset, throwing_dir))

    # 2. Knee forward velocity at and after foot strike (multiple windows)
    if kne is not None:
        vx = np.gradient(kne[0, :]) * rate
        vy = np.gradient(kne[1, :]) * rate
        v_forward = vx * throwing_dir[0] + vy * throwing_dir[1]

        result["knee_forward_vel_at_strike"] = float(v_forward[foot_strike_frame])

        for window_ms in [25, 50, 100, 150]:
            w_frames = int(window_ms / 1000.0 * rate)
            w_end = min(foot_strike_frame + w_frames, n_frames - 1)
            avg_vel = float(np.mean(v_forward[foot_strike_frame:w_end + 1]))
            result[f"knee_forward_vel_{window_ms}ms"] = avg_vel

        result["knee_forward_vel_post"] = result["knee_forward_vel_50ms"]
        result["knee_forward_decel"] = (
            result["knee_forward_vel_at_strike"] - result["knee_forward_vel_post"]
        )

    # 3. Head forward displacement around foot strike (±100ms)
    for key in ("RFHD", "LFHD", "C7"):
        head = markers.get(key)
        if head is not None:
            break
    else:
        head = None

    if head is not None:
        window = int(0.100 * rate)
        pre = max(0, foot_strike_frame - window)
        post = min(n_frames - 1, foot_strike_frame + window)

        head_disp = head[:2, post] - head[:2, pre]
        result["head_forward_disp"] = float(np.dot(head_disp, throwing_dir))

        head_disp_post = head[:2, post] - head[:2, foot_strike_frame]
        result["head_forward_disp_post"] = float(np.dot(head_disp_post, throwing_dir))

    # 4. Hip-ankle offset
    hip = markers.get(f"{side}ASI")
    if hip is not None and ank is not None:
        hip_xy = hip[:2, foot_strike_frame]
        ankle_xy = ank[:2, foot_strike_frame]
        hip_offset = hip_xy - ankle_xy
        result["hip_ankle_offset"] = float(np.dot(hip_offset, throwing_dir))

    # 5. Trunk forward lean at foot strike
    if c7 is not None and lasi is not None and rasi is not None:
        pelvis_mid = (lasi[:2, foot_strike_frame] + rasi[:2, foot_strike_frame]) / 2
        c7_xy = c7[:2, foot_strike_frame]
        trunk_lean = c7_xy - pelvis_mid
        result["trunk_forward_lean"] = float(np.dot(trunk_lean, throwing_dir))

    # 6b. Stride forward component (needs throwing_dir)
    if ank is not None and pivot_ank is not None:
        stride_vec = ank[:2, foot_strike_frame] - pivot_ank[:2, foot_strike_frame]
        result["stride_forward"] = float(np.dot(stride_vec, throwing_dir))

    # 7. Pelvis forward velocity and deceleration
    if lasi is not None and rasi is not None:
        pelvis_xy = (lasi[:2, :] + rasi[:2, :]) / 2
        pvx = np.gradient(pelvis_xy[0, :]) * rate
        pvy = np.gradient(pelvis_xy[1, :]) * rate
        pv_forward = pvx * throwing_dir[0] + pvy * throwing_dir[1]

        result["pelvis_vel_at_strike"] = float(pv_forward[foot_strike_frame])

        post_frames = int(0.050 * rate)
        post_end = min(foot_strike_frame + post_frames, n_frames - 1)
        result["pelvis_vel_post"] = float(
            np.mean(pv_forward[foot_strike_frame:post_end + 1])
        )
        result["pelvis_decel"] = (
            result["pelvis_vel_at_strike"] - result["pelvis_vel_post"]
        )

    return result


def compute_timing_features(markers, foot_strike_frame, rate, side="L"):
    """Timing and phase metrics relative to foot strike.

    Captures WHERE in the kinematic sequence the pitcher is when the front
    foot lands. Good timing = hips open, shoulders still closed, arm cocked.
    """
    result = {}
    throw_side = "R" if side == "L" else "L"

    # 1. Trunk rotation at foot strike + velocity
    trunk_rot = compute_trunk_rotation(markers)
    if trunk_rot is not None:
        result["trunk_rotation_at_strike"] = float(trunk_rot[foot_strike_frame])
        trunk_vel = compute_angular_velocity(trunk_rot, rate)
        result["trunk_rot_vel_at_strike"] = float(trunk_vel[foot_strike_frame])

        # Time from foot strike to peak trunk rotation velocity
        post_vel = np.abs(trunk_vel[foot_strike_frame:])
        if len(post_vel) > 0:
            peak_idx = int(np.argmax(post_vel))
            result["time_fs_to_peak_trunk_vel"] = float(peak_idx / rate)

    # 2. Elbow angle at foot strike + time to peak elbow velocity
    elbow = compute_elbow_flexion(markers, throw_side)
    if elbow is not None:
        result["elbow_angle_at_strike"] = float(elbow[foot_strike_frame])
        elbow_vel = np.abs(compute_angular_velocity(elbow, rate))

        post_vel = elbow_vel[foot_strike_frame:]
        if len(post_vel) > 0:
            peak_idx = int(np.argmax(post_vel))
            result["time_fs_to_peak_elbow_vel"] = float(peak_idx / rate)

    # 3. Shoulder abduction at foot strike
    shoulder = compute_shoulder_abduction(markers, throw_side)
    if shoulder is not None:
        result["shoulder_abd_at_strike"] = float(shoulder[foot_strike_frame])

    return result


def compute_release_features(markers, rate, side="R"):
    """Compute release point features: finger deceleration, wrist snap, direction.

    Proxy for 'finger on the ball' quality — how sharply the finger
    decelerates after release, how fast the wrist snaps, and how well
    the finger velocity aligns with the throwing direction.
    """
    result = {}

    fin = markers.get(f"{side}FIN")
    wra = markers.get(f"{side}WRA")
    wrb = markers.get(f"{side}WRB")
    if fin is None or wra is None or wrb is None:
        return result

    n_frames = fin.shape[1]

    # Linear speed of finger
    fin_vel = np.gradient(fin, axis=1) * rate
    fin_speed = np.sqrt(fin_vel[0] ** 2 + fin_vel[1] ** 2 + fin_vel[2] ** 2)

    # Release = peak finger speed
    release = int(np.nanargmax(fin_speed))
    result["release_finger_speed"] = float(fin_speed[release])

    # 1. Finger deceleration after release (sharpness of speed drop)
    for window_ms in [10, 25, 50]:
        w = int(window_ms / 1000 * rate)
        post = min(release + w, n_frames - 1)
        if post > release:
            drop = float(fin_speed[release] - fin_speed[post])
            decel = drop / (window_ms / 1000)
            result[f"release_decel_{window_ms}ms"] = decel

    # 2. Wrist snap: angular velocity of wrist-to-finger vector (3D)
    wrist_mid = (wra + wrb) / 2
    wf = fin - wrist_mid
    wf_len = np.linalg.norm(wf, axis=0, keepdims=True)
    wf_unit = wf / (wf_len + 1e-10)
    cross = np.cross(wf_unit[:, :-1].T, wf_unit[:, 1:].T).T
    snap_speed = np.linalg.norm(cross, axis=0) * rate  # rad/s

    window = int(0.050 * rate)
    pre = max(0, release - window)
    post_f = min(len(snap_speed) - 1, release + window)
    snap_deg = np.degrees(snap_speed)

    result["wrist_snap_at_release"] = float(snap_deg[min(release, len(snap_deg) - 1)])
    result["wrist_snap_peak"] = float(np.max(snap_deg[pre:post_f + 1]))

    # 3. Finger direction alignment with throwing direction
    throwing_dir = infer_throwing_direction(markers)
    if throwing_dir is not None:
        fin_dir_2d = fin_vel[:2, release]
        norm = np.linalg.norm(fin_dir_2d)
        if norm > 0:
            fin_dir_2d = fin_dir_2d / norm
            result["release_direction_alignment"] = float(np.dot(fin_dir_2d, throwing_dir))

    return result


def compute_whip_features(markers, rate, side="R"):
    """Compute arm whip (しなり) features from linear velocities.

    Measures speed amplification along the arm: elbow → wrist → finger.
    Higher ratios = more whip-like energy transfer.
    """
    result = {}

    def linear_speed(marker_data):
        v = np.gradient(marker_data, axis=1) * rate
        return np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

    elb = markers.get(f"{side}ELB")
    wra = markers.get(f"{side}WRA")
    fin = markers.get(f"{side}FIN")

    if elb is None or wra is None or fin is None:
        return result

    elb_speed = linear_speed(elb)
    wra_speed = linear_speed(wra)
    fin_speed = linear_speed(fin)

    peak_elb = float(np.nanmax(elb_speed))
    peak_wra = float(np.nanmax(wra_speed))
    peak_fin = float(np.nanmax(fin_speed))

    result["peak_elbow_linear_speed"] = peak_elb
    result["peak_wrist_linear_speed"] = peak_wra
    result["peak_finger_linear_speed"] = peak_fin

    # Whip ratios (speed amplification)
    if peak_elb > 0:
        result["whip_wrist_elbow"] = peak_wra / peak_elb
        result["whip_finger_elbow"] = peak_fin / peak_elb
    if peak_wra > 0:
        result["whip_finger_wrist"] = peak_fin / peak_wra

    # Timing delays (ms between peak velocities)
    t_elb = int(np.nanargmax(elb_speed))
    t_wra = int(np.nanargmax(wra_speed))
    t_fin = int(np.nanargmax(fin_speed))
    result["whip_delay_elbow_to_wrist"] = float((t_wra - t_elb) / rate)
    result["whip_delay_wrist_to_finger"] = float((t_fin - t_wra) / rate)
    result["whip_delay_elbow_to_finger"] = float((t_fin - t_elb) / rate)

    return result


def compute_smoothness_features(markers, rate, side="R"):
    """Compute movement smoothness, acceleration, and kinematic sequence features.

    Measures:
    - Angular acceleration (peak) for elbow, trunk, shoulder
    - RMS jerk (smoothness) for elbow angle and knee position
    - Kinematic sequence: proximal-to-distal peak velocity timing
      (pelvis → trunk → shoulder → elbow)
    """
    result = {}
    lead_side = "L" if side == "R" else "R"

    # --- Angular acceleration & jerk ---
    for name, compute_fn, s in [
        ("elbow", compute_elbow_flexion, side),
        ("trunk", compute_trunk_rotation, None),
        ("shoulder", compute_shoulder_abduction, side),
    ]:
        angle = compute_fn(markers, s) if s else compute_fn(markers)
        if angle is None:
            continue
        vel = compute_angular_velocity(angle, rate)
        accel = np.gradient(vel) * rate  # deg/s²
        jerk = np.gradient(accel) * rate  # deg/s³
        result[f"{name}_peak_accel"] = float(np.nanmax(np.abs(accel)))
        result[f"{name}_rms_jerk"] = float(np.sqrt(np.nanmean(jerk ** 2)))

    # --- Knee positional jerk (lead leg, 3D) ---
    kne = markers.get(f"{lead_side}KNE")
    if kne is not None and kne.shape[1] > 4:
        jerk_sq = np.zeros(kne.shape[1])
        for d in range(3):
            vel_d = np.gradient(kne[d, :]) * rate
            acc_d = np.gradient(vel_d) * rate
            jrk_d = np.gradient(acc_d) * rate
            jerk_sq += jrk_d ** 2
        result["knee_pos_rms_jerk"] = float(np.sqrt(np.nanmean(jerk_sq)))

    # --- Kinematic sequence: peak velocity timing ---
    timings = {}

    # Pelvis linear speed
    lasi = markers.get("LASI")
    rasi = markers.get("RASI")
    if lasi is not None and rasi is not None:
        pelvis_xy = (lasi[:2, :] + rasi[:2, :]) / 2
        pv = np.gradient(pelvis_xy, axis=1) * rate
        pelvis_speed = np.sqrt(pv[0] ** 2 + pv[1] ** 2)
        timings["pelvis"] = int(np.nanargmax(pelvis_speed))

    trunk_angle = compute_trunk_rotation(markers)
    if trunk_angle is not None:
        tv = compute_angular_velocity(trunk_angle, rate)
        timings["trunk"] = int(np.nanargmax(np.abs(tv)))

    shoulder_angle = compute_shoulder_abduction(markers, side)
    if shoulder_angle is not None:
        sv = compute_angular_velocity(shoulder_angle, rate)
        timings["shoulder"] = int(np.nanargmax(np.abs(sv)))

    elbow_angle = compute_elbow_flexion(markers, side)
    if elbow_angle is not None:
        ev = compute_angular_velocity(elbow_angle, rate)
        timings["elbow"] = int(np.nanargmax(np.abs(ev)))

    # Sequence gaps (seconds between peak velocities)
    seq_order = ["pelvis", "trunk", "shoulder", "elbow"]
    available = [s for s in seq_order if s in timings]
    prev = None
    for seg in available:
        if prev is not None:
            gap = (timings[seg] - timings[prev]) / rate
            result[f"seq_gap_{prev}_to_{seg}"] = float(gap)
        prev = seg

    # Kinematic sequence score: fraction of pairs in correct order
    if len(available) >= 3:
        times = [timings[s] for s in available]
        correct = sum(1 for i in range(len(times) - 1) if times[i] < times[i + 1])
        result["kinematic_seq_score"] = float(correct / (len(times) - 1))

    # Total sequence duration
    if len(available) >= 2:
        times = [timings[s] for s in available]
        result["seq_total_duration"] = float((max(times) - min(times)) / rate)

    return result


def compute_lead_leg_block_features(markers, rate, side="L", verbose=False):
    """Compute all lead leg block features.

    Orchestrates foot strike detection, ankle braking, and knee extension
    analysis into a unified feature set with 'llb_' prefix.

    Returns:
        dict of LLB features (empty dict if foot strike not detected).
    """
    fs = detect_foot_strike(markers, side, rate, verbose=verbose)
    if fs is None:
        return {}

    throwing_dir = infer_throwing_direction(markers)
    braking = compute_ankle_braking(markers, fs, rate, throwing_dir, side)
    knee = compute_lead_knee_extension_velocity(markers, fs, rate, side)

    result = {
        "llb_foot_strike_frame": fs,
        "llb_foot_strike_time_s": fs / rate,
    }
    if braking:
        result.update({f"llb_{k}": v for k, v in braking.items()})
    if knee:
        result.update({f"llb_{k}": v for k, v in knee.items()})

    # Position-based braking features (the real "braking" signal)
    pos = compute_position_braking_features(markers, fs, rate, throwing_dir, side)
    result.update({f"llb_{k}": v for k, v in pos.items()})

    # Timing/phase features relative to foot strike
    timing = compute_timing_features(markers, fs, rate, side)
    result.update({f"llb_{k}": v for k, v in timing.items()})

    return result


def analyze(filepath, mode="pitching"):
    """Run full joint angle analysis on a C3D file."""
    print(f"\nAnalyzing {mode}: {filepath}")
    markers, rate, n_frames = load_c3d(filepath)
    print(f"  Markers: {len(markers)}, Frames: {n_frames}, Rate: {rate} Hz")

    time = np.arange(n_frames) / rate

    # Determine throwing/hitting arm
    # Pitching: right-handed default; Hitting: use both sides
    sides = ["R", "L"] if mode == "hitting" else ["R"]

    results = {"frame": np.arange(n_frames), "time_s": time}

    for side in sides:
        side_label = "right" if side == "R" else "left"

        elbow = compute_elbow_flexion(markers, side)
        if elbow is not None:
            results[f"elbow_flexion_{side_label}"] = elbow

        shoulder = compute_shoulder_abduction(markers, side)
        if shoulder is not None:
            results[f"shoulder_abduction_{side_label}"] = shoulder

        knee = compute_knee_flexion(markers, side)
        if knee is not None:
            results[f"knee_flexion_{side_label}"] = knee

    trunk = compute_trunk_rotation(markers)
    if trunk is not None:
        results["trunk_rotation"] = trunk

    df = pd.DataFrame(results)
    return df, rate, time


def plot_kinematic_sequence(df, time, mode, output_dir):
    """Plot kinematic sequence: joint angles over time."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Kinematic Sequence — {mode.title()}", fontsize=14)

    angle_cols = [c for c in df.columns if c not in ("frame", "time_s")]

    # Elbow flexion
    ax = axes[0, 0]
    for col in [c for c in angle_cols if "elbow" in c]:
        ax.plot(time, df[col], label=col.replace("_", " ").title())
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Elbow Flexion")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Shoulder abduction
    ax = axes[0, 1]
    for col in [c for c in angle_cols if "shoulder" in c]:
        ax.plot(time, df[col], label=col.replace("_", " ").title())
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Shoulder Abduction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Trunk rotation
    ax = axes[1, 0]
    if "trunk_rotation" in df.columns:
        ax.plot(time, df["trunk_rotation"], label="Trunk Rotation", color="#e74c3c")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Trunk Rotation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Knee flexion
    ax = axes[1, 1]
    for col in [c for c in angle_cols if "knee" in c]:
        ax.plot(time, df[col], label=col.replace("_", " ").title())
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Knee Flexion")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / f"kinematic_sequence_{mode}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kinematic sequence plot: {path}")


def plot_angular_velocity(df, rate, mode, output_dir):
    """Plot angular velocities for key joints."""
    time = df["time_s"].values
    angle_cols = [c for c in df.columns if c not in ("frame", "time_s")]

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in angle_cols:
        vel = compute_angular_velocity(df[col].values, rate)
        ax.plot(time, vel, label=col.replace("_", " ").title(), alpha=0.8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular Velocity (deg/s)")
    ax.set_title(f"Angular Velocities — {mode.title()}")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    path = output_dir / f"angular_velocity_{mode}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Angular velocity plot: {path}")


def main():
    parser = argparse.ArgumentParser(description="Joint angle & angular velocity extraction")
    parser.add_argument("--input", type=str, default=None, help="C3D file path")
    parser.add_argument("--mode", choices=["pitching", "hitting"], default="pitching")
    args = parser.parse_args()

    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        filepath = args.input
    else:
        filepath = f"data/raw/{args.mode}_sample.c3d"

    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        return

    df, rate, time = analyze(filepath, args.mode)

    # Save CSV
    csv_path = output_dir / f"joint_angles_{args.mode}.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"  Joint angles CSV: {csv_path} ({len(df)} rows)")

    # Plots
    plot_kinematic_sequence(df, time, args.mode, output_dir)
    plot_angular_velocity(df, rate, args.mode, output_dir)

    # Summary statistics
    angle_cols = [c for c in df.columns if c not in ("frame", "time_s")]
    print(f"\n  Summary ({args.mode}):")
    for col in angle_cols:
        vals = df[col].dropna()
        print(f"    {col}: min={vals.min():.1f}°, max={vals.max():.1f}°, "
              f"range={vals.max()-vals.min():.1f}°")

    print("\nDone!")


if __name__ == "__main__":
    main()
