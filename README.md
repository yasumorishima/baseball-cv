# baseball-cv: Baseball Skeleton Analysis with Computer Vision

Biomechanical skeleton analysis pipeline for baseball pitching and hitting motions. Combines motion capture data (C3D) with computer vision (MediaPipe) to extract joint angles, angular velocities, and correlate body mechanics with performance metrics.

## Pipeline Overview

| Section | Script | Description |
|---------|--------|-------------|
| **Getting Started** | `skeleton_c3d.py` | Load Driveline OBP C3D files with [ezc3d](https://github.com/pyomeca/ezc3d) → 3D skeleton visualization & animation |
| | `skeleton_video.py` | MediaPipe Pose detection on video → skeleton overlay & keypoint CSV |
| | `skeleton_analysis.py` | Joint angle & angular velocity extraction from C3D data |
| **Pitching Analysis** | `statcast_correlation.py --mode pitching` | Feature extraction: 100 biomechanical features across 60 pitchers |
| | `body_efficiency_analysis.py` `efficient_thrower_gif.py` | **Efficient throwing**: 5-component model (R²=0.491→0.669), same arm speed → 10 mph gap |
| **Hitting Analysis** | `statcast_correlation.py --mode hitting` | Feature extraction: biomechanical features across 40 hitters |
| | `body_efficiency_hitting.py` `efficient_hitter_gif.py` | **Efficient hitting**: 4-component model (R²=0.159→0.599), same bat speed → 20 mph gap |

## Articles

| Section | Title | Links |
|---------|-------|-------|
| Getting Started | 3D Skeleton Detection from Baseball Motion Capture Data | [Zenn](https://zenn.dev/shogaku/articles/baseball-cv-skeleton-biomechanics) · [DEV.to](https://dev.to/yasumorishima/3d-skeleton-detection-from-baseball-motion-capture-data-with-driveline-c3d-29ja) |
| Pitching Analysis | Why Two Pitchers with the Same Arm Speed Differ by 10 mph | [Zenn](https://zenn.dev/shogaku/articles/baseball-cv-efficient-throwing) · [DEV.to](https://dev.to/yasumorishima/why-two-pitchers-with-the-same-arm-speed-differ-by-10-mph-a-motion-capture-analysis-ljk) |
| Hitting Analysis | Why Two Hitters with the Same Bat Speed Differ by 20 mph in Exit Velocity | [Zenn](https://zenn.dev/shogaku/articles/baseball-cv-hitting-frontleg) · [DEV.to](https://dev.to/yasumorishima/does-exit-velocity-come-from-the-front-foot-exploring-driveline-motion-capture-data-4d40) |

## Results

### Getting Started: Skeleton Visualization & Joint Angles

C3D files store 3D marker coordinates captured by optical motion capture. Each frame records the XYZ position of every marker attached to the athlete's body.

- **Pitching data**: 45 markers, 360 Hz, ~726 frames per trial
- **Hitting data**: 55 markers (45 body + 10 bat), 360 Hz, ~804 frames per trial

ezc3d loads the binary C3D format and returns arrays ready for NumPy/matplotlib:

```python
import ezc3d

c3d = ezc3d.c3d("pitching_sample.c3d")
points = c3d["data"]["points"]  # shape: (4, n_markers, n_frames)
labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
```

**Pitching motion** (45 body markers, 360 Hz):

![Pitching Skeleton](data/output/skeleton_pitching_anim.gif)

Wind-up → stride → arm acceleration → ball release, rendered as a 3D stick figure.

**Hitting motion** (45 body + 10 bat markers, 360 Hz):

![Hitting Skeleton](data/output/skeleton_hitting_anim.gif)

The 10 bat markers (red) are rendered separately from the body skeleton, making bat path visible alongside the hitter's kinematics.

> I contributed a bug fix to ezc3d ([PR #384](https://github.com/pyomeca/ezc3d/pull/384)) — fixing an `__eq__` early return bug — and then used the library for this analysis.

---

For setups without C3D equipment, Google's MediaPipe Pose detects 33 skeletal landmarks from standard video in real time.

```python
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

Outputs a keypoint CSV and an annotated video overlay. The pitching and hitting analyses below use C3D data, but MediaPipe is a practical entry point when professional equipment isn't available.

---

Joint angles (how much each joint opens/closes) and angular velocities (how fast each joint rotates, in degrees/sec) are computed frame-by-frame from the C3D marker positions.

**Pitching — Joint Angles across full motion:**

![Kinematic Sequence](data/output/kinematic_sequence_pitching.png)

| Joint | Min | Max | Range |
|-------|-----|-----|-------|
| Elbow Flexion (R) | 50.5° | 156.7° | 106.3° |
| Shoulder Abduction (R) | 4.6° | 117.7° | 113.1° |
| Trunk Rotation | 0.0° | 58.0° | 57.9° |
| Knee Flexion (R) | 99.1° | 163.8° | 64.7° |

Elbow flexion spans ~106° — the arm goes from nearly straight at extension to sharply bent near max external rotation, then extends again at release. Trunk rotation (58°) appears modest by comparison, but emerges as the strongest correlate of pitch velocity in the pitching analysis.

**Angular Velocities — when does each joint move fastest?**

![Angular Velocity](data/output/angular_velocity_pitching.png)

The time-series shows each joint's rotational speed per frame. This reveals the proximal-to-distal kinematic sequence: hips and trunk rotate first, followed by the shoulder, then elbow and wrist.

**Pitch speed correlation (16 pitchers, pitching analysis preview):**

| Feature | r | p-value |
|---------|---|---------|
| Peak Trunk Angular Velocity | 0.119 | 0.673 |
| Peak Elbow Angular Velocity | 0.094 | 0.739 |
| Peak Shoulder Abduction | 0.180 | 0.520 |
| **Trunk Rotation Range** | **0.425** | **0.114** |

With 16 samples the p-values don't reach significance, but trunk rotation range shows the strongest correlation (r=0.425). The full pitching analysis expands this with 60 pitchers.

> 📝 [3D Skeleton Detection from Baseball Motion Capture Data](https://zenn.dev/shogaku/articles/baseball-cv-skeleton-biomechanics) (Zenn · [DEV.to](https://dev.to/yasumorishima/3d-skeleton-detection-from-baseball-motion-capture-data-with-driveline-c3d-29ja))

---

### Pitching Analysis: Efficient Throwing

60 Driveline OBP pitchers analyzed. **Finding**: Pitchers with identical arm speed (24–26 m/s) vary by 13 mph in pitch speed. Four independent body mechanics factors explain 17.8% additional variance beyond arm speed alone.

**5-Component Model (R² = 0.669 vs 0.491 arm-speed-only):**

| Component | R² Added | Physical Meaning |
|-----------|----------|-----------------|
| Arm speed + height | 0.491 | Baseline: wrist linear speed |
| + Stride (translational) | +0.038 | Forward momentum delivered to ball |
| + Leg lift (elastic) | +0.036 | Elastic loading in the back leg |
| + Arm chain pattern | +0.028 | Does body drive the elbow vs arm self-generating? |
| + Knee smoothness | **+0.077** | Smooth lead leg = body drives arm (pelvis/arm ratio) |

**Same arm speed → 10.3 mph gap (Q1=79 mph vs Q5=89 mph):**

![Efficient Throwing Story](data/output/efficient_throwing_story.png)

**Q1 vs Q5 body mechanics breakdown (arm speed 24.6 vs 24.7 m/s — virtually identical):**

![Body Efficiency Breakdown](data/output/body_efficiency_breakdown.png)

**Skeleton animation** — Q1 (arm=26.6 m/s, pitch=80.8 mph, stride=0.30 m) vs Q5 (arm=25.0 m/s, pitch=91.8 mph, stride=0.89 m):

![Efficient Thrower Comparison](data/output/efficient_thrower_comparison.gif)

> **Knee smoothness** (suppressor variable): 単純相関では r=+0.12（速い腕 = 膝も激しい）だが、腕速度を同じにして比較すると r=−0.45\*\*\* — 滑らかな膝の動きは「腕が自分で速度を作る」のではなく「体幹が腕を振り回している」状態を示す。骨盤/腕速度比が17%高い。

> **短いストライドの根本原因**: アンクルブレーキング（着地後の足の急減速）が低い（Q1: 0.06 vs Q5: 3.58 m/s²）→ 着地した足が安定した台座を作れない → ストライドが制限される（r=+0.55\*\*\*）。「足が地面に刺さる」感覚がないと前への勢いが腕に伝わらない。

> 📝 [Why Two Pitchers with the Same Arm Speed Differ by 10 mph](https://zenn.dev/shogaku/articles/baseball-cv-efficient-throwing) (Zenn · [DEV.to](https://dev.to/yasumorishima/why-two-pitchers-with-the-same-arm-speed-differ-by-10-mph-a-motion-capture-analysis-ljk))

---

### Hitting Analysis: Efficient Hitting

40 Driveline OBP hitters analyzed. **Finding**: Bat speed alone barely predicts exit velocity (R²=0.097). Adding weight transfer (stride) alone jumps R² to 0.550 (+0.378) — the single largest factor. The top 20% of body efficiency hitters produce 20.9 mph more exit velocity despite having *lower* bat speed than the bottom 20%.

**4-Component Model (R² = 0.159 → 0.599):**

| Component | R² | Added | Physical Meaning |
|-----------|-----|-------|-----------------|
| Bat speed + height | 0.159 | baseline | Wrist linear speed at contact |
| + Hip rotation | 0.173 | +0.014 | Trunk rotational velocity driving the swing |
| **+ Weight transfer** | **0.550** | **+0.378** | **Stride length — forward momentum into the ball** |
| + Lead leg block | 0.599 | +0.048 | Ankle braking creates a stable base to rotate around |

> **Weight transfer dominates**: stride length adds 37.8% of explained variance in a single step. A longer, controlled stride channels the body's forward momentum into rotational power at contact.

**Q5 hitters produce 18.6 mph more — at similar bat speed and body size:**

| Group | Bat speed | Height | Weight | Exit velocity | Gap |
|-------|-----------|--------|--------|--------------|-----|
| Q1 (bottom 20%) | 9.42 m/s | 70.8 in | 196.0 lb | 79.8 mph | — |
| Q5 (top 20%) | 9.04 m/s | **69.9 in** | **191.9 lb** | **98.4 mph** | **+18.6 mph** |

The efficiency score controls for bat speed, height, and weight together — so Q1/Q5 groups are matched on physique. Q5 hitters are actually *shorter and lighter* yet produce 18.6 mph more exit velocity. The gap is mechanics, not size.

**Story plot** (scatter colored by hip rotation speed | R² staircase | Q1–Q5 bars):

![Efficient Hitting Story](data/output/efficient_hitting_story.png)

**Body mechanics breakdown** (Q1 vs Q5 z-scores at virtually identical bat speed):

![Body Efficiency Hitting Breakdown](data/output/body_efficiency_hitting_breakdown.png)

**Skeleton animation** — Q1 (bat=7.23 m/s, exit vel=74.6 mph, stride=0.72 m) vs Q5 (bat=7.80 m/s, exit vel=97.2 mph, stride=0.99 m):

![Efficient Hitter Comparison](data/output/efficient_hitter_comparison.gif)

Red = lead leg (front foot/stride leg). Orange star = foot strike landing point. Both clips are aligned to foot strike.

> **Body Efficiency Score**: Residual of exit velocity after controlling for bat speed, height, and weight together. A score of +9.70 mph means this hitter produces 9.7 mph *more* exit velocity than a hitter of the same size and bat speed — purely from body mechanics.

> **Root cause of short stride**: low ankle braking deceleration → foot fails to create a stable braking base → stride is curtailed → less forward momentum transfers into rotation at contact.

> 📝 [Why Two Hitters with the Same Bat Speed Differ by 20 mph in Exit Velocity](https://zenn.dev/shogaku/articles/baseball-cv-hitting-frontleg) (Zenn · [DEV.to](https://dev.to/yasumorishima/does-exit-velocity-come-from-the-front-foot-exploring-driveline-motion-capture-data-4d40))

## Setup

```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.9+
- ezc3d >= 1.5
- mediapipe >= 0.10
- opencv-python >= 4.8
- matplotlib, numpy, pandas, scipy

### Data

Sample C3D files are included in `data/raw/`. For the full dataset:
- [Driveline OpenBiomechanics Project](https://github.com/drivelineresearch/openbiomechanics) (CC BY-NC-SA 4.0)

For MediaPipe video demo, download a free baseball video from [Pexels](https://www.pexels.com/search/videos/baseball/) and save to `data/videos/`.

## Usage

```bash
# Getting Started: skeleton visualization
python skeleton_c3d.py                    # Both pitching & hitting
python skeleton_c3d.py --mode pitching    # Pitching only

# Getting Started: MediaPipe video skeleton detection
python skeleton_video.py --input data/videos/batting.mp4

# Getting Started: joint angle analysis
python skeleton_analysis.py --mode pitching
python skeleton_analysis.py --mode hitting

# Pitching Analysis: feature extraction (downloads C3D files, builds features_pitching.csv)
python statcast_correlation.py --mode pitching --download 60

# Pitching Analysis: body efficiency (scatter plots + R2 breakdown + skeleton animation)
python body_efficiency_analysis.py
python efficient_thrower_gif.py

# Hitting Analysis: feature extraction
python statcast_correlation.py --mode hitting --download 40

# Hitting Analysis: body efficiency
python body_efficiency_hitting.py
python efficient_hitter_gif.py
```

## Data Sources & Credits

- **Driveline OpenBiomechanics Project**: https://openbiomechanics.org/ (CC BY-NC-SA 4.0)
- **Pexels**: Free video clips (Pexels License)
- **Baseball Savant / Statcast**: Public leaderboard data

See [DATA_SOURCES.md](DATA_SOURCES.md) for full details and license restrictions.

## License Disclaimer

> ⚠️ **Driveline OpenBiomechanics データは非営利・教育用途のみ（CC BY-NC-SA 4.0）。プロ球団の従業員・契約者は使用不可。** このプロジェクトはポートフォリオ・教育目的で作成しています。

The Driveline OBP data is licensed under CC BY-NC-SA 4.0 (non-commercial only). Employees or contractors of professional sports organizations are restricted from using this data. This project is for educational and portfolio purposes.
