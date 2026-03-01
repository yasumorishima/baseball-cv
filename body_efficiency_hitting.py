"""Body Efficiency Analysis — hitting.

Discovers that hitters with identical bat speed vary in exit velocity.
Independent body mechanics factors explain this gap.

Findings (populated after first run):
  - R2 increases from baseline (bat speed + height) to multi-component model
  - Q1 vs Q5 (same bat speed): ? vs ? mph exit velocity

Requires:
    data/output/features_hitting.csv (from statcast_correlation.py --mode hitting)

Output:
    data/output/efficient_hitting_story.png
    data/output/body_efficiency_hitting_breakdown.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.font_manager as _fm
_jp_fonts = [f.name for f in _fm.fontManager.ttflist if "Noto" in f.name and "CJK" in f.name]
if _jp_fonts:
    plt.rcParams["font.family"] = _jp_fonts[0]

OUTPUT_DIR = Path("data/output")
FEATURES_CSV = OUTPUT_DIR / "features_hitting.csv"

TARGET = "exit_velocity_mph"
BAT_SPEED = "peak_wrist_linear_speed"

# Candidate body mechanics components for hitting.
# Based on biomechanics: hip rotation -> trunk -> arm -> bat (kinematic chain).
# Verified after first run; re-order by actual contribution if needed.
CANDIDATE_STEPS = [
    ([BAT_SPEED, "height_in"],
     "Bat speed + Height"),
    ([BAT_SPEED, "height_in", "peak_trunk_velocity"],
     "+ Hip rotation (trunk velocity)"),
    ([BAT_SPEED, "height_in", "peak_trunk_velocity", "llb_stride_forward"],
     "+ Weight transfer (stride)"),
    ([BAT_SPEED, "height_in", "peak_trunk_velocity", "llb_stride_forward",
      "llb_ankle_braking_decel"],
     "+ Lead leg block"),
    ([BAT_SPEED, "height_in", "peak_trunk_velocity", "llb_stride_forward",
      "llb_ankle_braking_decel", "trunk_rotation_range"],
     "+ Coil / Load"),
]


def compute_body_efficiency(df):
    """Residual of exit_velocity after regressing on bat speed + height + weight.

    Controlling for body size ensures Q1/Q5 groups differ in mechanics, not physique.
    """
    size_cols = [c for c in ["height_in", "weight_lb"] if c in df.columns]
    baseline_cols = [BAT_SPEED] + size_cols
    d = df[baseline_cols + [TARGET]].dropna().copy()
    if len(d) < 10:
        raise ValueError(f"Too few samples: {len(d)}")
    lm = LinearRegression().fit(d[baseline_cols], d[TARGET])
    df = df.copy()
    df.loc[d.index, "body_efficiency"] = (
        d[TARGET].values - lm.predict(d[baseline_cols].values)
    )
    df["eff_q"] = pd.qcut(df["body_efficiency"], q=5,
                          labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    return df


def compute_incremental_r2(df):
    """Compute R2 at each step of the multi-component model."""
    results = []
    for cols, label in CANDIDATE_STEPS:
        available = [c for c in cols if c in df.columns]
        if len(available) < 2:
            print(f"  Skipping '{label}': missing columns {set(cols) - set(df.columns)}")
            continue
        d = df[available + [TARGET]].dropna()
        if len(d) < 10:
            print(f"  Skipping '{label}': only {len(d)} complete rows")
            continue
        X = np.column_stack([d[available].values, np.ones(len(d))])
        coef, *_ = np.linalg.lstsq(X, d[TARGET].values, rcond=None)
        y_pred = X @ coef
        r2 = r2_score(d[TARGET].values, y_pred)
        results.append((r2, label, len(d)))
    return results


def print_top_correlations(df, top_n=15):
    """Print top correlated features with exit velocity (helps identify model components)."""
    skip = {"user_id", "session_id", "side", "swing_number", "filename",
            "llb_foot_strike_frame", "llb_foot_strike_time_s",
            TARGET, "body_efficiency", "eff_q"}
    feature_cols = [c for c in df.columns if c not in skip]

    results = []
    for feat in feature_cols:
        valid = df[[TARGET, feat]].dropna()
        if len(valid) < 5:
            continue
        r, p = stats.pearsonr(valid[feat], valid[TARGET])
        results.append((feat, r, p, len(valid)))

    results.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n  Top {top_n} features correlated with {TARGET}:")
    print(f"  {'Feature':<40} {'r':>7} {'p':>8} {'n':>4}")
    print(f"  {'-'*40} {'-'*7} {'-'*8} {'-'*4}")
    for feat, r, p, n in results[:top_n]:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {feat:<40} {r:+.3f} {p:8.4f} {n:4d} {sig}")


def plot_story(df, r2_steps, out_path):
    """3-panel: scatter colored by smoothness | R2 staircase | Q1/Q5 bars."""
    fig = plt.figure(figsize=(18, 6))

    # Panel 1: Bat speed vs Exit velocity, colored by trunk rotation speed
    ax1 = fig.add_subplot(1, 3, 1)
    color_col = "peak_trunk_velocity" if "peak_trunk_velocity" in df.columns else BAT_SPEED
    d_plot = df[[BAT_SPEED, TARGET, color_col]].dropna()
    sc = ax1.scatter(
        d_plot[BAT_SPEED], d_plot[TARGET],
        c=d_plot[color_col], cmap="RdYlGn", s=60, alpha=0.8,
        edgecolors="gray", linewidths=0.5,
    )
    plt.colorbar(sc, ax=ax1, label=f"{color_col} (green=fast)")
    z = np.polyfit(d_plot[BAT_SPEED], d_plot[TARGET], 1)
    xline = np.linspace(d_plot[BAT_SPEED].min(), d_plot[BAT_SPEED].max(), 50)
    ax1.plot(xline, np.polyval(z, xline), color="black", lw=2, alpha=0.4)
    ax1.set_xlabel("バットスピード（手首ピーク速度 m/s）", fontsize=11)
    ax1.set_ylabel("打球速度 (mph)", fontsize=11)
    ax1.set_title("バットスピードが同じでも打球速度に差がある\n"
                  "（緑=腰の回転が速い、赤=遅い）", fontsize=10)

    # Panel 2: Incremental R2 bars
    ax2 = fig.add_subplot(1, 3, 2)
    r2_vals = [r for r, *_ in r2_steps]
    labels_short = ["バット\nスピード", "+腰の\n回転", "+体重\n移動",
                    "+前脚\nブレーキ", "+コイル\nロード"][:len(r2_vals)]
    colors = ["#2980b9", "#27ae60", "#8e44ad", "#e67e22", "#e74c3c"][:len(r2_vals)]
    ax2.bar(range(len(r2_vals)), r2_vals, color=colors, edgecolor="white", lw=2)
    for i, r2 in enumerate(r2_vals):
        prev = r2_vals[i - 1] if i > 0 else 0
        ax2.text(i, r2 + 0.01, f"{r2:.3f}", ha="center", fontweight="bold", fontsize=10)
        if i > 0:
            ax2.text(i, r2 - 0.04, f"+{r2 - prev:.3f}", ha="center",
                     fontsize=8, color="white", fontweight="bold")
    ax2.set_xticks(range(len(labels_short)))
    ax2.set_xticklabels(labels_short, fontsize=10)
    ax2.set_ylabel("R²（打球速度の説明できる割合）", fontsize=11)
    ax2.set_ylim(0, min(max(r2_vals) + 0.15, 1.0))
    ax2.set_title("各身体要素を追加するごとに\n予測精度が上がる", fontsize=10)

    # Panel 3: Q1/Q5 exit velocity bars (with bat speed annotation)
    ax3 = fig.add_subplot(1, 3, 3)
    q_grp = (df.groupby("eff_q", observed=True)
               .agg(exit_velo=(TARGET, "mean"),
                    bat_speed=(BAT_SPEED, "mean"))
               .reset_index().dropna())
    x = np.arange(len(q_grp))
    colors_q = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
    ax3.bar(x, q_grp["exit_velo"], color=colors_q, edgecolor="white", lw=2)
    for i, (ev, bat) in enumerate(zip(q_grp["exit_velo"], q_grp["bat_speed"])):
        ax3.text(i, ev + 0.3, f"{ev:.1f}", ha="center", fontweight="bold", fontsize=10)
        ax3.text(i, ev - 3.5, f"bat\n{bat:.1f}", ha="center", fontsize=8, color="white")
    ax3.set_xticks(x)
    ax3.set_xticklabels(["Q1\n（最低）", "Q2", "Q3", "Q4", "Q5\n（最高）"], fontsize=10)
    ax3.set_ylabel("打球速度の平均 (mph)", fontsize=11)
    q1_ev = q_grp[q_grp["eff_q"] == "Q1"]["exit_velo"].values[0] if "Q1" in q_grp["eff_q"].values else 80
    q5_ev = q_grp[q_grp["eff_q"] == "Q5"]["exit_velo"].values[0] if "Q5" in q_grp["eff_q"].values else 95
    ax3.set_ylim(max(q1_ev - 10, 60), q5_ev + 5)
    ax3.set_title(f"体の使い方で打球速度が変わる\n"
                  f"（Q1={q1_ev:.1f} mph vs Q5={q5_ev:.1f} mph）", fontsize=10)

    n_total = len(df.dropna(subset=[TARGET]))
    r2_max = r2_vals[-1] if r2_vals else 0
    fig.suptitle(
        f"打撃の効率性：体の使い方がバットスピードを超えた影響を持つ（n={n_total}, R²={r2_max:.3f}）",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_breakdown(df, out_path):
    """2-panel: Q1/Q5 body mechanics z-scores | scatter with Q quintiles."""
    # Use only features that exist in the data
    candidate_vars = ["peak_trunk_velocity", "llb_stride_forward",
                      "llb_ankle_braking_decel", "trunk_rotation_range"]
    body_vars = [v for v in candidate_vars if v in df.columns]
    if len(body_vars) < 2:
        print(f"  Warning: too few body mechanics columns for breakdown plot. "
              f"Available: {body_vars}")
        return

    label_map = {
        "peak_trunk_velocity": "腰の回転\n（体幹速度）",
        "llb_stride_forward": "体重移動\n（ストライド幅）",
        "llb_ankle_braking_decel": "前脚のブレーキ\n（足首制動）",
        "trunk_rotation_range": "コイル/ロード\n（回転可動域）",
    }
    body_labels = [label_map.get(v, v) for v in body_vars]
    # All of these: higher = more efficient (for stride, velocity, decel, range)
    flip = [+1] * len(body_vars)

    q1 = df[df["eff_q"] == "Q1"][body_vars].mean()
    q5 = df[df["eff_q"] == "Q5"][body_vars].mean()
    means = df[body_vars].mean()
    stds = df[body_vars].std()

    q1_z = ((q1 - means) / stds).values * flip
    q5_z = ((q5 - means) / stds).values * flip

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(body_labels))
    w = 0.35

    q1_ev = df[df["eff_q"] == "Q1"][TARGET].mean()
    q5_ev = df[df["eff_q"] == "Q5"][TARGET].mean()
    q1_bat = df[df["eff_q"] == "Q1"][BAT_SPEED].mean()
    q5_bat = df[df["eff_q"] == "Q5"][BAT_SPEED].mean()

    ax1.bar(x - w / 2, q1_z, w, label=f"Q1（体の効率 最低・打球速度 {q1_ev:.1f} mph）",
            color="#e74c3c", alpha=0.8, edgecolor="white")
    ax1.bar(x + w / 2, q5_z, w, label=f"Q5（体の効率 最高・打球速度 {q5_ev:.1f} mph）",
            color="#27ae60", alpha=0.8, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(body_labels, fontsize=11)
    ax1.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
    ax1.set_ylabel("Zスコア（プラス = より効率的）", fontsize=11)
    ax1.set_title(
        f"体の動き比較：バットスピードほぼ同じ（{q1_bat:.1f} vs {q5_bat:.1f} m/s）\n"
        f"→ 打球速度は {q5_ev - q1_ev:.1f} mph 差！",
        fontsize=11,
    )
    ax1.legend(fontsize=10)
    for xi, (z1, z5) in enumerate(zip(q1_z, q5_z)):
        for val, offset in [(z1, -w / 2), (z5, +w / 2)]:
            y_text = val + 0.05 if val >= 0 else (val - 0.13 if val > -0.45 else val + 0.12)
            ax1.text(xi + offset, y_text, f"{val:+.2f}", ha="center", fontsize=9)

    d_plot = df[[BAT_SPEED, TARGET, "eff_q"]].dropna()
    cmap = {"Q1": "#e74c3c", "Q2": "#e67e22", "Q3": "#f1c40f",
            "Q4": "#2ecc71", "Q5": "#27ae60"}
    for q in ["Q2", "Q3", "Q4"]:
        d_q = d_plot[d_plot["eff_q"] == q]
        ax2.scatter(d_q[BAT_SPEED], d_q[TARGET], c=cmap[q], s=40, alpha=0.6)
    for q in ["Q1", "Q5"]:
        d_q = d_plot[d_plot["eff_q"] == q]
        ev = d_q[TARGET].mean()
        bat = d_q[BAT_SPEED].mean()
        ax2.scatter(d_q[BAT_SPEED], d_q[TARGET],
                    c=cmap[q], s=80, label=f"{q}: {ev:.1f} mph (bat={bat:.1f})",
                    edgecolors="black", linewidths=1.5, zorder=5)
    z = np.polyfit(d_plot[BAT_SPEED], d_plot[TARGET], 1)
    xline = np.linspace(d_plot[BAT_SPEED].min(), d_plot[BAT_SPEED].max(), 50)
    ax2.plot(xline, np.polyval(z, xline), "k--", lw=2, alpha=0.4)
    ax2.set_xlabel("バットスピード（手首ピーク速度 m/s）", fontsize=11)
    ax2.set_ylabel("打球速度 (mph)", fontsize=11)
    ax2.set_title("「体の使い方が効率的な打者」は実在する\n"
                  "（同じバットスピードでも打球速度に幅がある）", fontsize=11)
    ax2.legend(fontsize=9, loc="upper left")

    r2_delta = 0
    if len(CANDIDATE_STEPS) >= 2:
        first_cols = [c for c in CANDIDATE_STEPS[0][0] if c in df.columns]
        last_cols = [c for c in CANDIDATE_STEPS[-1][0] if c in df.columns]
        if len(first_cols) >= 2 and len(last_cols) >= 2:
            d0 = df[first_cols + [TARGET]].dropna()
            d1 = df[last_cols + [TARGET]].dropna()
            X0 = np.column_stack([d0[first_cols].values, np.ones(len(d0))])
            X1 = np.column_stack([d1[last_cols].values, np.ones(len(d1))])
            c0, *_ = np.linalg.lstsq(X0, d0[TARGET].values, rcond=None)
            c1, *_ = np.linalg.lstsq(X1, d1[TARGET].values, rcond=None)
            r2_delta = r2_score(d1[TARGET].values, X1 @ c1) - r2_score(d0[TARGET].values, X0 @ c0)

    fig.suptitle(
        f"打撃の効率性：体の使い方がバットスピードを超えて打球速度を説明する（追加説明力 +{r2_delta * 100:.1f}%）\n"
        f"体効率スコアとは：バットスピードが同じ選手と比べたとき、体の使い方で何mph上乗せできているか",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not FEATURES_CSV.exists():
        print(f"Error: {FEATURES_CSV} not found. Run:")
        print("  python statcast_correlation.py --mode hitting --download 40")
        return

    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df)} rows from {FEATURES_CSV}")
    print(f"Columns: {list(df.columns)}")

    if TARGET not in df.columns:
        print(f"Error: '{TARGET}' column not found in features CSV.")
        return

    df = compute_body_efficiency(df)

    print_top_correlations(df)

    r2_steps = compute_incremental_r2(df)

    print("\n=== Multi-Component Model ===")
    prev = 0
    for r2, label, n in r2_steps:
        print(f"  {label}: R2={r2:.3f} (+{r2 - prev:.3f})  n={n}")
        prev = r2

    q1 = df[df["eff_q"] == "Q1"]
    q5 = df[df["eff_q"] == "Q5"]
    print(f"\nQ1: exit={q1[TARGET].mean():.1f} mph  bat={q1[BAT_SPEED].mean():.2f} m/s  n={len(q1)}")
    print(f"Q5: exit={q5[TARGET].mean():.1f} mph  bat={q5[BAT_SPEED].mean():.2f} m/s  n={len(q5)}")
    print(f"Gap: {q5[TARGET].mean() - q1[TARGET].mean():.1f} mph at similar bat speed")

    # Print Q1/Q5 file names for GIF creation
    print("\n=== Q1 candidate files (for GIF) ===")
    if "filename" in df.columns:
        q1_files = df[df["eff_q"] == "Q1"][["filename", TARGET, BAT_SPEED,
                                             "llb_stride_forward",
                                             "body_efficiency"]].dropna(subset=[TARGET]).head(5)
        print(q1_files.to_string(index=False))
        print("\n=== Q5 candidate files (for GIF) ===")
        q5_files = df[df["eff_q"] == "Q5"][["filename", TARGET, BAT_SPEED,
                                             "llb_stride_forward",
                                             "body_efficiency"]].dropna(subset=[TARGET]).head(5)
        print(q5_files.to_string(index=False))

    plot_story(df, r2_steps, OUTPUT_DIR / "efficient_hitting_story.png")
    plot_breakdown(df, OUTPUT_DIR / "body_efficiency_hitting_breakdown.png")
    print("\nDone. Next: create efficient_hitter_gif.py using Q1/Q5 files above.")


if __name__ == "__main__":
    main()
