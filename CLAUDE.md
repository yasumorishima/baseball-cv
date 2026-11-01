# CLAUDE.md — baseball-cv

Biomechanics of pitching and hitting from OpenBiomechanics Project motion-capture files (C3D): 3D skeletons, joint angles, kinematic sequence, "body efficiency" models. **Dormant**: no work is scheduled.

## Session protocol (cloud)
1. Read this file first. There is nothing scheduled: act only on what the owner asks in this session.
2. Before ending, add a dated line to "Progress log" (and update "Next tasks" if anything changed), then open a PR to `master` from the session branch and merge it yourself; report the merged commit.
3. No `gh` CLI in cloud (use the GitHub MCP tools); `workflow_dispatch` returns 403 from cloud.
4. Reply to the owner in Japanese (です/ます). Do not claim anything you did not run in this session; say where each number was measured.

## Progress log
- 2026-02-24 – 2026-03-04: pipeline, pitching (60 pitchers) and hitting (40 hitters) analyses, three articles (links in README). Last workflow runs 2026-03-02 (success).
- 2026-10-08: licences added (PR #1, #2, merge `74dd2f8`): code MIT (`LICENSE`), everything under `data/` CC BY-NC-SA 4.0 with the source's exclusion (`data/LICENSE.md`).
- 2026-11-01: this file written for cloud. Nothing run.

## What this is
`statcast_correlation.py` downloads C3D trials from the OpenBiomechanics GitHub repo and builds `data/output/features_*.csv`; `body_efficiency_*.py` fit the efficiency models; `efficient_*_gif.py`, `llb_*.py`, `head_stability_*.py` draw the figures in `data/output/` that README embeds. `skeleton_video.py` is a MediaPipe demo on a user-supplied video (not in the repo).

## Non-negotiable discipline
1. **Data licence (carry it everywhere).** All OpenBiomechanics data and anything derived from it (everything under `data/`) is CC BY-NC-SA 4.0: attribution, non-commercial only, derivatives under the same licence. The source adds: *"Any employee or contractor employed by, associated with, or a significant shareholder of a professional sports organization or financial analysis firm is forbidden to use The OpenBiomechanics Project data for any use whatsoever."* Keep the README disclaimer and `data/LICENSE.md`; put new derived outputs under `data/` so the licence covers them; never use this data for commercial work or work for a professional sports organisation.
2. Only the two C3D files in `data/raw/` are committed (needed by the GIF workflow); downloaded trials stay git-ignored. Do not commit bulk raw data.
3. Keep the code licence MIT and the plain MIT text in `LICENSE` (GitHub's licence detection depends on it).

## Next tasks
None scheduled; only act when the owner asks. Known problems, as candidates only:
- `test-llb.yml` and the no-argument defaults of `skeleton_c3d.py` / `skeleton_analysis.py` read `data/raw/pitching_sample.c3d` and `data/raw/hitting_sample.c3d` (`statcast_correlation.py` also globs for `<mode>_sample*.c3d`), which were deleted in `65f542b` (2026-02-28). `test-llb.yml` last passed on 2026-02-28 06:10 UTC, before that deletion, so it would now fail at its first step; README's "Sample C3D files are included in `data/raw/`" is out of date.
- There is no test that runs on push/PR.

## How to run (cloud, not yet tried there)
```bash
pip install -r requirements.txt     # mediapipe/opencv are only needed for skeleton_video.py
python statcast_correlation.py --mode pitching --download 60   # downloads C3D into data/raw/
python body_efficiency_analysis.py && python efficient_thrower_gif.py
python statcast_correlation.py --mode hitting --download 40
python body_efficiency_hitting.py && python efficient_hitter_gif.py
```
- Downloads use `api.github.com` (directory listing) and `github.com/drivelineresearch/openbiomechanics/raw/main/...`. The script sends `GITHUB_TOKEN` if set; without it GitHub's unauthenticated API limit applies.
- **GitHub Actions:** `generate-gif.yml` also runs on any push to `master` that changes `efficient_*_gif.py`, `body_efficiency_*.py` or `llb_{knee_detail_hitting,wall_hitting,brace_analysis}.py`, and commits regenerated images back to `master` as `github-actions[bot]`. Merging such a PR therefore produces a second commit; pull before the next change. The other two workflows are manual only (not triggerable from cloud).

## Credits (keep in README)
OpenBiomechanics Project (https://www.openbiomechanics.org/, CC BY-NC-SA 4.0); ezc3d; MediaPipe; Pexels (videos, not committed); Baseball Savant (MLB Advanced Media). Full details in `DATA_SOURCES.md`.
