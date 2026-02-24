# Data Sources

## Driveline OpenBiomechanics Project (OBP)

- **URL**: https://github.com/drivelineresearch/openbiomechanics
- **Website**: https://www.openbiomechanics.org/
- **License**: CC BY-NC-SA 4.0
- **Data used**: C3D motion capture files from `baseball_pitching/data/c3d/` and `baseball_hitting/data/c3d/`
- **Content**: 3D marker positions (47 body markers + 10 bat markers) captured at 360 Hz
- **Metadata**: Pitch speed, exit velocity, height, weight embedded in filenames

### C3D File Naming Convention

**Pitching**: `USERid_SESSIONid_HEIGHT_WEIGHT_PITCHNUMBER_PITCHTYPE_PITCHSPEED.c3d`
- Example: `000002_003034_73_207_002_FF_809.c3d` → Fastball at 80.9 mph

**Hitting**: `USERid_SESSIONid_HEIGHT_WEIGHT_SIDE_SWINGNUMBER_EXITVELOCITY.c3d`
- Exit velocity: last 3 digits with implied decimal (905 = 90.5 mph)

### License Restrictions

> CC BY-NC-SA 4.0: Non-commercial use only. Employees or contractors of professional
> sports organizations or financial analysis firms are prohibited from using this data.
> This project uses the data for educational/portfolio purposes only.

## Free Baseball Videos (MediaPipe Demo)

- **Source**: Pexels (https://www.pexels.com/)
- **License**: Pexels License (free for personal and commercial use, no attribution required)
- **Usage**: Short clips (5-10 seconds) for MediaPipe pose detection demonstration
- **Note**: Video files are not included in the repository (.gitignore). Download URLs below.

### Download Instructions

1. Visit https://www.pexels.com/search/videos/baseball/
2. Download a short batting or pitching clip (720p recommended for RAM-constrained environments)
3. Save to `data/videos/` folder

## Baseball Savant / Statcast

- **URL**: https://baseballsavant.mlb.com/
- **Usage**: Public leaderboard metrics (exit velocity, pitch velocity) for correlation analysis
- **Credit**: MLB Advanced Media, Baseball Savant
