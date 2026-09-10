# Data provenance and access

Download the trajectory data from https://github.com/hrussel/lstm-lameness-detection . Cite https://doi.org/10.1016/j.atech.2026.101831 . Required contents:

```
data/videos_lameness_scores.csv
data/videos_keypoints/001.csv
... remaining trajectories listed in results/pose_results/manifest.csv
```

The analysis snapshot was downloaded on 2026-09-07. Its ZIP SHA256 is `6f063d874a41d689a04d89522b0e86f2f24a999a8b7d9fa6e9de360b88907677`. The original remote commit was not recorded. A future download of main may differ: the runner rejects changed label or trajectory content. ZIP container differences alone need not imply data differences; omit `--archive` if using another container with individually matching data files.

Individual trajectory SHA256 values are in `results/pose_results/manifest.csv`; the label checksum is in `results/pose_results/data_audit.json`. Nine keypoints are ordered LFHoof, RFHoof, LHHoof, RHHoof, Nose, HeadTop, Spine1, Spine2, Spine3. Each has `_x` and `_y` columns; frame indices are interpreted at 30 Hz. Labels use the released `hard_vote` and cow `ID` fields.

The upstream snapshot did not include a separate LICENSE file. Availability does not imply unrestricted redistribution. Retrieve data directly from the source and comply with its terms. Only derived evaluation outputs and provenance are included here. CattleLameness demonstration photographs and the separate exploratory video study are excluded from this benchmark package.
