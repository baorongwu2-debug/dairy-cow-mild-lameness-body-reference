# Body-referenced posture and validation-gated temporal fusion for mild lameness recognition in dairy cows

Analysis code and reproducibility materials accompanying a manuscript prepared for *Computers and Electronics in Agriculture*. This repository does not imply journal acceptance. The study is a secondary computational analysis of released pose trajectories; no new animal recording, handling or intervention was performed.

## Data and scope

The source is [Russello et al., Lameness detection in dairy cows using pose estimation and bidirectional LSTMs](https://doi.org/10.1016/j.atech.2026.101831), with data released at [hrussel/lstm-lameness-detection](https://github.com/hrussel/lstm-lameness-detection).

The complete dataset has 272 trajectories from 98 cows. The primary endpoint comprises 239 trajectories from 92 cows: 143 normal and 96 mild-score clips. Raw trajectories, third-party code, photographs and trained binary model files are not redistributed. See [data/README.md](data/README.md) for retrieval and hash verification.

## Main results and evaluation coverage

| Model / representation | ROC AUC | Evaluation |
|---|---:|---|
| Body-referenced posture SVM | 0.881 | Mean across five cow-grouped split seeds |
| Image-coordinate posture + motion SVM | 0.726 | Mean across five seeds |
| Body-referenced posture + motion SVM | 0.824 | Mean across five seeds |
| Block-aware multiple-kernel SVM | 0.880 | Mean across five seeds |
| Extra Trees | 0.879 | Seed 42 only |
| k-nearest neighbors | 0.852 | Seed 42 only |
| Gaussian naive Bayes | 0.824 | Seed 42 only |
| Compact bidirectional LSTM | 0.845 | Seed 42 only |
| Temporal CNN | 0.866 | Seed 42 only |
| Validation-gated posture–CNN fusion | 0.880 | Exploratory, seed 42 only |
| Posture SVM, strict session holdout | 0.867 | Eight acquisition batches within the same farm |

The original planned combined representation did not outperform the posture-only comparator. Fusion sensitivity increased from 0.781 to 0.833 at seed 42, with additional false positives; its paired AUC difference was -0.003 (cow-cluster interval [-0.022, 0.017]). This does not establish a statistically supported AUC improvement. Five seeds are repeated partitions, not five independent datasets.

Cow isolation applies to downstream model development. Upstream pose-estimator independence is not established. Session holdout is within one farm. These results do not establish cross-farm generalization or prediction before visible clinical signs. The fusion gate and CNN early stopping use the same inner validation subset; fusion remains exploratory.

## Quick verification (no dataset or third-party Python packages needed)

```sh
python scripts/verify_results.py
```

This recomputes AUC from saved predictions, checks prediction coverage and cow separation, and compares stored summaries. It does not retrain models.

## Reproduce model fitting

Use Python 3.12 and the recorded versions in `requirements.txt`:

```sh
python -m venv .venv
# Activate .venv for your operating system, then:
python -m pip install -r requirements.txt
python scripts/reproduce_results.py --data-root /path/to/lstm-lameness-detection-main/data --archive /path/to/russello_official.zip --stage all
```

The runner validates all trajectory and label hashes before fitting. Results go to `reproduced/`, preserving `results/` as the original evidence. The original ZIP is optional: without it, the recomputed audit reports a null archive hash and the individually verified data hashes remain authoritative. `--stage base`, `enhanced`, `sequence`, or `fusion` runs one stage; enhanced and fusion require the feature file produced by base. `--check-only` checks source data without training. Runtime depends on hardware; no GPU is required.

## Repository contents

- `src/`: actual analysis implementations, preserving fitting logic.
- `scripts/`: portable reproduction runner and independent saved-result verifier.
- `results/`: recorded predictions, splits, summaries and historical audit reports.
- `data/`: source information and hashes, without raw data.
- `pose_protocol.md`, `enhanced_protocol.md`: dated local analysis plans, not public preregistrations.
- `provenance/`: packaging changes and original script hashes.
- `CITATION.cff`: software citation, using manuscript author names.

## Funding

This work was supported by the Shandong Provincial Major Agricultural Applied Technology Innovation Project (Research on Key Technologies for Intelligent Integrated Prevention and Control of Livestock and Poultry Diseases in Large-Scale Farming).

Original funding designation supplied by the author: 山东省农业重大应用技术创新项目（规模化养殖畜禽疫病智能化综合防控关键技术研究）. No grant number was supplied. The English wording is a translation of this designation.

## License and citation

Original analysis code and documentation are provided under the MIT License. The license does not cover third-party source data or images. Cite the upstream dataset paper and this software. No permanent DOI has been assigned to this local package; a DOI must be obtained through an actual archive deposit. See `RELEASE.md`.
