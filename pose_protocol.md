# Analysis plan fixed before pose model evaluation

Date 2026-09-07. Dataset: official hrussel/lstm-lameness-detection archive downloaded on this date. 272 trajectories, 98 cow IDs, inherited hard_vote scores: 143 score 1, 96 score 2, 20 score 3, 13 score 4.

Primary endpoint: normal versus mild lameness, score 1 versus score 2 (239 clips). Secondary endpoint: score 1 versus scores 2–4 (272 clips). Mild lameness is an observed score category, not future disease onset.

Primary representation: nine landmarks (four hooves, nose, head top, three spine landmarks), coordinates expressed in the per-frame trunk reference system. Origin is midpoint of Spine1 and Spine2; horizontal basis follows Spine2 to Spine1; vertical basis is its 90-degree rotation; scale is median trunk length over the full clip. This removes shared translation, rotation and uniform size without requiring test-set statistics. Savitzky–Golay smoothing uses 7 frames and degree 2, per clip. Use actual frame-index differences at the published 30 Hz; reject nonmonotonic frames and missing coordinates rather than silently filling them.

Static block: coordinate median, IQR and 10th/90th percentiles (72 features). Dynamic block: first-derivative median, IQR, 10th/90th percentiles; second-derivative IQR; coordinate autocorrelation at 1, 5, 15 and 30 frame lags (162 features). Combined dimension 234. Autocorrelation is normalized separately per coordinate; constant coordinates return zero. Frame gaps, if found, will be disclosed and autocorrelation restricted to the released sample grid.

Primary classifier: StandardScaler plus RBF SVM, C in 0.1, 1, 10, 100 and gamma in scale, 0.01, 0.1; class_weight balanced. Decision score threshold zero. All scaling and hyperparameter selection inside inner training folds. Primary selection metric ROC AUC. Outer five-fold StratifiedGroupKFold; inner three-fold StratifiedGroupKFold; all grouped by cow ID. Five outer seeds 42,123,456,789,2024. No adaptive feature redesign using these results.

Comparators: identical descriptors in raw image coordinates; body-reference static-only; body-reference dynamic-only; body-reference combined logistic regression (C grid 0.01,0.1,1,10); body-reference combined random forest (300 trees, max_depth 3 or None, min_samples_leaf 2 or 5). Full results retained irrespective of rank. These are controlled baselines, not replications of published T-LEAP/BLSTM performance.

Uncertainty: primary seed 42 OOF scores, 2000 paired cow-cluster bootstrap replicates; retain every clip of sampled cows, including repeated clusters. Percentile intervals describe this OOF evaluation conditional on fitted models; they do not propagate model-refitting uncertainty. Repeated-seed mean and SD represent split sensitivity, not independent sample replication. Paired AUC difference of combined versus static is the principal contrast. No post hoc selection of a favorable seed.

Mechanistic robustness diagnostic: apply a shared coordinate rotation of 20 degrees, isotropic scaling 1.5 and translation (250,-150) to every trajectory; test descriptor invariance and fixed model score stability. This synthetic geometric perturbation is not an external-farm validation or empirical camera-distance experiment.

Upstream limitation: published pose extractor trained using 28 videos from this collection. Cow independence applies to downstream classifier splits; end-to-end unseen-cow independence of the pose extractor cannot be established from the released trajectories.

Additional datasets will be logged with availability and compatibility. Sensor datasets, unlabeled pose data and internet clips must not be pooled as if they share these clinical labels and acquisition conditions.

Implementation amendment before completion of any outer evaluation: the rigid-transform check exposed floating-point noise in structurally constant coordinates amplified by StandardScaler. Added VarianceThreshold(1e-12) fitted inside every training fold before scaling for all classifiers. This is a numerical-stability correction, not a choice based on classification performance.
