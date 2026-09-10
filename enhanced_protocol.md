# Confirmatory extension protocol fixed before enhanced-model outcomes

Date: 2026-09-08. The earlier pose study and its results are already known. This extension is prospective only with respect to the new comparisons below. It does not replace or relabel the earlier primary analysis.

## Aim

Test whether block-aware fusion improves normal-versus-mild discrimination relative to early concatenation, and quantify performance under an acquisition-session holdout on the same commercial farm.

## Proposed method

The proposed block-aware multiple-kernel SVM constructs separate RBF kernels for the 72 posture and 162 motion descriptors after training-only variance filtering and standardization. Each block bandwidth is the median nonzero squared distance in its training subset. The final kernel is alpha times the posture kernel plus one minus alpha times the motion kernel. Inner grouped validation selects C from 0.1, 1, 10; a shared bandwidth multiplier from 0.5, 1, 2; and alpha from 0.50, 0.75, 0.90, 1.00. Including alpha 1 permits rejection of an unhelpful motion block rather than forcing fusion.

## Comparators and endpoints

The confirmatory comparator is the previously specified posture-only RBF SVM under the same outer splits. Additional classical comparators are ExtraTrees, k-nearest neighbours and Gaussian naive Bayes on the posture block. A compact bidirectional LSTM operating on resampled body-referenced landmark sequences is a separate deep sequence comparator. The primary endpoint remains released score 1 versus score 2. Score 1 versus scores 2-4 is secondary.

## Cow-grouped evaluation

Use the same five outer seeds and five folds as the earlier study. Inner tuning is grouped by cow. Report pooled out-of-fold ROC AUC for each seed and mean plus sample SD across seeds. The planned paired contrast is proposed multiple-kernel SVM minus posture-only SVM at seed 42 using 2000 cow-cluster bootstrap samples.

## Commercial-farm session holdout

The source_video prefix defines eight acquisition sessions: 0522, 0527, 0603, 0604, 0709, 0711, 0712 and 0713. Hold out each session once. Remove from training every cow appearing in that test session. Tune on the remaining training cows only. Pool the eight untouched session predictions for descriptive AUC, balanced accuracy, sensitivity and specificity. This is a temporal/acquisition-shift validation within one commercial farm, not independent-farm validation. Sessions are inferred from released identifiers and must be described as acquisition batches unless confirmed by source metadata.

## Integrity

All choices above are fixed before running enhanced_study.py. The script must save every prediction, split, parameter and data count. If a session training set or inner split lacks both classes, report it rather than modifying the endpoint after seeing performance.

Implementation amendment before any enhanced result completed: the initial exhaustive kernel grid was stopped before producing output because repeated kernel recomputation was impractically slow. The bandwidth multiplier was fixed at 1, the conventional median-distance value, and alpha was restricted to 0.75, 0.90 and 1.00; C remained 0.1, 1 and 10. This computational amendment was made without observing enhanced-model performance.

A second feasibility amendment was made before output: repeated evaluation is restricted to the primary mild endpoint for the proposed method and posture SVM. ExtraTrees, k-nearest neighbours, Gaussian naive Bayes, bidirectional LSTM and temporal CNN are evaluated on the complete seed-42 cow-grouped folds. This keeps the computational budget focused on early mild recognition; the previously completed secondary endpoint remains available as a consistency analysis.
