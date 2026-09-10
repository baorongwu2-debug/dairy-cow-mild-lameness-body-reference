# Recorded analysis evidence

`pose_results`: 15,330 out-of-fold predictions, six classifiers/representations, two endpoints, five split seeds; 300 outer splits and 900 inner splits.

`enhanced_results`: block-aware kernel fusion and posture SVM across five seeds; Extra Trees, kNN and Gaussian NB at seed 42; strict same-farm session holdout. Single-seed split standard deviations were recorded as NaN in the original summary, meaning undefined, not zero. For interoperable JSON, undefined NaN values are serialized as null and JSON whitespace is compacted. Numerical finite results are unchanged; original JSON file hashes are recorded under provenance.

`sequence_results`: compact BiLSTM and temporal CNN, seed 42, both endpoints.

`fusion_results`: exploratory validation-gated posture/CNN model, seed 42, mild endpoint.

Base, enhanced and sequence split indices refer to the complete 272-row manifest. Fusion split indices refer to the 239-row mild subset. Historical verification files report checks at analysis time; the current independent verifier is `scripts/verify_results.py`. Historical environment hashes refer to original local scripts, before packaging. The packaged scripts differ only as documented under provenance.

Decision scores are thresholded at zero and are not calibrated clinical probabilities. Animal IDs are the pseudonymous identifiers already released by the source study.
