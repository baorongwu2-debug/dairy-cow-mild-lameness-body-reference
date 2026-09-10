# Packaging changes

The numerical fitting code in pose_study.py, enhanced_study.py, sequence_baselines.py, temporal_fusion.py, finalize_pose.py and enhanced_finalize.py is retained from the original experiments. Original script hashes are supplied. Line endings may differ.

Changes: run_study.py now contains only portable I/O helpers; it no longer imports the unrelated video study or a machine-specific runtime. ROOT can be set with LAMENESS_WORKDIR. pose_study.py records a null archive hash if the original ZIP is absent. No model, feature, split seed, hyperparameter or saved prediction was changed. The runner copies protocols/data to an isolated output directory and refuses to reuse an existing completed stage.

The stored historical reports are evidence from earlier runs. Current validation is independently recorded in package_verification.json after running scripts/verify_results.py. Model retraining is a separate operation.

Result JSON is compacted, and nonstandard NaN values (undefined single-seed standard deviations) are converted to null. Finite numerical values and all prediction CSV files are unchanged. Original JSON hashes are recorded separately.
