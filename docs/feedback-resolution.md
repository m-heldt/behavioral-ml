# Feedback resolution

This record addresses the applicable correctness and reproducibility findings
in the local feedback history, including its October 13 model-notebook review.
Historical numbers and deadlines are not treated as current requirements.

## Corrected

| Finding | Resolution |
|---|---|
| EER loses crossing points through ROC thinning | All score-based ROC/EER paths retain intermediate points. A hand-checkable regression case changes the erroneous 0.1 estimate to 0.2. The discrete crossing convention is documented. |
| Global logistic regression receives signed differences | Comparison vectors default to absolute enrollment-template differences; signed differences remain an explicit experimental option. |
| Card-owner grouping can leak the impostor's own records | Global scoring requires typist candidates. All plausible preceding-session candidates must be on the same fold side as the owner. Unknown/cross-fold attempts are excluded, with coverage reported. Attribution remains conditional, not ground truth. |
| Duplicate SVC runs, ineffective centered P1, repeated result rows | The model notebook runs each named experiment once through shared helpers. Metrics group indexed scores into exactly one row per user. |
| Isolation Forest differs in inputs and has no seed | Default comparisons use the same 18-key features and preprocessing; random seed is explicit. |
| Training helpers assume 88 users, use inconsistent returns, or require probabilities | Shared APIs derive users from data, preserve indexed score Series, clone estimators, and accept margins, sample scores, or genuine probabilities. Enrollment/probe sizes are validated. |
| Feature names contain a lambda's memory address | Aggregation names are stable; IQR columns have an explicit suffix. |
| Plot functions show immediately and cannot share axes | Plot functions return axes and accept labels; the notebook controls display. |
| Mouse timestamp deduplication removes distinct events | All nonempty raw mouse events are retained. Zero-time steps contribute geometry, with undefined rates left missing. Negative elapsed times fail explicitly. |
| Release-only events contaminate keyboard latencies | Their synthetic ordering timestamp is excluded from observed-press timing calculations, on both sides of a next-key interval. Counts still retain the physical action. |
| Habit counts include raw press/release and auto-repeat records | The package builder counts transformed physical actions. |
| Missing click dwell becomes measured zero | Only missing count-like burst values are filled with zero; dwell is imputed inside training/enrollment. |
| Useful optional fields are discarded | Keyboard fields are preserved. Undocumented false_enters trailers are retained as attempt metadata, not promoted to model features. |
| Metadata makes grouped feature calculations expensive | Attempt metadata remains on the raw table and is omitted from derived event tables, avoiding repeated pandas deep copies during aggregation. |
| Legacy experiment imports and placeholder CLI are broken | The entry point runs fixed baselines through the same implementation as the model notebook, and saves predictions, fold IDs, metrics, settings, and provenance. |
| Notebook kernel is a runtime dependency | ipykernel is now a development dependency; the lockfile metadata is updated and uv's offline lock check passes, with all third-party versions, hashes, and platform markers unchanged. |
| Downloader has import-time side effects and cannot be safely rerun | Downloading is explicit, checksum-verified, staged in a temporary directory, and never overwrites existing raw data. |
| Notebook outputs, comparisons, and README claims are stale | The exploration sequence is retained, corrected package logic is reused, static obsolete metric claims are removed, and fresh baseline tables replace the old README numbers. |
| Low-FAR and per-user EER are presented too strongly | Reports distinguish pooled/per-user statistics, include probe counts and missing-class users, and label evaluation-selected operating points as descriptive. Separate threshold-selection/evaluation helpers support future calibration. |

## Already resolved before this change

- Git history, a substantive README, the model notebook, a package layout,
  a first fixture test, JSON filtering, corrected flight-time and straightness
  formulas, and the corrected correction-key aggregation already existed.
- Mouse movement features were already joined into the updated exploration
  table; that earlier accidental omission was no longer present.
- The old grid-search leakage example, very low XGBoost learning rate, and
  unused CatBoost/LightGBM/NGBoost dependencies were no longer present.
- Raw data, private feedback, and the job posting were already untracked.
  The unrelated dataset directory that previously broke pytest was removed.

## Scope and remaining limits

The feedback's optional SQL/API work, MLflow orchestration, extra feature
families, new datasets, sequence models, and confidence intervals are extensions,
not necessary repairs. Existing third-party dependency pins are preserved.

No raw file was edited or redownloaded. Generated manifests record the files
actually used; this does not independently revalidate every file against the
original archive. The dataset's single-session design and unrecorded attacker
identities require better collection to resolve fully.

The baseline run yields 1,760 attempts, 18 key features, 21 mouse features, and
39 combined features. It preserves 706,984 nonempty mouse events. The strict
common-probe comparison contains 440 genuine and 112 impostor probes; 61 users
have both classes. Scores and metrics are saved under the ignored artifacts
directory, with fixed settings rather than test-selected hyperparameters.

## Validation

- Synthetic regression tests cover EER crossings/ties, enrollment chronology,
  preprocessing isolation, classifier score methods, conservative typist
  separation, physical key actions, movement boundaries, and download safety.
- Every code cell in each of the three notebooks is executed in order, with
  each notebook in a fresh IPython process and newly captured outputs, using
  the existing read-only KMT data.
- The command-line baseline and notebook results are compared on indexed scores.
- Whitespace, notebook execution order, source imports, and public file
  references are checked before delivery.

Verified on October 28, 2026: 27 tests pass; all 88 code cells across the three
notebooks have consecutive execution counts and no error or warning outputs.
The command-line and notebook runs agree exactly on both metric tables and all
8,144 indexed prediction rows. uv's offline lock check passes, with unchanged
third-party versions, hashes, and markers. The three model-report plots were
also visually inspected.
