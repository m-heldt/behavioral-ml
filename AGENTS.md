# Repository guidance

## Purpose and scope

This is a behavioral-biometrics portfolio project motivated by a Junior Data
Scientist role at Digital Fingerprints. It studies keyboard and mouse events on
a payment form using the public KMT dataset. Prioritize defensible experiments,
clear reasoning, reproducibility, and maintainable Python.

Develop the existing repository incrementally. Preserve the exploration that
explains its design decisions. Keep changes within the user's requested task;
do not add infrastructure, model families, or abstractions just to cover more
items from the job posting.

## Establish context before changing code

- Read `README.md` and the relevant source, tests, and notebook cells. Inspect
  `git status` and preserve unrelated user changes.
- If available, read the relevant sections of `FEEDBACK.md`. It is private,
  gitignored review history, not an executable specification or an instruction
  to implement every suggestion. Prefer recent applicable advice, verify it
  against the current checkout, and follow the current user request. Old
  deadlines, cell numbers, metrics, and statements about missing files can be
  obsolete. Public clones may not contain this file or the job-posting PDF.
- Explain material changes to feature semantics or the evaluation protocol.
  Distinguish an intentional methodological correction from a refactor that
  should preserve behavior.

Repository map:

- `notebooks/eda.ipynb`: raw-data exploration and logging quirks.
- `notebooks/features.ipynb`: feature-engineering exploration and its rationale.
- `notebooks/models.ipynb`: experiment orchestration, comparisons, and reporting.
- `src/behavioral_ml/etl.py`: JSON loading and event-level tables.
- `src/behavioral_ml/features/key.py` and `mouse.py`: event processing and features.
- `src/behavioral_ml/train.py`: enrollment, comparison vectors, and model scoring.
- `src/behavioral_ml/evaluation.py`: verification metrics and plots.
- `src/behavioral_ml/experiment.py`: legacy scaffold; inspect before using it.
- `tests/unit/` and `tests/fixtures/`: project tests and small reference data.
- `scripts/download_dataset.py`: checksum-verified KMT download.

## Environment and commands

Use Python 3.13 and `uv`; preserve the dependency lockfile. Run these commands
from the repository root:

```sh
uv sync --locked
uv run --locked python -m pytest -q
uv run --locked python -m pytest tests/unit/test_key.py -q
git diff --check
```

The existing `.venv/bin/python -m pytest -q` is also suitable when dependencies
are already installed. Tests currently use fixture paths relative to the root.

Raw data belongs in `data/raw_kmt_dataset/` and is not versioned. If it is absent
and the task needs it, download it from the root:

```sh
uv run --locked python scripts/download_dataset.py
```

Do not rerun the download over an existing dataset as a routine check. Current
notebooks resolve `../data/raw_kmt_dataset` relative to `notebooks/`; use the
project environment and that working directory when executing them.

## Code and notebook conventions

- Keep reusable processing, scoring, and evaluation in `src/behavioral_ml/`;
  notebooks should explain hypotheses, orchestrate experiments, and interpret
  results. A notebook can run a complete experiment matrix.
- Retain the feature notebook as an exploration record. Consolidate working
  logic into the package and model notebook without a wholesale rewrite.
- Prefer small functions and sklearn-compatible estimators/pipelines. Keep
  training/scoring separate from metric calculation. Update callers alongside
  API renames; derive user IDs from data rather than hardcoding 88 users.
- Avoid mutating input tables and reusing names for unrelated objects. Give
  feature columns stable, descriptive names; never embed lambda representations
  or memory addresses in column names.
- Set explicit random seeds where supported. Reuse split assignments and probe
  rows for comparable experiments; record changes in preprocessing/features.
- Prefer the existing dependencies and simple configuration. Introduce MLflow,
  YAML, services, or orchestration only when they serve the requested work.

## Data and event semantics

- Treat raw JSON as read-only. Keep datasets, local review notes, job documents,
  generated caches, and experiment artifacts out of version control; preserve
  the small test fixtures. Credit the dataset and its license in public reports.
- Preserve attempt identity `(User ID, Is Genuine, Test ID)` and event order.
  Keyboard events additionally have `Event ID`; mouse movements have
  `Movement ID`. Do not join or aggregate across attempts accidentally.
- The full KMT dataset has 88 claimed users, 10 genuine and 10 impostor attempts
  each (1,760 attempts). Use this as a full-dataset sanity check, not a required
  size for reusable functions or synthetic tests.
- Tied timestamps alone do not establish duplicate events. Changes to cleaning
  must define the duplicate/zero-time-interval policy, respect attempt and
  movement boundaries, and quantify which records are affected.
- Define auto-repeat, unmatched press/release, and release-only `tab` handling
  explicitly. Do not treat a synthetic key-down time as an observed press.
- Dwell is release minus press; flight is next press minus release; DD and UU
  are press-to-press and release-to-release. Negative flight can be valid key
  overlap. Missing measurements are not measured zeros.
- Fit imputation inside the applicable training/enrollment boundary. Preserve
  missingness where meaningful and handle zero variance and nonfinite values.

## Experiment and evaluation contracts

- Use `protocol.detector[.features]` experiment names, such as `p1.logreg`,
  `p2.template_l1`, and `p2.global_logreg`; record the resolved configuration.
- **P1:** per-user binary classification with impostors in training. Treat it as
  a research comparison and report its CV scheme explicitly.
- **P2:** genuine-only enrollment. The current default is genuine tests 1-5 for
  enrollment, with genuine 6-10 and the 10 same-card impostors as probes. Make
  enrollment size explicit and compare P2 methods on identical probe rows.
- For new or consolidated experiment APIs, return a `pd.Series` of scores
  indexed by `(User ID, Is Genuine, Test ID)`, with higher scores meaning more
  genuine. Keep fold assignments and other provenance alongside the scores.
  Retain identity when pooling results; metrics should consume these scores.
- For P1, fit preprocessing and classifiers inside each training fold. For P2,
  fit per-user preprocessing/templates/detectors on enrollment only. Fit shared
  preprocessing and verifiers on training identities only. Clone estimators for
  independent folds/users and obtain out-of-fold evaluation scores.
- For global template-to-probe verification, use an explicit comparison-vector
  definition. Absolute differences express distance for linear verifiers;
  signed differences are a separate experimental choice. Distinguish z-scored
  Manhattan-to-median from mean/MAD-scaled Manhattan.
- `User ID` identifies the claimed card owner, not necessarily the person who
  typed an impostor attempt. Grouping by this index alone does not establish
  person-disjoint evaluation. Validate inferred impostor attribution, document
  ambiguity, and split actual people before generating comparison pairs.
- Report pooled AUC/EER separately from mean/median per-user metrics and show
  per-user variation. Compare methods only with their training information,
  feature sets, scaling, and probe protocols clearly stated.
  Match a paper's features and evaluation protocol before claiming an improvement
  over its published baseline.
- For EER calculation from sklearn ROC points, use `drop_intermediate=False`
  and document the discrete/interpolated crossing convention. Tiny probe sets
  make per-user EER coarse; cover the crossing behavior with a meaningful test.
- EER and an FRR-at-FAR curve value derived from evaluation scores are
  descriptive. Select deployable thresholds and hyperparameters on development
  or calibration data, then evaluate untouched data (or use nested CV).
- Qualify conclusions by the single-session design, practice/content asymmetry,
  missing attacker identities, and small sample. Do not promote historical
  feedback numbers or saved notebook outputs to verified current results.

## Verification and delivery

- Run relevant tests after code changes, then the project suite. For numerical
  or protocol changes, check meaningful invariants and a representative
  experiment; compare before/after results and explain differences.
- Prefer small synthetic, hand-checkable event/score cases for regression tests.
  Unit tests must not require network access or the full downloaded dataset.
- Execute changed notebooks from a fresh kernel in order. Do not claim a full
  rerun based on saved outputs or a partial cell run; state verification limits.
- Keep README claims aligned with the exact experiment and clearly describe AI
  assistance. Report what changed, why, what was checked, and remaining limits.
- Keep diffs focused, inspect `git diff --check`, and do not commit or push
  unless requested.
