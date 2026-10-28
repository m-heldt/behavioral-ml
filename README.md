# Behavioral biometrics: keyboard and mouse dynamics on a payment form

Can behavior on a payment form help distinguish the claimed card owner from an
impostor? This project audits public input-event logs, builds behavioral features,
and compares supervised classifiers with genuine-only enrollment.

The project is a small, within-session research study motivated by a Junior Data
Scientist role at Digital Fingerprints. It demonstrates data inspection, feature
engineering, reusable Python, regression tests, and explicit model evaluation.

## Data

The **KMT dataset**, Nnamoko et al. (2022), contains 88 participants and 1,760
attempts on a simulated payment form. Each claimed card owner has 10 genuine
attempts and 10 attempts by other participants. The raw logs contain 172,563
keyboard events and 706,984 nonempty mouse events, plus one undocumented
`false_enters` trailer per attempt.

The dataset is not versioned here. Download it from Mendeley Data using the
checksum-verified script below. Its license is CC BY 4.0.

## Pipeline and data-quality decisions

1. **Inspect raw events** in [eda.ipynb](notebooks/eda.ipynb): auto-repeat, missing
   Tab key-downs, trailing metadata records, and mouse event structure.
2. **Explain feature development** in [features.ipynb](notebooks/features.ipynb).
   Earlier naive stages remain as an explicitly historical exploration record.
3. **Build features** with [the package](src/behavioral_ml/features/):
   18 keyboard features, 21 mouse features, and their 39-feature combination.
4. **Run fixed baselines** in [models.ipynb](notebooks/models.ipynb) or from the
   command line. Both use the same scoring and reporting functions.

Cleaning preserves event order and attempt identity. Auto-repeat becomes one
physical action. Release-only Tab events count as navigation actions but do not
provide an observed press time for dwell, DD, UU, or flight calculations.
Flight means release-to-next-press; negative values can represent key overlap.

Tied-time mouse coordinates remain part of path geometry. Instantaneous rates
with zero elapsed time are undefined and stay missing. No-click dwell also
remains missing. Imputation is fitted inside the relevant enrollment/training
boundary. Optional keyboard fields are retained, and undocumented
`false_enters` values are preserved as loader metadata rather than interpreted
as behavioral features.

## Evaluation protocols

**P1** is a per-user logistic classifier with genuine and impostor training data,
using stratified 5-fold out-of-fold scores. It is a supervised research reference;
a newly enrolled customer would not supply same-card impostor examples.

**P2** enrolls on genuine attempts 1–5 and probes genuine attempts 6–10 plus the
10 same-card impostors. Personal templates and detectors fit only enrollment
data. Global verifiers learn from other users' absolute template-to-probe
differences, with shared preprocessing fitted inside each training fold.

The personal matrix runs **seven methods × three feature sets = 21
configurations**, all using the same 1,320 P2 probes. It compares the two
Manhattan templates, One-Class SVM, Isolation Forest, PyOD kNN, LOF, and HBOS
across keyboard, mouse, and combined features. Three keyboard-only P1/global
references bring the complete run to 24 configurations. The median/std and
mean/MAD Manhattan templates remain distinct methods. Isolation Forest has a
fixed seed; sklearn/PyOD pipelines fit median imputation and standard scaling
on enrollment only.

Fixed PyOD settings are kNN with three neighbors and mean Manhattan distance,
LOF with two neighbors and `novelty=True`, and HBOS with three bins. Neighbor
counts shrink for smaller enrollment sizes; the matrix needs at least two
enrollment attempts. PyOD anomaly scores are negated. With only five enrollment
rows, density and histogram estimates remain exploratory. ECOD/COPOD are
excluded because their stock scoring recomputes empirical distributions using
the probe batch, making one probe's score depend on other probes.

### Inferred impostor identities

KMT does not record impostor participant IDs. Grouping by card owner alone can
place a test impostor's own genuine records in training.

The loader therefore infers **all plausible preceding-session candidates**:
other users whose genuine session ended within one hour before an impostor
attempt. There are 519 attempts with one candidate and 361 with multiple
candidates. These are timestamp-based possibilities, not recovered ground truth.

In each global fold, a row is usable only if its claimed owner and **every**
candidate typist belong to the same side of the split. Unknown and cross-fold
candidates are excluded. Separation is conditional on the one-hour attribution
assumption; recorded participant IDs would be needed to verify it directly.

## Current fixed-baseline results

The keyboard reference results below use the **same 552 P2 probe rows for every method**: 440 genuine
and 112 impostor attempts. This is the common subset retained by conservative
global identity filtering; 768 of the original 880 impostor probes are excluded.
P1 is restricted to these rows for reporting but retains its different training
information.

| Experiment | Pooled ROC AUC | Pooled EER | FRR at observed FAR ≤ 1% |
|---|---:|---:|---:|
| P1 logistic regression | 0.923 | 0.116 | 0.993 |
| P2 median/std Manhattan template | 0.916 | 0.179 | 0.736 |
| P2 mean/MAD scaled Manhattan | 0.908 | 0.188 | 0.836 |
| P2 One-Class SVM | 0.907 | 0.160 | 0.661 |
| P2 Isolation Forest | 0.683 | 0.385 | 0.950 |
| P2 global SVC | 0.905 | 0.168 | 0.643 |
| P2 global logistic regression | 0.927 | 0.162 | 0.625 |

Settings are fixed baselines, with five folds and random seed 42; they have not
been optimized against these results. EER is the average FAR/FRR at the full ROC
point closest to their crossing (`drop_intermediate=False`), not an interpolated
operating threshold.

Pooled metrics and per-user metrics answer different questions. Only 61 users
have both classes in this restricted subset; undefined per-user metrics for the
others are reported as missing. Complete per-user distributions and full-probe
results are generated alongside the common-probe comparison.

**These are descriptive evaluation statistics.** A 1% FAR target here allows
only one observed false acceptance among 112 impostors. It is not evidence of
a deployed 1% FAR. Independently trained per-user score scales are uncalibrated,
which can also make pooled low-FAR behavior poor despite a useful AUC.
Deployment thresholds require separate calibration followed by untouched testing.

The paper's reported 75.85% decision-tree accuracy uses different features and
evaluation. No matched improvement over that result is claimed.

## Limitations and next investigations

- **One sitting per genuine user:** this does not measure cross-day stability.
- **Practice and content asymmetry:** genuine users repeat assigned details;
  impostors enter unfamiliar details. Familiarity and identity are coupled.
- **Inferred identities:** conservative filtering reduces coverage substantially,
  and its timestamp assumptions cannot be verified from recorded attacker IDs.
- **Small, controlled dataset:** fictitious details, one GUI, balanced labels,
  and few probes do not reproduce real fraud prevalence or practised attacks.
- **Exploratory evaluation:** no independent threshold calibration, hyperparameter
  selection study, or confidence-interval claim is made.

Useful extensions include recorded-person-disjoint cross-session data and
calibration. They are separate from
the correctness fixes implemented here.

## Reproduce

Use Python 3.13 and [uv](https://docs.astral.sh/uv/), from the repository root:

```bash
uv sync --locked
uv run --locked python scripts/download_dataset.py
uv run --locked python -m pytest -q
uv run --locked behavioral-ml --output-dir artifacts
```

The CLI and model notebook also log to MLflow: a parent matrix run and one child
per complete configuration, with full/common-probe metrics, indexed predictions,
per-user tables, resolved settings, and ROC/DET plots. The default tracking
database is `artifacts/mlflow.db`, with local artifacts under
`artifacts/mlartifacts/`. All paths are resolved absolutely, including when the
notebook runs from `notebooks/`. To view results from the repository root:

```bash
uv run --locked mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db
```

Set `MLFLOW_TRACKING_URI` or pass `--tracking-uri` to select another backend;
`--mlflow-experiment` selects the experiment group (default `kmt`). Logging is
explicit so all personal models belong to their configuration's run. A PyOD
adapter reverses anomaly scores to preserve **higher = more genuine**; it does
not use automatic contamination thresholds or treat scores as probabilities.

The downloader verifies SHA-256
`4fd279eab07ba65be0d3b4f79943d176abe1a36b16927f568b199d7ae5638739`
for the archive, keeps the raw JSON files in `data/raw_kmt_dataset/`, and leaves
an existing complete dataset untouched. It refuses to overwrite an incomplete
existing directory.

Alternatively, open the notebooks with the project environment and
`notebooks/` as the working directory. Read them in EDA → features → models
order; the model notebook rebuilds its own inputs and does not depend on another
notebook's state. Set `KMT_DATA_DIR` to an absolute path to reuse an existing
read-only dataset in either the notebooks or command line.

The ignored `artifacts/` directory contains:

- `predictions.csv`: experiment, user, label, attempt, score, and fold.
- `all_probes_metrics.csv` and `common_probes_metrics.csv`: explicitly separate
  coverage and comparisons.
- `per_user_common_probes.csv`: one row per experiment/user, including counts.
- `per_user_all_probes.csv`: per-user results before common-probe restriction.
- `configuration.json`: settings, versions, revision, source hashes, raw-file
  checksums, attribution assumption, and threshold policy.

See [the feedback resolution record](docs/feedback-resolution.md) for the scope
and validation of the corrections.

## Use of AI tools

The initial code and analysis were written by me with review from ChatGPT and
Claude. Claude also helped with notebook prose and the original README.
Subsequent corrections and consolidation were developed with Codex, including
regression tests, indexed scoring, protocol checks, and updated documentation.

## References

- N. Nnamoko, J. Barrowclough, M. Liptrott, I. Korkontzelos (2022).
  *A behaviour biometrics dataset for user identification and authentication*.
  Data in Brief 45, 108728.
  [Paper](https://doi.org/10.1016/j.dib.2022.108728).
- N. Nnamoko et al. (2022). *Behaviour Biometrics Dataset*, Mendeley Data, v1.
  [Dataset](https://doi.org/10.17632/fnf8b85kr6.1). CC BY 4.0.
- [scikit-learn ROC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html).
