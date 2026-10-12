# Behavioral biometrics: keystroke and mouse dynamics on a payment form

Can we tell whether the person typing card details is the card's owner, just from *how* they type and move the mouse?

This project explores a public keystroke and mouse dynamics dataset (88 users, 1,760 form-filling attempts), engineers behavioral features from raw input events and builds per-user models that separate genuine users from impostors.

> **Status:** work in progress. EDA and feature engineering are done; evaluation with biometric metrics (ROC, DET, EER) is in progress in `notebooks/models.ipynb`.

## Why it matters

Banks and payment providers use behavioral biometrics as a passive risk signal. If the behavior on a payment form doesn't match the card owner's usual behavior, the transaction can be stepped up to stronger authentication, while genuine customers pass without extra friction.

## Data

**KMT dataset** (keystroke, mouse and touchscreen dynamics), Nnamoko et al. (2022):
- 88 participants typed fictitious card details (name, card number, CVC, expiry date) into a GUI app resembling an online payment form. The app was built with the Kivy library and logged every key and mouse event with timestamps.
- Each card has **10 genuine attempts**, typed by its assigned user, and **10 impostor attempts**, typed by 10 other participants once each: 1,760 attempts in total.
- The raw data has about 173k key events (key, press/release, text field, time) and 707k mouse events (event type, coordinates, time, movement ID).

The data isn't included in this repository. It's available on Mendeley Data under the CC BY 4.0 license (see [Getting started](#getting-started)).

## Approach

1. **Exploratory data analysis** ([`notebooks/eda.ipynb`](notebooks/eda.ipynb)): data structure, data quality and potential discriminating habits. Main findings:
   - Not every key press has a matching release: a held key is logged as a series of `pressed` events (operating-system auto-repeat).
   - The `tab` key-down is almost never logged, only its release.
   - Each attempt's mouse events end with an empty record holding only an undocumented `false_enters` field.
   - Clicks have no `Movement ID`, and click coordinates carry floating-point noise.
2. **Feature engineering** ([`notebooks/features.ipynb`](notebooks/features.ipynb), [`src/behavioral_ml/features.py`](src/behavioral_ml/features.py)): raw key events are first converted into one row per physical key press, merging auto-repeats. Then 30 features are computed per attempt:
   - **keyboard timing:** dwell time (how long a key is held) and flight time (release to next press), as median and IQR;
   - **mouse movement:** path length, duration, straightness, and velocity, acceleration and jerk statistics per movement, aggregated per attempt;
   - **clicks and scrolling:** click dwell time, number of clicks, scroll bursts;
   - **habits:** counts of `tab`, numpad, arrow, correction (`backspace`, `delete`), `capslock`, `enter` and `spacebar` presses.
3. **Models** ([`notebooks/features.ipynb`](notebooks/features.ipynb), [`notebooks/models.ipynb`](notebooks/models.ipynb)):
   - one classifier per user (genuine vs impostor), as in the paper: decision tree, logistic regression, XGBoost and SVC;
   - biometric evaluation: ROC and DET curves, and the equal error rate (EER);
   - a one-class template detector enrolled on genuine attempts only, which is the realistic setting: at enrollment a bank has only the customer's own data.

## Results so far

| Setup | Features | Evaluation | Result |
|---|---|---|---|
| Paper baseline: decision tree per user | 3 (average dwell, flight, mouse trajectory) | stratified 80/20 split | accuracy 75.85% |
| Logistic regression per user | 30 | stratified 5-fold CV | accuracy 0.855 ± 0.095 |
| SVC (RBF) per user | 30 | stratified 5-fold CV | accuracy 0.853 ± 0.083 |
| Logistic regression per user | 4 keyboard timing features | 5-fold CV; out-of-fold scores pooled over users, one global threshold | ROC AUC 0.851, **EER 0.218** |
| One-class template (median of standardized features, Manhattan distance) | 4 keyboard timing features | enroll on genuine attempts 1–5; test on genuine 6–10 and the 10 impostor attempts | mean per-user ROC AUC 0.921 (median 0.97) |

Accuracies are means over the 88 users (± std across users). The accuracy rows aren't directly comparable with the paper (different split, more features). A pooled EER with one global threshold is a stricter measure than per-user AUC, so the last two rows aren't comparable with each other either.

## Insights

- **Raw logs need cleaning before timing features make sense.** The first repeated press of a held key comes after about 0.25 s or about 0.5 s, then repeats follow every ~28 ms. That pattern is auto-repeat, not real presses. Of 3,188 `tab` presses, 3,169 have no logged key-down.
- **Habits differ between attempts.** Only 901 of 1,760 attempts use `tab` to move between fields at all, which makes navigation style a candidate feature.
- **Accuracy is a poor metric here.** With 4–5 test attempts per user, one error changes a user's accuracy by 20–25 percentage points. This is why the evaluation moves to ROC curves and EER.
- **Genuine-only enrollment already works for most users.** The one-class template reaches a median per-user AUC of 0.98, but the worst user is at 0.22, so the error is very uneven across users.

## Limitations

- **One session per user.** All genuine attempts come from a single sitting, so the results are within-session and likely optimistic; behavior drifts across days.
- **Practice asymmetry.** The genuine user types the same card 10 times, while each impostor types it once, so models may partly learn familiarity rather than identity.
- **Lab setting.** Fictitious card details, one GUI app, 88 participants and balanced classes, unlike real-world fraud.
- **Unrealistic training data.** The per-user binary classifiers are trained on impostor attempts on the same card, which a real system wouldn't have at enrollment.

## Next steps

- Evaluate all models under one protocol with biometric metrics: EER and false rejection rate at a fixed false acceptance rate.
- Compare genuine-only detectors with a global verifier trained on other users.
- Add per-field features from the `Input Box` column (time per field, `tab` vs mouse navigation).
- Move the notebook feature code into the package and cover it with unit tests.
- Track experiments with MLflow.

## Repository structure

```
├── notebooks/
│   ├── eda.ipynb            # exploratory data analysis of the raw events
│   ├── features.ipynb       # feature engineering log and baseline models
│   └── models.ipynb         # biometric evaluation (ROC, DET, EER), one-class detector
├── src/behavioral_ml/
│   ├── etl.py               # load_data: raw JSON files -> event-level DataFrames
│   └── features.py          # mouse kinematics, click/scroll and keystroke features
├── scripts/
│   └── download_dataset.py  # downloads the raw dataset into data/raw_kmt_dataset/
├── tests/                   # unit tests (planned)
├── data/raw_kmt_dataset/    # raw JSON files (not in the repository)
├── pyproject.toml
└── uv.lock
```

## Getting started

Requirements: Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Download the dataset by running this from the repository root:

```bash
uv run python scripts/download_dataset.py
```

The script downloads the [Behaviour Biometrics Dataset](https://data.mendeley.com/datasets/fnf8b85kr6/1) archive from Mendeley Data, verifies its SHA-256 checksum, and keeps only the raw JSON files (`raw_kmt_user_0001.json` … `raw_kmt_user_0088.json`) in `data/raw_kmt_dataset/`.

Then open the notebooks with the project's virtual environment as the kernel (e.g. in VS Code), or start Jupyter with:

```bash
uv run --with jupyter jupyter lab
```

Run the notebooks in order: `eda.ipynb` → `features.ipynb` → `models.ipynb`.

## Use of AI tools

I used AI assistants as reviewers: Claude (Anthropic) and ChatGPT (OpenAI) for review and suggestions on the evaluation protocol and the code. Claude also proofread the notebook comments and drafted this README. The code and the analysis are my own.

## References

- N. Nnamoko, J. Barrowclough, M. Liptrott, I. Korkontzelos (2022). *A behaviour biometrics dataset for user identification and authentication.* Data in Brief 45, 108728. [doi:10.1016/j.dib.2022.108728](https://doi.org/10.1016/j.dib.2022.108728)
- N. Nnamoko, J. Barrowclough, M. Liptrott, I. Korkontzelos (2022). *Behaviour Biometrics Dataset*, Mendeley Data, v1. [doi:10.17632/fnf8b85kr6.1](https://doi.org/10.17632/fnf8b85kr6.1). License: CC BY 4.0.
