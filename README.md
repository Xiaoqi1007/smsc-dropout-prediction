# SMSC dropout analysis

Code accompanying *Interpretable multimodal AI for early detection of dropout from child maltreatment interventions*.

Version 1.8.0. This repository includes preprocessing, participant split generation, model fitting and selection, evaluation, Figures 2–4, and greedy KL excerpt selection. Figure 1 is a separately prepared schematic. Study data, clinical field dictionaries, task protocol tables, manual speaker corrections, fitted models, and analysis outputs are not distributed.

## Installation

The downstream analyses use Python 3.10.20 and the versions in `requirements.txt`:

```sh
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests -v
```

For transcription, emotion inference, semantic task matching, text embeddings, or the subset predictor, install the additional packages in `requirements-preprocessing.txt` in a separate environment. These model packages have their own platform requirements. FFmpeg must be available for audio extraction. Model interfaces follow [WhisperX](https://github.com/m-bain/whisperX), [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-8B), and [MERaLiON-SER](https://huggingface.co/MERaLiON/MERaLiON-SER-v1). The optional inference environment is not a frozen environment for the reported analysis. Model weights must be obtained from their respective providers; gated models use a locally supplied `HF_TOKEN` environment variable.

## Preprocessing

`scripts/preprocess.py` exposes the following commands. Run a command with `--help` for its arguments. All paths refer to local authorized inputs.

| Command | Input and operation |
| --- | --- |
| `baseline` | Raw questionnaire table and a private JSON scoring specification: aliases, event filters, sum/mean scoring, reversed items, missing-item rules, categorical recoding, and age calculation. |
| `extract-sessions` | Repeated source-form fields and a private JSON map to participant/session records. |
| `sessions` | Session metadata and a private field map; calculate elapsed days and note word counts. |
| `outcomes` | Attendance records plus participant outcomes and last attended session; add the pre-assessment row and derive the final-session dropout indicator. |
| `transcribe` | Audio/video to word-aligned WhisperX transcription and speaker diarization. |
| `roles` | Anonymous speaker turns to clinician/participant roles using bilingual lexical evidence; optional local manual corrections. |
| `tasks` | Role-tagged turns and a local task protocol to dialogue chunks and task annotations. |
| `speech` | Task-tagged chunks to role-specific and task-specific word counts and ratios. |
| `emotion-segments` | Recording manifest and MERaLiON outputs to keyed emotion segments. |
| `emotion` | Segment labels to current counts/proportions and pooled proportions from strictly earlier sessions. |
| `text` | Ordered baseline responses, session notes, and transcript excerpts to session text. |
| `merge` | Left-join modality features to observed participant/session records. |
| `embed` | Session text to normalized Qwen3-Embedding-8B embeddings and compatible caches. |
| `splits` | Participant-level training/validation/test assignments using `configs/splits.json`. |

The questionnaire scoring JSON uses `id_column`, optional `event_column`/`event_values` and `aliases`, and a `features` list. Each feature supplies `name`, `operation`, and `columns`; supported operations are `copy`, `map`, `age_years`, `sum`, and `mean`. Sum/mean items may supply `range`, `reverse`, and `minimum_items`. A repeated-session specification supplies `id_column` and a `sessions` list, each with `session` and a source-to-output `fields` map. A session metadata map maps local column names to `id`, `session`, optional `session_date` and `session0_date`, and other desired output names. Keep these specifications outside version control.

Task protocol CSV/XLSX columns are `arm`, `module`, `task_order`, `task`, `task_label`, `task_category`, `task_count_in_module`, `cue_boundary`, and `cue_words`. The task code retains clinician–participant dialogue pairs during chunking, uses ordered segmentation for P-STAIR, and flexible task/safety matching for SC. Private manual role corrections map anonymous speaker labels to `Clinician` or `Participant`.

Examples of the transcription-to-task sequence:

```sh
python scripts/preprocess.py transcribe --input data/recording.wav --output results/turns.json
python scripts/preprocess.py roles --input results/turns.json --output results/roles.json
python scripts/preprocess.py tasks --input results/roles.json --config data/task_protocol.csv --participant local_key --session 1 --arm pstair --output results/tasks.csv
python scripts/preprocess.py speech --input results/tasks.csv --output results/speech.csv
```

For audio emotion inference, run `python scripts/audio_emotion.py --help`. `MERaLiON/MERaLiON-SER-v1` processes consecutive 30-second segments of the recording's first audio stream at 16 kHz. A local recording manifest for `emotion-segments` has `id`, `session`, and `path`, where `path` points to the corresponding inference JSON.

Session text inputs for `embed` have `id`, `session`, and `text`. The default encoder is `Qwen/Qwen3-Embedding-8B`, with 4,096-dimensional normalized embeddings and a maximum input length of 32,768 tokens. Text source rows for `text` have `id`, `session`, `source_order`, `excerpt_order`, and `text`; the local investigator specifies source/excerpt order before merging.

Split generation stratifies jointly on final dropout outcome and attendance duration (sessions 0, 1–9, and 10–23). It uses a 20% test allocation, followed by a 20% validation allocation within the remaining participants, with validation random-state offset 100003. Every session of a participant shares its assignment. An optional reference comparison runs after generation and does not determine assignments.

## Downstream inputs

Place authorized processed inputs under `data/` or edit the paths in `configs/analysis.json`:

| File | Required contents |
| --- | --- |
| `all_modalities_id_session_wide.csv` | One row per participant/session; `id`, `session`, `ever_dropout`, `next_session_dropout`, and structured predictors. |
| `participant_splits.csv` | `doc_id`, `seed`, `split` (`train`, `validation`, or `test`). |
| `preassessment_tabular_by_id.csv` | Processed pre-assessment predictors keyed by `record_id`. |
| `embeddings/text_input_full_text/cache/text/shared.csv` | Session text with participant/session keys. |
| `embeddings/text_input_full_text/cache/embeddings/shared/` | `embeddings.npy`, `embeddings_done.npy`, and `embeddings_metadata.json`. |

Identifiers are opaque strings. No filename convention or study identifier format is required. Feature groups are defined by prefixes in `src/smsc/_core/defaults.json`; source field names and language-specific aliases belong in private preprocessing specifications. The loader retains rows with both outcome fields present and applies the configured session window.

## Model fitting and evaluation

```sh
python scripts/reproduce.py --config configs/analysis.json
```

The model combines pre-assessment XGBoost, session-metadata XGBoost, emotion/speech logistic regression, and a text neural network. At session t, the text input is the cumulative mean of session embeddings from session 0 through t. The four raw probabilities are averaged equally and calibrated using validation records. All component models and the clinician-rating baseline use training weights proportional to inverse participant row count multiplied by inverse class frequency. Neural-network validation loss uses participant weights; calibration and test metrics weight session rows equally.

Preprocessing is fitted on training records. Validation AUROC selects XGBoost parameters, and validation binary cross-entropy controls neural-network early stopping. Training and validation remain separate for final fitting. Emotion/speech predictors are standardized once before logistic regression. AUPRC uses trapezoidal integration. Means and sample SDs summarize repeated, overlapping participant splits.

The supplied configuration excludes the duplicate session field, participant total words, combined word totals, total emotion-segment count, and recording duration in minutes. Duration in seconds, clinician word count, task-specific participant counts, and all seven current and prior emotion proportions, including Neutral, remain.

For training-set model comparison, run each seed listed in the configuration, then summarize:

```sh
python scripts/model_selection.py --config configs/analysis.json --project . --output results/model_selection --seed 0
python scripts/model_selection.py --config configs/analysis.json --output results/model_selection --summarize
```

The comparison uses participant-grouped three-fold cross-validation inside each outer training partition. XGBoost, logistic regression, and neural networks are compared for each of four sources, and all 81 source-family combinations are ranked by equal-weight fusion AUROC. Original validation/test records do not enter these folds. The summary requires completed results for every configured seed.

## Explanations and figures

```sh
python scripts/reproduce.py --config configs/analysis.json --explanations --cases data/private_cases.json
```

The private case JSON supplies `risk_seed`, `dropout_id`, `completer_id`, `case_seed`, `case_session`, and a two-element `case_ids` list. Optional `text_summaries` supply manually reviewed context for Figure 4. Cases must belong to the specified test partition.

Set `SMSC_FEATURE_LABELS` to a private JSON feature-to-label map for manuscript terminology. Without it, plots derive readable labels from column names. The clinical variable dictionary is not included in this repository. PNG exports use 300 dpi; PDF/SVG provide vector alternatives.

Permutation importance permutes features within treatment arm and session for 30 repeats. Figure 4 colors use the mean contrast between higher and lower feature values when the contrast is estimable in at least eight splits and its absolute mean exceeds 0.01 percentage points. This rule does not require eight splits to share the same sign. Local SHAP uses a nominal permutation-pair budget of 128; the actual count depends on the number of varying features and is recorded with each case. Calibration-panel error bars are limited to the probability interval by the smaller of the SD and the distances to 0 and 1.

## Excerpt selection

This is a separate subset predictor, not the main cumulative-history text model. Excerpt rows contain `id`, `session`, `excerpt_order`, `text`, and `ever_dropout`. Random-subset training draws subset size and membership independently of content and outcome. Inputs concatenate an embedding of selected excerpts in their original order, normalized selected-subset size, and normalized total session excerpt count. The predictor has one 256-unit GELU layer with dropout 0.3 and a two-class softmax output. The defaults use 100 random states per session, subsets of up to five excerpts, Adam with learning rate 0.00001, batches of 4,096 states, at most 80 epochs, and validation-loss early stopping.

```sh
python scripts/select_excerpts.py states --input data/excerpts.csv --splits data/participant_splits.csv --seed 0 --output results/states.npz
python scripts/select_excerpts.py train --input results/states.npz --seed 0 --output results/subset_model
python scripts/select_excerpts.py select --input data/excerpts.csv --splits data/participant_splits.csv --seed 0 --predictor results/subset_model --output results/selected.csv
python scripts/select_excerpts.py review --input results/selected.csv --output results/review.csv
```

Greedy selection starts at the training-set dropout prior. At each step it selects the remaining excerpt maximizing Bernoulli KL divergence from the current prefix prediction, restoring source order before evaluating each candidate subset. The default selection budget is five excerpts. The review sample uses sessions 1–9, up to five excerpts in each outcome-consistent direction per split, and at most one excerpt per participant and split. No annotated excerpts or precomputed selection results are supplied.

## Cluster execution

```sh
bash hpc/run_analysis.sh python
# or, from the repository root:
sbatch hpc/run_analysis.sbatch
```

`SMSC_CONFIG` selects another analysis configuration, and `SMSC_CASES` supplies an optional local case file. The wrappers have no installation-specific account, host, or email settings.

## Data access, citation, and license

Keep clinical records, recording/transcript files, source dictionaries, private configurations, predictions, model checkpoints, logs, and generated reports outside the public repository. The supplied `.gitignore` excludes these file types and directories. Generated feature snapshots and hashed identifiers are not anonymized datasets. Data access is subject to the study's consent and data-use conditions; contact the manuscript's corresponding author.

Cite the accompanying manuscript using `CITATION.cff`. All rights reserved; see `LICENSE` for terms.
