"""SHAP of calibrated risk using same-session training backgrounds."""

from pathlib import Path
import numpy as np, pandas as pd, shap
from threadpoolctl import threadpool_limits
from .explain_support import FrozenModel, importance_label
from .experiment import write_json

THREADS = 1
SHAP_PAIRS = 128
SHAP_RANDOM_SEED = 20261001


def compute_case_shap(state, case_id, session, output):
    PRIVATE = Path(output)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    predictions = state["predictions"].reset_index(drop=True)
    rows = predictions.index[
        predictions.id.eq(str(case_id)) & predictions.session.eq(session)
    ]
    if len(rows) != 1 or predictions.loc[rows[0], "split"] != "test":
        raise ValueError(
            "Choose exactly one observed test participant-session in this seed."
        )
    case_row = int(rows[0])
    refs = predictions.index[
        predictions.split.eq("train") & predictions.session.eq(session)
    ]
    if len(refs) < 5:
        raise ValueError(
            "At least five same-session training participants are required."
        )
    adapter = FrozenModel(state)
    with threadpool_limits(limits=THREADS):
        probability = adapter.predict_probability(adapter.encoded)
    np.testing.assert_allclose(probability, predictions.probability, atol=1e-06, rtol=0)
    reference = float(probability[refs].mean())
    predicted = float(probability[case_row])
    background = adapter.encoded.loc[refs].to_numpy()
    case = adapter.encoded.loc[[case_row]].to_numpy()
    masker = shap.maskers.Independent(background, max_samples=len(background))
    np.testing.assert_array_equal(masker.data, background)
    varying_features = int((~masker.invariants(case[0])).any(axis=0).sum())
    max_evals_per_run = SHAP_PAIRS // 4 * (2 * len(adapter.columns) + 1)
    pairs_per_run = (
        max_evals_per_run // (2 * varying_features + 1) if varying_features else 0
    )
    (estimates, bases) = ([], [])
    with threadpool_limits(limits=THREADS):
        for repeat in range(4):
            print(f"Individual SHAP: run {repeat + 1}/4", flush=True)
            explainer = shap.PermutationExplainer(
                adapter.predict_probability,
                masker,
                feature_names=adapter.columns,
                link=shap.links.identity,
                seed=SHAP_RANDOM_SEED + repeat,
            )
            result = explainer(
                case, max_evals=max_evals_per_run, batch_size=32, silent=True
            )
            phi = np.asarray(result.values[0], dtype=float)
            base = float(np.asarray(result.base_values).reshape(-1)[0])
            np.testing.assert_allclose(base, reference, atol=1e-06, rtol=0)
            np.testing.assert_allclose(base + phi.sum(), predicted, atol=1e-06, rtol=0)
            estimates.append(phi)
            bases.append(base)
    estimates = np.asarray(estimates)
    phi = estimates.mean(axis=0)
    mc_se = estimates.std(axis=0, ddof=1) / 2
    table = pd.DataFrame(
        dict(
            feature=adapter.columns,
            label=[importance_label(c) for c in adapter.columns],
            contribution_pp=100 * phi,
            mc_se_pp=100 * mc_se,
        )
    )
    table["abs_contribution_pp"] = table.contribution_pp.abs()
    table = table.sort_values("abs_contribution_pp", ascending=False).reset_index(
        drop=True
    )
    verification = dict(
        reference_probability=reference,
        prediction=predicted,
        additivity_error=float(reference + phi.sum() - predicted),
        maximum_mc_se_pp=float(mc_se.max() * 100),
        background_n=len(refs),
        requested_pairs=SHAP_PAIRS,
        actual_pairs=4 * pairs_per_run,
        pairs_per_run=pairs_per_run,
        n_input_features=len(adapter.columns),
        n_varying_features=varying_features,
        max_evals_per_run=max_evals_per_run,
        seed=int(state["seed"]),
        session=int(session),
        patient_id=str(case_id),
    )
    table.to_csv(PRIVATE / "variable_contributions.csv", index=False)
    np.save(PRIVATE / "shap_repeat_estimates.npy", estimates)
    adapter.inputs.iloc[[case_row]].to_csv(
        PRIVATE / "case_input_values.csv", index=False
    )
    state["frame"].iloc[refs][["id", "session"]].to_csv(
        PRIVATE / "reference_participants.csv", index=False
    )
    write_json(PRIVATE / "case_verification.json", verification)
    return (table, verification)
