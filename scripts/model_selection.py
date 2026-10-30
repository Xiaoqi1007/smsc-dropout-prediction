"""Participant-grouped training-set cross-validation for four-source fusion."""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier
from smsc._core.xgb import xgb_parameters
from smsc.utils import sha256_text as participant_key
from smsc.config import load_config, training_config
from smsc.experiment import sha, write_json
from smsc._core import components, pipeline
from smsc._core.preassessment import filter_preassessment

SOURCES = ["preassessment", "session", "emotion_speech", "language"]
FAMILIES = ["xgboost", "logistic_regression", "neural_network"]


def prepare(source, frame, embedding, fit, config, base, audit):
    if source == "language":
        return (embedding, {"representation": "cumulative_history"})
    if source == "preassessment":
        columns, _ = filter_preassessment(config, frame, fit)
        raw = frame[columns].apply(pd.to_numeric, errors="coerce")
        raw = raw.replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        matrix = imputer.fit(raw.loc[fit]).transform(raw).astype(np.float32)
        return (matrix, {"raw_requested": columns})
    columns = (
        audit["video_columns"]
        if source == "emotion_speech"
        else [
            c
            for c in audit["tabular_columns"]
            if c.startswith(tuple(config["session_prefixes"]))
            or c in config["session_columns"]
        ]
    )
    matrix, _, schema = pipeline.prepare_block(
        base,
        frame,
        columns,
        fit,
        config["max_categorical_levels"],
        scale_numeric=(source == "session"),
        dtype="float64" if source == "emotion_speech" else "float32",
    )
    return (matrix, schema)


def refit_network(config, base, matrix, y, ids, fit, held, seed, epochs):
    """Fit a fresh network for an epoch count selected within the training fold."""
    torch = base.torch
    base.set_seed(seed)
    device = base.resolve_device(config)
    nnc = config["nn"]
    scaler = StandardScaler().fit(matrix[fit])
    xtrain = torch.from_numpy(scaler.transform(matrix[fit]).astype(np.float32))
    target = torch.from_numpy(y[fit].astype(np.float32))
    weights = torch.from_numpy(
        base.training_weights(config, ids[fit], y[fit]).astype(np.float32)
    )
    model = base.EmbeddingNN(
        matrix.shape[1], int(nnc["hidden_units"]), float(nnc["dropout"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(nnc["learning_rate"]),
        weight_decay=float(nnc["l2_strength"]),
    )
    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed * 1000003 + epoch)
        order = torch.randperm(len(target), generator=generator)
        for start in range(0, len(target), int(nnc["batch_size"])):
            ix = order[start : start + int(nnc["batch_size"])]
            xb, yb, wb = (
                xtrain[ix].to(device),
                target[ix].to(device),
                weights[ix].to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            each = torch.nn.functional.binary_cross_entropy_with_logits(
                model(xb), yb, reduction="none"
            )
            loss = (each * wb).sum() / wb.sum()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite network loss")
            loss.backward()
            optimizer.step()
    model.eval()
    return base.predict_torch_model(
        model, scaler, matrix[held], device, int(nnc["prediction_batch_size"])
    )


def run(args):
    specification = load_config(args.config)
    config = training_config(specification, args.project)
    base, bundle, split_rows, audit = pipeline.load_inputs(config)
    base.torch.set_num_threads(args.threads)
    seed = args.seed
    outer = base.participant_masks(bundle, split_rows, config, seed)
    current = base.load_current_embedding(config, bundle, "full_text", seed)
    embedding = base.cumulative_history(config, bundle, current)[0][outer["train"]]
    frame = bundle.frame.loc[outer["train"]].reset_index(drop=True)
    forbidden = set(bundle.frame.loc[~outer["train"], "id"])
    assert not set(frame.id) & forbidden
    people = (
        frame[["id", "ever_dropout"]]
        .drop_duplicates("id")
        .sort_values("id")
        .reset_index(drop=True)
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    input_paths = [Path(config[k]) for k in ["data_path", "split_path"]]
    input_paths += [Path(config["preassessment"]["cm_path"])]
    input_paths += [base.embedding_paths(config, "full_text", seed)["embedding"]]
    provenance = {
        "seed": seed,
        "n_training_participants": len(people),
        "n_training_rows": len(frame),
        "training_participant_keys": [participant_key(v) for v in people.id],
        "validation_test_overlap": len(set(frame.id) & forbidden),
        "config": specification,
        "input_hashes": {p.name: sha(p) for p in input_paths},
        "feature_audit": audit,
    }
    audit_path = out / f"input_audit_seed_{seed}.json"
    if audit_path.exists():
        if json.loads(audit_path.read_text()) != provenance:
            raise ValueError(
                "Inputs or configuration changed; use a new output directory"
            )
    else:
        write_json(audit_path, provenance)
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    assignments = []
    for fold, (a, b) in enumerate(cv.split(people.id, people.ever_dropout)):
        fit_ids, held_ids = (set(people.id.iloc[a]), set(people.id.iloc[b]))
        inner_a, inner_b = train_test_split(
            a,
            test_size=0.2,
            stratify=people.ever_dropout.iloc[a],
            random_state=seed * 100 + fold,
        )
        assert not fit_ids & held_ids
        for i in b:
            assignments.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "participant_key": participant_key(people.id.iloc[i]),
                }
            )
        for source in SOURCES:
            path = out / f"seed_{seed}_fold_{fold}_{source}.json"
            if path.exists():
                continue
            started = time.monotonic()
            sf = (
                frame.drop_duplicates("id").reset_index(drop=True)
                if source == "preassessment"
                else frame
            )
            em = None if source == "preassessment" else embedding
            ids, y = (sf.id.to_numpy(), sf.ever_dropout.to_numpy())
            fit, held = (
                sf.id.isin(fit_ids).to_numpy(),
                sf.id.isin(held_ids).to_numpy(),
            )
            it = sf.id.isin(set(people.id.iloc[inner_a])).to_numpy()
            iv = sf.id.isin(set(people.id.iloc[inner_b])).to_numpy()
            assert not set(ids[it]) & set(ids[iv])
            assert not set(ids[it | iv]) & set(ids[held])
            matrix, schema = prepare(source, sf, em, fit, config, base, audit)
            weights = base.training_weights(config, ids[fit], y[fit])
            records, predictions = ([], [])

            def record(family, candidate, probability, **extra):
                assert np.isfinite(probability).all()
                assert ((probability >= 0) & (probability <= 1)).all()
                records.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "source": source,
                        "family": family,
                        "candidate": candidate,
                        "auroc": float(roc_auc_score(y[held], probability)),
                        "n_fit_people": len(set(ids[fit])),
                        "n_held_people": len(set(ids[held])),
                        **extra,
                    }
                )
                predictions.append(
                    pd.DataFrame(
                        {
                            "participant_key": [participant_key(v) for v in ids[held]],
                            "session": sf.session.to_numpy()[held],
                            "y": y[held],
                            "probability": probability,
                            "family": family,
                            "candidate": candidate,
                        }
                    )
                )

            for index, candidate in enumerate(config["xgb"]["candidates"]):
                model = XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=args.threads,
                    random_state=seed,
                    verbosity=0,
                    **xgb_parameters(candidate),
                )
                model.fit(matrix[fit], y[fit], sample_weight=weights)
                record("xgboost", index, model.predict_proba(matrix[held])[:, 1])
            scaler = StandardScaler().fit(matrix[fit])
            model = LogisticRegression(
                C=config["source_lr"]["C"],
                max_iter=config["source_lr"]["max_iter"],
                solver="lbfgs",
                random_state=seed,
            )
            model.fit(scaler.transform(matrix[fit]), y[fit], sample_weight=weights)
            record(
                "logistic_regression",
                0,
                model.predict_proba(scaler.transform(matrix[held]))[:, 1],
            )
            inner_matrix, _ = prepare(source, sf, em, it, config, base, audit)
            _, payload, _ = components.fit_text_model(
                config,
                base,
                inner_matrix,
                y,
                ids,
                {"train": it, "validation": iv},
                seed,
            )
            epochs = payload["epochs"]
            probability = refit_network(
                config, base, matrix, y, ids, fit, held, seed, epochs
            )
            record("neural_network", 0, probability, epochs=epochs)
            pd.concat(predictions).to_csv(path.with_suffix(".csv"), index=False)
            write_json(
                path,
                {
                    "records": records,
                    "schema": schema,
                    "seconds": time.monotonic() - started,
                },
            )
            print(
                f"seed={seed} fold={fold} source={source} complete ({time.monotonic() - started:.1f}s)",
                flush=True,
            )
    pd.DataFrame(assignments).to_csv(out / f"folds_seed_{seed}.csv", index=False)


def summarize(args):
    out = Path(args.output)
    specification = load_config(args.config)
    seeds = specification["seeds"]
    first_inputs = None
    for seed in seeds:
        audit = json.loads((out / f"input_audit_seed_{seed}.json").read_text())
        if audit["config"] != specification:
            raise ValueError("Cross-validation configuration differs across splits")
        if first_inputs is None:
            first_inputs = audit["input_hashes"]
        elif audit["input_hashes"] != first_inputs:
            raise ValueError("Cross-validation input files differ across splits")
    records = []
    for seed, fold, source in itertools.product(seeds, range(3), SOURCES):
        records.extend(
            json.loads((out / f"seed_{seed}_fold_{fold}_{source}.json").read_text())[
                "records"
            ]
        )
    scores = pd.DataFrame(records)
    assert len(scores) == len(seeds) * 3 * 4 * 6
    assert not scores.duplicated(
        ["seed", "fold", "source", "family", "candidate"]
    ).any()
    scores.to_csv(out / "all_fold_scores.csv", index=False)
    averages = (
        scores.groupby(["seed", "source", "family", "candidate"])
        .auroc.mean()
        .reset_index()
    )
    selected = averages.sort_values(
        ["auroc", "candidate"], ascending=[False, True]
    ).drop_duplicates(["seed", "source", "family"])
    selected.to_csv(out / "source_models_by_split.csv", index=False)
    source_summary = (
        selected.groupby(["source", "family"])
        .auroc.agg(auroc_mean="mean", auroc_sd="std")
        .reset_index()
    )
    source_summary.to_csv(out / "source_model_summary.csv", index=False)
    candidates = {
        (int(r.seed), r.source, r.family): int(r.candidate)
        for r in selected.itertuples()
    }
    combinations = list(itertools.product(FAMILIES, repeat=4))
    fusion_rows = []
    for seed in seeds:
        assignments = pd.read_csv(out / f"folds_seed_{seed}.csv")
        expected = set(
            json.loads((out / f"input_audit_seed_{seed}.json").read_text())[
                "training_participant_keys"
            ]
        )
        assert (
            assignments.participant_key.is_unique
            and set(assignments.participant_key) == expected
        )
        for fold in range(3):
            branches = {}
            for source in SOURCES:
                pred = pd.read_csv(out / f"seed_{seed}_fold_{fold}_{source}.csv")
                if source == "session":
                    base_rows = pred.loc[
                        pred.family.eq("logistic_regression"),
                        ["participant_key", "session", "y"],
                    ].reset_index(drop=True)
                    assert not base_rows.duplicated(
                        ["participant_key", "session"]
                    ).any()
                    assert set(base_rows.participant_key) == set(
                        assignments.loc[assignments.fold.eq(fold), "participant_key"]
                    )
            for source in SOURCES:
                pred = pd.read_csv(out / f"seed_{seed}_fold_{fold}_{source}.csv")
                for family in FAMILIES:
                    part = pred.loc[
                        pred.family.eq(family)
                        & pred.candidate.eq(candidates[seed, source, family])
                    ]
                    keys = (
                        ["participant_key"]
                        if source == "preassessment"
                        else ["participant_key", "session"]
                    )
                    aligned = base_rows.merge(
                        part[keys + ["y", "probability"]],
                        on=keys,
                        how="left",
                        validate="many_to_one",
                        suffixes=("", "_source"),
                    )
                    assert aligned.probability.notna().all()
                    np.testing.assert_array_equal(aligned.y, aligned.y_source)
                    branches[source, family] = aligned.probability.to_numpy()
                    match = scores.loc[
                        scores.seed.eq(seed)
                        & scores.fold.eq(fold)
                        & scores.source.eq(source)
                        & scores.family.eq(family)
                        & scores.candidate.eq(candidates[seed, source, family])
                    ]
                    assert (
                        abs(
                            roc_auc_score(part.y, part.probability)
                            - match.auroc.iloc[0]
                        )
                        < 1e-12
                    )
            for index, families in enumerate(combinations, 1):
                probability = np.mean(
                    [branches[s, f] for s, f in zip(SOURCES, families)], axis=0
                )
                y = base_rows.y.to_numpy()
                auroc = roc_auc_score(y, probability)
                n1, n0 = (int(y.sum()), int((1 - y).sum()))
                independent = (
                    rankdata(probability)[y == 1].sum() - n1 * (n1 + 1) / 2
                ) / (n1 * n0)
                assert abs(auroc - independent) < 1e-12
                fusion_rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "combination": index,
                        **dict(zip(SOURCES, families)),
                        "auroc": auroc,
                    }
                )
    fusion = pd.DataFrame(fusion_rows)
    fusion.to_csv(out / "fusion_by_fold.csv", index=False)
    by_split = (
        fusion.groupby(["seed", "combination"] + SOURCES).auroc.mean().reset_index()
    )
    by_split.to_csv(out / "fusion_by_split.csv", index=False)
    ranking = (
        by_split.groupby(["combination"] + SOURCES)
        .auroc.agg(auroc_mean="mean", auroc_sd="std")
        .reset_index()
    )
    ranking = ranking.sort_values(
        ["auroc_mean", "combination"], ascending=[False, True]
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(out / "all_81_combinations_ranked.csv", index=False)
    ranking.head(3).to_csv(out / "top3_fusion_combinations.csv", index=False)
    write_json(
        out / "verification.json",
        {
            "n_splits": len(seeds),
            "n_folds": 3,
            "n_candidate_source_scores": len(scores),
            "n_fusion_scores": len(fusion),
            "sd_unit": "ten split means",
            "best_combination": ranking.iloc[0].to_dict(),
        },
    )
    print(source_summary.to_string(index=False), flush=True)
    print(ranking.head(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if not args.summarize and args.seed not in load_config(args.config)["seeds"]:
        parser.error("--seed must identify a split in the configuration")
    with threadpool_limits(args.threads):
        summarize(args) if args.summarize else run(args)
