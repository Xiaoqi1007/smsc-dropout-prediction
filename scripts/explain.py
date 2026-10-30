"""Training-defined within-arm contrasts and optional prespecified SHAP cases."""

from pathlib import Path
import sys, argparse

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / "src"))
import pandas as pd
from threadpoolctl import threadpool_limits
from smsc.config import load_config
from smsc.experiment import read_json
from smsc.explain_support import (
    FrozenModel,
    participant_arms,
    load_model,
    direction_for_feature,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--cases", type=Path)
    p.add_argument("--skip-directions", action="store_true")
    a = p.parse_args()
    spec = load_config(a.config)
    run = P / "results/experiments" / spec["name"]
    D = run / "reports/figure4/directions"
    D.mkdir(parents=True, exist_ok=True)
    if not a.skip_directions:
        with threadpool_limits(1):
            for seed in spec["seeds"]:
                state = load_model(run, seed)
                adapter = FrozenModel(state)
                arms = participant_arms(state["frame"]).set_index("id").treatment_arm
                records = []
                for arm in ["P-STAIR", "SC"]:
                    adapter.arm_mask = state["frame"].id.map(arms).eq(arm).to_numpy()
                    for feature in adapter.columns:
                        records.append(
                            dict(
                                treatment_arm=arm,
                                **direction_for_feature(adapter, feature),
                            )
                        )
                pd.DataFrame(records).to_csv(D / f"seed_{seed}.csv", index=False)
                print("Directions", seed, flush=True)
        effects = pd.concat(
            [pd.read_csv(D / f"seed_{s}.csv") for s in spec["seeds"]], ignore_index=True
        )
        effects.to_csv(D / "direction_by_seed.csv", index=False)
        summary = []
        for (arm, feature), g in effects.groupby(["treatment_arm", "feature"]):
            valid = g[g.status.eq("ok")]
            mean = valid.effect_pp.mean()
            direction = (
                "Unavailable"
                if len(valid) < 8
                else (
                    "Positive"
                    if mean > 0.01
                    else "Negative" if mean < -0.01 else "Near zero"
                )
            )
            summary.append(
                dict(
                    treatment_arm=arm,
                    feature=feature,
                    direction=direction,
                    direction_mean_pp=mean,
                    direction_sd_pp=valid.effect_pp.std(),
                    direction_n_seeds=len(valid),
                    positive_seeds=int((valid.effect_pp > 0.01).sum()),
                    negative_seeds=int((valid.effect_pp < -0.01).sum()),
                    nonmonotone_seeds=int(valid.nonmonotone.sum()),
                )
            )
        pd.DataFrame(summary).to_csv(D / "summary.csv", index=False)
    if a.cases:
        from smsc.case_shap import compute_case_shap

        cases = read_json(a.cases)
        state = load_model(run, cases["case_seed"])
        for index, pid in enumerate(cases["case_ids"]):
            compute_case_shap(
                state,
                pid,
                cases["case_session"],
                run / f"reports/figure4/shap_cases/case_{index}",
            )
            print("SHAP case complete", index, flush=True)


if __name__ == "__main__":
    main()
