"""Figure 4: treatment-arm importance and individual SHAP explanations."""

from pathlib import Path
import argparse, sys, json, textwrap, shutil

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / "src"))
import numpy as np, pandas as pd, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from smsc.config import load_config
from smsc.experiment import read_json, write_json
from smsc.explain_support import compact_contributions, case_metadata, label

p = argparse.ArgumentParser()
p.add_argument("--config", type=Path, required=True)
p.add_argument("--cases", type=Path, required=True)
a = p.parse_args()
spec = load_config(a.config)
run = P / "results/experiments" / spec["name"]
cases = read_json(a.cases)
OUT = run / "reports/figure4"
for f in ["tables", "figures"]:
    (OUT / f).mkdir(exist_ok=True)
importance = pd.read_csv(OUT / "arm_importance/tables/feature_importance_both_arms.csv")
directions = pd.read_csv(OUT / "directions/summary.csv")
summary = importance.merge(
    directions, on=["treatment_arm", "feature"], validate="one_to_one"
)
assert summary.n_seeds.eq(len(spec["seeds"])).all()
assert not set(spec.get("features", {}).get("exclude_columns", [])) & set(
    summary.feature
)
summary["label"] = summary.feature.map(label)
summary.to_csv(OUT / "tables/arm_importance_and_directions.csv", index=False)
summary = pd.read_csv(OUT / "tables/arm_importance_and_directions.csv")
RED, BLUE, GRAY = "#C44E52", "#0072B2", "#92999f"
colors = {"Positive": RED, "Negative": BLUE, "Unavailable": GRAY, "Near zero": GRAY}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig = plt.figure(figsize=(19.5, 16.5), facecolor="white")
axes = [
    fig.add_axes(r)
    for r in [
        [0.18, 0.647, 0.30, 0.273],
        [0.68, 0.647, 0.30, 0.273],
        [0.18, 0.275, 0.30, 0.222],
        [0.68, 0.275, 0.30, 0.222],
    ]
]


def labels(table):
    result = []
    for value in table.label:
        if value.startswith("Participant word count: "):
            lines = ["Participant word count:"] + textwrap.wrap(
                value.split(": ", 1)[1], 32, break_on_hyphens=False
            )
        elif value == "Participant-to-clinician word count ratio":
            lines = ["Participant-to-clinician", "word count ratio"]
        else:
            lines = textwrap.wrap(value, 32, break_on_hyphens=False)
        result.append("\n".join(lines))
    return result


for ax, arm, letter, x in zip(axes[:2], ["P-STAIR", "SC"], ["A", "B"], [0.02, 0.52]):
    t = summary[summary.treatment_arm.eq(arm)].sort_values("rank").head(10)
    ax.barh(
        np.arange(10),
        t.importance_mean,
        xerr=t.importance_sd,
        color=[colors[d] for d in t.direction],
        height=0.58,
        capsize=2.5,
        error_kw={"elinewidth": 1, "ecolor": "#444"},
    )
    ax.set_yticks(np.arange(10), labels(t), fontsize=10.5)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, pad=9)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(
        "Decrease in subgroup test AUROC (ΔAUROC)", fontsize=10.5, labelpad=10
    )
    ax.axvline(0, color="#777", lw=0.8)
    ax.grid(axis="x", color="#eceff2")
    ax.set_axisbelow(True)
    fig.text(x, 0.97, letter, fontsize=20, weight="bold")
    fig.text(
        x + 0.03,
        0.97,
        f"{arm}: Top 10 feature importance",
        fontsize=16,
        weight="semibold",
    )
    t.to_csv(OUT / f"tables/Figure4{letter}_top10.csv", index=False)
fig.legend(
    handles=[
        Patch(color=RED, label="Higher predicted risk"),
        Patch(color=BLUE, label="Lower predicted risk"),
        Patch(color=GRAY, label="No clear contrast"),
    ],
    loc="lower center",
    bbox_to_anchor=(0.53, 0.565),
    ncol=3,
    frameon=False,
    fontsize=11,
    title="A–B colors: mean model response to higher vs lower feature values",
)
texts = cases.get("text_summaries", ["", ""])
frame = pd.read_csv(run / "data/features.csv", dtype={"id": str})
for index, (ax, letter, x) in enumerate(zip(axes[2:], ["C", "D"], [0.02, 0.52])):
    folder = OUT / f"shap_cases/case_{index}"
    table = pd.read_csv(folder / "variable_contributions.csv")
    table["label"] = table.feature.map(label)
    v = json.loads((folder / "case_verification.json").read_text())
    if int(v["session"]) != int(cases["case_session"]) or str(v["patient_id"]) != str(
        cases["case_ids"][index]
    ):
        raise ValueError("SHAP results do not match the requested participant-session")
    detail = case_metadata(frame, v)
    arm = detail["treatment_arm"]
    shown = compact_contributions(table, 7)
    delta = shown.contribution_pp.to_numpy()
    base = v["reference_probability"] * 100
    final = v["prediction"] * 100
    ends = base + np.cumsum(delta)
    starts = np.r_[base, ends[:-1]]
    np.testing.assert_allclose(ends[-1], final, atol=1e-10)
    lo, hi = min(base, final, *starts, *ends), max(base, final, *starts, *ends)
    span = max(hi - lo, 8)
    ax.set_xlim(lo - 0.12 * span, hi + 0.24 * span)
    for i, (start, end, value) in enumerate(zip(starts, ends, delta)):
        ax.barh(
            i,
            abs(value),
            left=min(start, end),
            height=0.56,
            color=RED if value > 0 else BLUE,
            zorder=3,
        )
        if i < len(delta) - 1:
            ax.plot([end, end], [i + 0.28, i + 0.72], color="#aaa", ls=":", lw=0.8)
        ax.text(
            max(start, end) + 0.025 * span, i, f"{value:+.1f}", va="center", fontsize=10
        )
    ax.set_yticks(np.arange(len(shown)), labels(shown), fontsize=10.5)
    ax.set_ylim(len(shown) - 0.5, -0.7)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.spines["left"].set_visible(False)
    ax.axvline(base, color="#888", ls="--", lw=0.9)
    ax.axvline(final, color="#333", lw=1)
    ax.set_xlabel("Calibrated dropout probability (%)", fontsize=10.5, labelpad=10)
    ax.set_title(
        f"Reference: {base:.1f}%   →   Prediction: {final:.1f}%", fontsize=11.5, pad=12
    )
    ax.grid(axis="x", color="#eceff2")
    ax.set_axisbelow(True)
    fig.text(x, 0.545, letter, fontsize=20, weight="bold")
    fig.text(
        x + 0.03,
        0.545,
        f"{arm}: individual prediction and text context",
        fontsize=15.5,
        weight="semibold",
    )
    fig.text(
        x + 0.03,
        0.529,
        f'Held-out participant · Session {detail["session"]} · Observed outcome: {detail["outcome"]}',
        fontsize=10.5,
        color="#505A63",
    )
    table.to_csv(OUT / f"tables/Figure4{letter}_SHAP_all_features.csv", index=False)
    shown.to_csv(OUT / f"tables/Figure4{letter}_SHAP_displayed.csv", index=False)
    textscore = float(
        table.loc[table.feature.eq("text_nn_score"), "contribution_pp"].iloc[0]
    )
    fig.text(
        x + 0.03,
        0.170,
        "Anonymous text context",
        fontsize=13.5,
        weight="semibold",
        va="top",
    )
    fig.text(
        x + 0.03,
        0.147,
        "\n".join(textwrap.wrap(texts[index], 77)),
        fontsize=13,
        va="top",
        linespacing=1.3,
    )
    fig.text(
        x + 0.03,
        0.075,
        f"Language-only model risk score contribution: {textscore:+.1f} percentage points",
        fontsize=11,
        color="#505A63",
    )
fig.legend(
    handles=[
        Patch(color=RED, label="Raises this prediction"),
        Patch(color=BLUE, label="Lowers this prediction"),
    ],
    loc="lower center",
    bbox_to_anchor=(0.53, 0.190),
    ncol=2,
    frameon=False,
    fontsize=11,
    title="C–D: SHAP contributions in percentage points",
)
for ext in ["png", "pdf", "svg"]:
    fig.savefig(
        OUT / f"figures/Figure_4_importance_SHAP.{ext}",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
plt.close(fig)

for ext in ["png", "pdf", "svg"]:
    shutil.copy2(
        OUT / f"figures/Figure_4_importance_SHAP.{ext}",
        run / f"reports/figures/Figure_4_importance_SHAP.{ext}",
    )
checks = []
for i in range(2):
    c = pd.read_csv(OUT / f"shap_cases/case_{i}/variable_contributions.csv")
    v = read_json(OUT / f"shap_cases/case_{i}/case_verification.json")
    np.testing.assert_allclose(
        v["reference_probability"] + c.contribution_pp.sum() / 100,
        v["prediction"],
        atol=1e-12,
        rtol=0,
    )
    assert not set(spec.get("features", {}).get("exclude_columns", [])) & set(c.feature)
    checks.append(
        dict(
            case_index=i,
            additivity_error=v["additivity_error"],
            background_n=v["background_n"],
            features=v["n_input_features"],
        )
    )
write_json(
    OUT / "verification.json",
    dict(
        seeds=len(spec["seeds"]),
        repeats=30,
        features_per_arm=int(summary.groupby("treatment_arm").size().max()),
        shap_cases=checks,
    ),
)
print("Figure 4 assembled", OUT)
