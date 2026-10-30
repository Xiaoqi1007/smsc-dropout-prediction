"""Run training, model verification and analyses in the local project."""

from pathlib import Path
import argparse, subprocess, sys

P = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--config", type=Path, default=P / "configs/analysis.json")
p.add_argument("--cases", type=Path)
p.add_argument("--explanations", action="store_true")
a = p.parse_args()
config = a.config.resolve()
commands = [
    [sys.executable, str(P / "run.py"), c, "--config", str(config)]
    for c in ["train", "verify", "evaluate"]
]
commands.append(
    [sys.executable, str(P / "scripts/analyze.py"), "--config", str(config)]
    + (["--cases", str(a.cases.resolve())] if a.cases else [])
)
if a.explanations:
    commands += [
        [sys.executable, str(P / "scripts/importance.py"), "--config", str(config)],
        [sys.executable, str(P / "scripts/explain.py"), "--config", str(config)]
        + (["--cases", str(a.cases.resolve())] if a.cases else []),
    ]
    if a.cases:
        commands.append(
            [
                sys.executable,
                str(P / "scripts/figure4.py"),
                "--config",
                str(config),
                "--cases",
                str(a.cases.resolve()),
            ]
        )
for c in commands:
    print("Running", Path(c[1]).name, c[2:], flush=True)
    subprocess.run(c, cwd=P, check=True)
