"""Prepare local records, transcripts, features, embeddings, and participant splits."""

from pathlib import Path
import argparse
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smsc.preprocessing import features


def read_frame(path):
    return pd.read_csv(path, dtype={"id": str, "doc_id": str, "record_id": str})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in [
        "baseline",
        "extract-sessions",
        "sessions",
        "outcomes",
        "speech",
        "emotion",
        "emotion-segments",
        "text",
        "merge",
        "embed",
        "roles",
        "tasks",
        "transcribe",
    ]:
        command = commands.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name in {"baseline", "extract-sessions", "sessions", "tasks"}:
            command.add_argument("--config", type=Path, required=True)
        if name in {"text", "merge", "outcomes"}:
            command.add_argument("--sessions", type=Path, required=True)
        if name == "merge":
            command.add_argument("--modality", type=Path, action="append", default=[])
        if name in {"embed", "transcribe"}:
            command.add_argument(
                "--model",
                default="Qwen/Qwen3-Embedding-8B" if name == "embed" else "large-v3",
            )
            command.add_argument("--device", default="cpu")
            command.add_argument("--batch-size", type=int, default=8)
        if name == "roles":
            command.add_argument("--overrides", type=Path)
        if name == "tasks":
            command.add_argument("--session", type=int, required=True)
            command.add_argument("--participant", required=True)
            command.add_argument("--arm", choices=["pstair", "sc"], required=True)
    command = commands.add_parser("splits")
    command.add_argument("--config", type=Path, default=ROOT / "configs/splits.json")
    command.add_argument("--project", type=Path, default=ROOT)
    args = parser.parse_args()

    if args.command == "splits":
        from smsc.preprocessing.splits import export_splits

        folder, _ = export_splits(args.project, json.loads(args.config.read_text()))
        print(folder)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "transcribe":
        from smsc.preprocessing.transcription import Transcriber

        transcribe = Transcriber(args.model, args.device, args.batch_size)
        args.output.write_text(
            json.dumps(transcribe(args.input), ensure_ascii=False, indent=2) + "\n"
        )
        return
    if args.command == "roles":
        from smsc.preprocessing.speaker_roles import (
            assign_speakers_to_roles,
            parse_turns_from_transcript,
        )

        turns = (
            json.loads(args.input.read_text())
            if args.input.suffix == ".json"
            else parse_turns_from_transcript(args.input.read_text())
        )
        roles, _, _, _ = assign_speakers_to_roles(turns)
        if args.overrides:
            roles.update(json.loads(args.overrides.read_text()))
        for index, turn in enumerate(turns):
            turn["original_speaker"] = turn["speaker"]
            turn["speaker"] = roles[turn["speaker"]]
            turn["turn_index"] = index
        args.output.write_text(json.dumps(turns, ensure_ascii=False, indent=2) + "\n")
        return
    if args.command == "tasks":
        from smsc.preprocessing.tasks import configure_tasks, process_session

        configure_tasks(args.config)
        rows, *_ = process_session(
            json.loads(args.input.read_text()), args.session, args.arm
        )
        for row in rows:
            row.update(id=args.participant, session=args.session, arm=args.arm)
        pd.DataFrame(rows).to_csv(args.output, index=False)
        return
    frame = read_frame(args.input)
    if args.command == "baseline":
        result = features.baseline_features(frame, json.loads(args.config.read_text()))
    elif args.command == "extract-sessions":
        result = features.extract_session_records(
            frame, json.loads(args.config.read_text())
        )
    elif args.command == "sessions":
        result = features.session_metadata(frame, json.loads(args.config.read_text()))
    elif args.command == "outcomes":
        result = features.outcome_sessions(read_frame(args.sessions), frame)
    elif args.command == "speech":
        result = features.speech_features(frame)
    elif args.command == "emotion":
        result = features.emotion_features(frame)
    elif args.command == "emotion-segments":
        result = features.segments_from_recordings(
            frame, lambda path: json.loads(Path(path).read_text())
        )
    elif args.command == "text":
        result = features.combine_text_sources(frame, read_frame(args.sessions))
    elif args.command == "merge":
        result = features.merge_modalities(
            read_frame(args.sessions),
            frame,
            *(read_frame(path) for path in args.modality),
        )
    else:
        from smsc.preprocessing.embeddings import QwenEncoder, write_embedding_cache

        encoder = QwenEncoder(args.model, args.device, batch_size=args.batch_size)
        write_embedding_cache(frame, args.output, encoder)
        return
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
