"""MERaLiON speech-emotion inference on consecutive audio segments."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid

EMOTIONS = ("Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgusted", "Surprised")
DIMENSIONS = ("Valance", "Arousal", "Dominance")
SAMPLE_RATE = 16000
CHUNK_SECONDS = 30
EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".mts", ".m2ts"}
MODEL_ID = "MERaLiON/MERaLiON-SER-v1"


def source_signature(path):
    stat = Path(path).stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_payload(payload, video):
    """Accept complete legacy outputs and new outputs with matching source stats."""
    info, rows = (payload["video_info"], payload["snippet_emotions"])
    if info["video_filename"] != video.name:
        raise ValueError("video_filename mismatch")
    duration = info["video_length_seconds"]
    if not finite(duration) or duration <= 0:
        raise ValueError("invalid audio duration")
    if not finite(info["video_length_mins"]) or not math.isclose(
        info["video_length_mins"], duration / 60, abs_tol=1e-06
    ):
        raise ValueError("inconsistent duration fields")
    expected = math.ceil(duration / CHUNK_SECONDS)
    if (
        not isinstance(rows, list)
        or len(rows) != expected
        or type(info["total_snippets"]) is not int
        or (info["total_snippets"] != expected)
    ):
        raise ValueError("incomplete snippet coverage")
    for index, row in enumerate(rows):
        if (
            not finite(row["start_second"])
            or row["start_second"] != index * CHUNK_SECONDS
        ):
            raise ValueError("invalid snippet start")
        logits = [row[f"{label}_raw_logit"] for label in EMOTIONS]
        probs = [row[f"{label}_softmax"] for label in EMOTIONS]
        dims = [row[f"{label}_dim"] for label in DIMENSIONS]
        if not all((finite(value) for value in logits + probs + dims)):
            raise ValueError("missing or non-finite model outputs")
        if any((p < 0 or p > 1 for p in probs)) or not math.isclose(
            sum(probs), 1, abs_tol=0.0001
        ):
            raise ValueError("invalid probabilities")
        if row["predicted_emotion"] != EMOTIONS[max(range(7), key=logits.__getitem__)]:
            raise ValueError("label/logit mismatch")
    if "source_signature" in info and info["source_signature"] != source_signature(
        video
    ):
        raise ValueError("source video changed since extraction")


def completion_status(output, video):
    if not output.exists():
        return (False, "missing")
    try:
        validate_payload(json.loads(output.read_text(encoding="utf8")), video)
        return (True, "complete")
    except (
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        AttributeError,
        OSError,
    ) as exc:
        return (False, f"invalid: {exc}")


def discover(video_dir, output_dir, recursive=False):
    """Preserve filename case on HPC storage and reject identical output paths."""
    iterator = video_dir.rglob("*") if recursive else video_dir.iterdir()
    videos = sorted(
        (p for p in iterator if p.is_file() and p.suffix.lower() in EXTENSIONS)
    )
    pairs, seen = ([], {})
    for video in videos:
        relative = video.relative_to(video_dir)
        output = output_dir / relative.parent / (video.stem + "_SER.json")
        key = str(output)
        if key in seen:
            raise ValueError(f"Output name collision: {seen[key]} and {video}")
        seen[key] = video
        pairs.append((video, output))
    return pairs


@contextmanager
def directory_lock(output_dir):
    """Linux/macOS advisory lock; the OS releases it if the process is killed."""
    import fcntl

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".ser-run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another SER job is using this output directory."
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def extract_pcm(video, destination):
    """Decode the first audio stream to a temporary mono float32 file."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode:
        raise RuntimeError(f"ffprobe failed: {probe.stderr[-2000:]}")
    streams = json.loads(probe.stdout).get("streams", [])
    if not streams or not streams[0].get("channels"):
        raise ValueError("No decodable audio stream")
    channels = int(streams[0]["channels"])
    mix = "+".join((f"{1 / channels:.12g}*c{i}" for i in range(channels)))
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-xerror",
            "-y",
            "-i",
            str(video),
            "-map",
            "0:a:0",
            "-vn",
            "-af",
            "pan=mono|c0=" + mix,
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
            str(destination),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-2000:]}")
    size = destination.stat().st_size
    if not size or size % 4:
        raise ValueError("Empty or incomplete decoded audio")
    return size // 4


@contextmanager
def meralion_construction_context():
    """Materialize the custom model's logspace architecture constants on CPU."""
    import torch
    from torch.overrides import TorchFunctionMode

    class ConstructorConstantsOnCPU(TorchFunctionMode):

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            if func is torch.logspace:
                kwargs["device"] = torch.device("cpu")
            return func(*args, **kwargs)

    with torch.device("cpu"), ConstructorConstantsOnCPU():
        yield


class MeralionPredictor:

    def __init__(
        self,
        model_path=MODEL_ID,
        device="auto",
        offline=False,
        cache_dir=None,
        revision=None,
    ):
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if cache_dir:
            os.environ["HF_HUB_CACHE"] = str(Path(cache_dir).expanduser().resolve())
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForAudioClassification,
            AutoProcessor,
        )
        from huggingface_hub import hf_hub_download

        self.torch = torch
        if not hasattr(torch.nn, "RMSNorm"):
            raise RuntimeError(
                "This MERaLiON model requires PyTorch with nn.RMSNorm (>=2.4)."
            )
        self.device = (
            ("cuda:0" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        if self.device.startswith("cuda") and (not torch.cuda.is_available()):
            raise RuntimeError(
                "CUDA requested but unavailable in this Python environment."
            )
        options = dict(trust_remote_code=True, local_files_only=offline)
        if cache_dir:
            options["cache_dir"] = str(Path(cache_dir).expanduser().resolve())
        if revision:
            options["revision"] = revision
        self.processor = AutoProcessor.from_pretrained(model_path, **options)
        config = AutoConfig.from_pretrained(model_path, **options)
        whisper_name = getattr(config, "whisper_name", None)
        if whisper_name and (not Path(whisper_name).is_dir()):
            whisper_config = hf_hub_download(
                repo_id=whisper_name,
                filename="config.json",
                local_files_only=offline,
                cache_dir=options.get("cache_dir"),
            )
            config.whisper_name = str(Path(whisper_config).parent)
        with meralion_construction_context():
            self.model = AutoModelForAudioClassification.from_pretrained(
                model_path,
                config=config,
                low_cpu_mem_usage=False,
                device_map=None,
                **options,
            )
        self.model.to(self.device)
        self.model.eval()
        self.metadata = {
            "model": str(model_path),
            "requested_revision": revision,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "torch_version": torch.__version__,
            "device": self.device,
            "constructor_logspace_device": "cpu",
        }
        print("Model ready on", self.device, flush=True)

    def __call__(self, samples):
        torch = self.torch
        inputs = self.processor(
            samples,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        with torch.inference_mode():
            out = self.model(
                **{
                    k: v.to(self.device)
                    for k, v in inputs.items()
                    if k in ("input_features", "attention_mask")
                }
            )
        logits, dims = (out["logits"], out["dims"])
        if tuple(logits.shape) != (1, 7) or dims.numel() != 3:
            raise ValueError("Unexpected MERaLiON output shape")
        scores = logits[0].float().cpu().tolist()
        probs = torch.softmax(logits, dim=1)[0].float().cpu().tolist()
        dimensions = dims.reshape(-1).float().cpu().tolist()
        row = {"predicted_emotion": EMOTIONS[max(range(7), key=scores.__getitem__)]}
        for name, score, prob in zip(EMOTIONS, scores, probs):
            row[f"{name}_raw_logit"] = score
            row[f"{name}_softmax"] = prob
        row.update(
            {f"{name}_dim": value for name, value in zip(DIMENSIONS, dimensions)}
        )
        return row


def analyze_video(video, predictor, temp_dir=None):
    import numpy as np

    signature = source_signature(video)
    with tempfile.TemporaryDirectory(prefix="smsc-ser-", dir=temp_dir) as work:
        pcm = Path(work) / "audio.f32"
        count = extract_pcm(video, pcm)
        step = SAMPLE_RATE * CHUNK_SECONDS
        rows = []
        with pcm.open("rb") as handle:
            for start in range(0, count, step):
                raw = handle.read(min(step, count - start) * 4)
                samples = np.frombuffer(raw, dtype="<f4").copy()
                if not np.isfinite(samples).all():
                    raise ValueError("Non-finite decoded audio")
                row = predictor(samples)
                row["start_second"] = start / SAMPLE_RATE
                rows.append(row)
                if len(rows) % 20 == 0:
                    print(
                        f"  {len(rows)}/{math.ceil(count / step)} snippets", flush=True
                    )
    if signature != source_signature(video):
        raise RuntimeError(
            "Video changed during processing; retry after upload completes."
        )
    seconds = count / SAMPLE_RATE
    payload = {
        "video_info": {
            "video_filename": video.name,
            "video_length_seconds": seconds,
            "video_length_mins": seconds / 60,
            "total_snippets": len(rows),
            "source_signature": signature,
        },
        "snippet_emotions": rows,
        "extraction_info": {
            **predictor.metadata,
            "sample_rate": SAMPLE_RATE,
            "snippet_seconds": CHUNK_SECONDS,
            "decoder": "ffmpeg",
            "audio_stream": 0,
            "channel_mix": "arithmetic mean",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    validate_payload(payload, video)
    return payload


def save_result(output, payload):
    """Commit only complete JSON; preserve an invalid previous output as a backup."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf8",
            dir=output.parent,
            prefix=".ser-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if output.exists():
            backup = output.with_name(output.name + ".invalid-" + uuid.uuid4().hex)
            shutil.copy2(output, backup)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_batch(
    video_dir,
    output_dir,
    model_path=MODEL_ID,
    device="auto",
    offline=False,
    cache_dir=None,
    revision=None,
    recursive=False,
    dry_run=False,
    limit=None,
    temp_dir=None,
):
    video_dir, output_dir = (
        Path(video_dir).expanduser().resolve(),
        Path(output_dir).expanduser().resolve(),
    )
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video folder not found: {video_dir}")
    if (
        video_dir == output_dir
        or video_dir in output_dir.parents
        or output_dir in video_dir.parents
    ):
        raise ValueError("Input and output must be separate, non-nested directories.")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    pairs = discover(video_dir, output_dir, recursive)
    if not pairs:
        raise ValueError(
            f"No supported videos found in {video_dir}; try --recursive for subfolders."
        )

    def scan():
        pending, complete = ([], 0)
        for video, output in pairs:
            valid, reason = completion_status(output, video)
            if valid:
                complete += 1
            else:
                pending.append((video, output, reason))
        print(
            f"Videos: {len(pairs)} | complete: {complete} | pending: {len(pending)}",
            flush=True,
        )
        return (pending, complete)

    if dry_run:
        pending, complete = scan()
        for video, _, reason in pending:
            print(f"PENDING [{reason}] {video.relative_to(video_dir)}", flush=True)
        return {
            "discovered": len(pairs),
            "skipped": complete,
            "pending": len(pending),
            "succeeded": 0,
            "failed": 0,
            "dry_run": True,
        }
    with directory_lock(output_dir):
        pending, complete = scan()
        selected = pending if limit is None else pending[:limit]
        summary = {
            "discovered": len(pairs),
            "skipped": complete,
            "pending": len(pending),
            "selected": len(selected),
            "succeeded": 0,
            "failed": 0,
            "deferred": len(pending) - len(selected),
        }
        if not selected:
            print("All videos already have complete results.", flush=True)
            return summary
        for command in ("ffmpeg", "ffprobe"):
            if shutil.which(command) is None:
                raise RuntimeError(
                    f"{command} is missing; load/install FFmpeg before running."
                )
        if temp_dir:
            temp_dir = str(Path(temp_dir).expanduser().resolve())
            Path(temp_dir).mkdir(parents=True, exist_ok=True)
        predictor = MeralionPredictor(model_path, device, offline, cache_dir, revision)
        logs = output_dir / "_logs"
        logs.mkdir(exist_ok=True)
        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        log_path = logs / (run_id + ".jsonl")
        with log_path.open("x", encoding="utf8") as log:
            for index, (video, output, reason) in enumerate(selected, 1):
                print(f"[{index}/{len(selected)}] {video.name} ({reason})", flush=True)
                record = {
                    "video": str(video),
                    "output": str(output),
                    "previous_status": reason,
                }
                try:
                    payload = analyze_video(video, predictor, temp_dir)
                    save_result(output, payload)
                    summary["succeeded"] += 1
                    record["status"] = "succeeded"
                except Exception as exc:
                    summary["failed"] += 1
                    record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    print("  FAILED:", record["error"], flush=True)
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
        summary["log"] = str(log_path)
        (logs / (run_id + ".summary.json")).write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf8"
        )
        print(json.dumps(summary, indent=2), flush=True)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", default="videos")
    parser.add_argument(
        "--output-dir", default="videos_audio_speech_emotion_recognition"
    )
    parser.add_argument(
        "--model", default=MODEL_ID, help="Hugging Face ID or local model directory"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, etc.")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--cache-dir", help="Hugging Face hub cache, including the Whisper config"
    )
    parser.add_argument("--revision", help="Optional model revision or commit hash")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List missing/invalid results without loading models",
    )
    parser.add_argument(
        "--limit", type=int, help="Maximum pending videos to process this run"
    )
    parser.add_argument(
        "--temp-dir", help="Temporary decoded audio directory; defaults to system temp"
    )
    args = parser.parse_args(argv)
    options = vars(args)
    options["model_path"] = options.pop("model")
    summary = run_batch(**options)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
