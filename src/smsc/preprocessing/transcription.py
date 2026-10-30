"""WhisperX transcription, alignment, and speaker diarization."""

from typing import Any, Dict, Optional
import os
import re

MERGE_GAP_SECONDS = 2.0
LEFT_ATTACHED = set("¿¡([{«")
RIGHT_ATTACHED = set(".,!?;:%)]}»")


class Transcriber:
    def __init__(
        self,
        model="large-v3",
        device="cpu",
        batch_size=8,
        token=None,
        minimum_speakers=2,
    ):
        import whisperx
        from whisperx.diarize import DiarizationPipeline

        self.whisperx = whisperx
        self.device = device
        self.batch_size = batch_size
        self.minimum_speakers = minimum_speakers
        compute_type = "float16" if device == "cuda" else "float32"
        self.asr = whisperx.load_model(model, device, compute_type=compute_type)
        self.diarization = DiarizationPipeline(
            token=token or os.environ.get("HF_TOKEN"), device=device
        )
        self.alignment = {}

    def __call__(self, media_path):
        from whisperx.diarize import assign_word_speakers

        audio = self.whisperx.load_audio(str(media_path))
        raw = self.asr.transcribe(audio, batch_size=self.batch_size)
        language = raw["language"]
        if language not in self.alignment:
            self.alignment[language] = self.whisperx.load_align_model(
                language_code=language, device=self.device
            )
        model, metadata = self.alignment[language]
        aligned = self.whisperx.align(
            raw["segments"],
            model,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        speakers = self.diarization(audio, min_speakers=self.minimum_speakers)
        assigned = assign_word_speakers(speakers, aligned)
        turns, _ = build_turns(assigned)
        for turn in turns:
            turn["language"] = language
        return turns


def join_token(existing: str, token: str) -> str:
    existing = re.sub("\\s+", " ", str(existing)).strip()
    token = re.sub("\\s+", " ", str(token)).strip()
    if not token:
        return existing
    if not existing:
        return token
    if token[0] in RIGHT_ATTACHED or token.startswith(("'", "’")):
        return existing + token
    if existing[-1] in LEFT_ATTACHED:
        return existing + token
    return existing + " " + token


def speaker_label(index: int) -> str:
    index += 1
    letters = []
    while index:
        index, remainder = divmod(index - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "".join(reversed(letters))


def build_turns(result: Dict[str, Any]):
    items = []
    previous_speaker = "SPEAKER_UNKNOWN"
    for segment in result.get("segments", []):
        words = [
            word
            for word in segment.get("words") or []
            if str(word.get("word", "")).strip()
        ]
        if words:
            for word in words:
                raw_speaker = str(
                    word.get("speaker") or segment.get("speaker") or previous_speaker
                )
                items.append(
                    {
                        "raw_speaker": raw_speaker,
                        "start": word.get("start", segment.get("start")),
                        "end": word.get("end", segment.get("end")),
                        "text": str(word.get("word", "")).strip(),
                    }
                )
                previous_speaker = raw_speaker
        elif str(segment.get("text", "")).strip():
            raw_speaker = str(segment.get("speaker") or previous_speaker)
            items.append(
                {
                    "raw_speaker": raw_speaker,
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "text": str(segment.get("text", "")).strip(),
                }
            )
            previous_speaker = raw_speaker
    turns = []
    for item in items:
        merge = False
        if turns and turns[-1]["raw_speaker"] == item["raw_speaker"]:
            previous_end = turns[-1].get("end")
            current_start = item.get("start")
            merge = (
                previous_end is None
                or current_start is None
                or float(current_start) - float(previous_end) <= MERGE_GAP_SECONDS
            )
        if merge:
            turns[-1]["text"] = join_token(turns[-1]["text"], item["text"])
            if item.get("end") is not None:
                turns[-1]["end"] = item["end"]
        else:
            turns.append(dict(item))
    mapping = {}
    for turn in turns:
        raw_speaker = turn["raw_speaker"]
        if raw_speaker not in mapping:
            mapping[raw_speaker] = speaker_label(len(mapping))
        turn["speaker"] = mapping[raw_speaker]
    return (turns, mapping)


def format_timestamp(seconds: Optional[float]) -> str:
    if seconds is None:
        return "??:??:??.???"
    seconds = max(float(seconds), 0.0)
    hours = int(seconds // 3600)
    minutes = int(seconds % 3600 // 60)
    remainder = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remainder:06.3f}"


def render_transcripts(turns):
    plain_lines = []
    timestamped_lines = []
    for turn in turns:
        text = turn["text"].strip()
        if not text:
            continue
        plain_lines.append(f"{turn['speaker']}: {text}")
        timestamped_lines.append(
            f"[{format_timestamp(turn.get('start'))} - {format_timestamp(turn.get('end'))}] {turn['speaker']}: {text}"
        )
    return ("\n\n".join(plain_lines) + "\n", "\n\n".join(timestamped_lines) + "\n")
