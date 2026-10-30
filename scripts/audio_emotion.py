"""Extract MERaLiON-SER emotion probabilities from local recordings."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smsc.preprocessing.audio_emotion import main

if __name__ == "__main__":
    main()
