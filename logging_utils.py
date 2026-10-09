from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from typing import TextIO


class TeeStream:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    @property
    def encoding(self) -> str | None:
        return getattr(self.streams[0], "encoding", None)

    def fileno(self) -> int:
        return self.streams[0].fileno()


@contextmanager
def tee_output(log_path: str | Path):
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        sys.stdout = TeeStream(original_stdout, handle)  
        sys.stderr = TeeStream(original_stderr, handle)  
        try:
            yield path
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
