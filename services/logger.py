import atexit
import sys
from pathlib import Path

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        for stream in self.streams:
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


def enable_trace(log_file_path: str = "logs/output.log"):
    log_path = Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")
    log_file = log_path.open("a", encoding="utf-8", buffering=1)

    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    def close_log():
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.flush()
        log_file.close()

    atexit.register(close_log)
