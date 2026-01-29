from __future__ import annotations

import select
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ProcessStatus:
    pid: int
    running: bool
    exit_code: Optional[int]
    tail: List[str]


class ProcessManager:
    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None
        self._log_buffer: List[str] = []

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, env: dict) -> None:
        if self.running:
            return
        venv_python = Path.cwd() / ".venv" / "bin" / "python"
        python_exe = str(venv_python) if venv_python.exists() else "python"
        command = [python_exe, "run.py"]
        self._process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._log_buffer.clear()

    def stop(self) -> None:
        if not self._process:
            return
        if self._process.poll() is None:
            self._process.terminate()

    def refresh(self) -> None:
        if not self._process or not self._process.stdout:
            return
        while True:
            ready, _, _ = select.select([self._process.stdout], [], [], 0)
            if not ready:
                break
            line = self._process.stdout.readline()
            if not line:
                break
            self._log_buffer.append(line.rstrip())

    def status(self, tail_lines: int = 200) -> ProcessStatus:
        self.refresh()
        running = self.running
        exit_code = None if running else (self._process.poll() if self._process else None)
        tail = self._log_buffer[-tail_lines:]
        pid = self._process.pid if self._process else -1
        return ProcessStatus(pid=pid, running=running, exit_code=exit_code, tail=tail)
