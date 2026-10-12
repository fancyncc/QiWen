"""Run validation commands while filtering configured secrets from their output."""
from __future__ import annotations

import os
import re
import subprocess
import sys

from dotenv import dotenv_values
from .runtime import ROOT


def redactor():
    values = {**dotenv_values(ROOT / ".env"), **os.environ}
    secrets = [v for k, v in values.items() if v and re.search(r"KEY|TOKEN|PASSWORD|SECRET|BASE_URL", k, re.I)]
    def clean(text):
        for value in sorted(secrets, key=len, reverse=True):
            text = text.replace(value, "[REDACTED]")
        text = re.sub(r"(?i)(bearer\s+)[\w.\-]+", r"\1[REDACTED]", text)
        text = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[REDACTED]@", text)
        text = re.sub(r"(?i)(api[_-]?key|access[_-]?token|authorization)([\s\"':=]+)[^\s,\"'}]+", r"\1\2[REDACTED]", text)
        return text
    return clean


def main():
    command = sys.argv[1:]
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("Supply validation command")
    clean = redactor()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", errors="replace", cwd=os.getcwd())
    for line in process.stdout:
        print(clean(line), end="", flush=True)
    return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
