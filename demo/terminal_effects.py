from __future__ import annotations

import builtins
import os
import sys
import time
from typing import Any


TYPEWRITER_DELAY = float(os.getenv("TYPEWRITER_DELAY", "0.005"))


def type_text(text: str, delay: float = TYPEWRITER_DELAY, end: str = "") -> None:
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    if end:
        sys.stdout.write(end)
        sys.stdout.flush()


def type_print(*values: Any, sep: str = " ", end: str = "\n", **_: Any) -> None:
    text = sep.join(str(value) for value in values)
    type_text(text, end=end)


def type_input(prompt: str = "") -> str:
    if prompt:
        type_text(prompt)
    return builtins.input()
