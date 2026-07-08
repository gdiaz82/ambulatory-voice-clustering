"""Standard logging setup.

Replaces the ad-hoc ``print`` calls and the ``_Tee`` stdout-hijacking hack used in
the original research scripts. Call :func:`configure_logging` once at process start
(the CLI does this); everywhere else just use ``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import logging
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(level: int = logging.INFO, logfile: Path | None = None) -> None:
    """Configure the root logger with a console handler and an optional file handler.

    Args:
        level: Root logging level.
        logfile: If given, logs are additionally written to this file (parent
            directories are created as needed).
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, mode="w", encoding="utf-8"))

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    for handler in handlers:
        handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    for handler in handlers:
        root.addHandler(handler)
