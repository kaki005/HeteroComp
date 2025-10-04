import logging

import rich.pretty as pretty
from rich.logging import RichHandler


def log_init(level=logging.INFO):
    handler = RichHandler(markup=True, rich_tracebacks=True)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )
