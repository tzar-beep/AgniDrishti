"""CLI: ``python -m agnidrishti.ingest [--only grid|terrain|weather|firms]``."""

import argparse
import logging
import sys

from agnidrishti import grid
from agnidrishti.ingest import firms, terrain, weather
from agnidrishti.utils import FetchError, setup_logging

log = logging.getLogger("agnidrishti.ingest")
STEPS = {"grid": grid.main, "terrain": terrain.run, "weather": weather.run, "firms": firms.run}


def main() -> int:
    """Run the requested ingest steps in order; return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=list(STEPS), action="append")
    args = parser.parse_args()
    setup_logging()
    for name in args.only or list(STEPS):
        log.info("=== ingest step: %s ===", name)
        try:
            STEPS[name]()
        except (FetchError, RuntimeError) as exc:
            log.error("Step %s failed: %s", name, exc)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
