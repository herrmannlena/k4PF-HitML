#!/usr/bin/env python3
"""Minimal CI sanity check for PerformMLPF.py's output.

Asserts the HitPF collection is non-empty somewhere across the processed
events -- catches a pipeline that runs cleanly (exit 0) but silently
produces no particles, which a pure "did it crash" smoke test would miss.
"""
import sys

from podio.root_io import Reader


def main():
    if len(sys.argv) != 2:
        print("usage: check_output.py <output_HitPF.root>", file=sys.stderr)
        return 2

    reader = Reader(sys.argv[1])
    n_events = 0
    n_particles = 0
    for event in reader.get("events"):
        n_events += 1
        n_particles += len(event.get("HitPF"))

    print(f"{n_events} events, {n_particles} HitPF particles total")

    if n_events == 0:
        print("FAIL: no events found", file=sys.stderr)
        return 1
    if n_particles == 0:
        print("FAIL: HitPF is empty across all events", file=sys.stderr)
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
