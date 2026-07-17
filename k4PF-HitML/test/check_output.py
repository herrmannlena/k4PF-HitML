#!/usr/bin/env python3
#
# Copyright (c) 2020-2024 Key4hep-Project.
#
# This file is part of Key4hep.
# See https://key4hep.github.io/key4hep-doc/ for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Minimal CI sanity check for PerformMLPF.py's output.

Asserts the HitPF collection is non-empty
"""
import sys

from podio.root_io import Reader


def main():
    if len(sys.argv) != 2:
        print("usage: check_output.py <output_HitPF.edm4hep.root>", file=sys.stderr)
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
