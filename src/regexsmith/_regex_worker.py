"""Runs untrusted regexes in a process that is allowed to die.

``re`` holds the GIL inside a match and cannot be cancelled from another thread, so
the only dependable timeout is killing a separate process. This file is started with
``python -I -S`` and imports nothing outside the standard library, which keeps a
respawn after a kill to tens of milliseconds and keeps it clear of torch and Metal.

Protocol: a "ready" line once imports are done, then one JSON object per line on stdin
and one result per line on stdout, flushed per query. The parent watches that stream,
so it always knows which query hung, and a slow start is never mistaken for a hang.
"""

import json
import re
import sys

MUST_ACCEPT, MUST_REJECT = 1, 2


def run(query: list, positives: list[str], negatives: list[str]) -> str:
    pattern, mode = query
    try:
        fullmatch = re.compile(pattern).fullmatch
    except re.error:
        return "e"
    if mode & MUST_ACCEPT and not all(fullmatch(text) for text in positives):
        return "0"
    if mode & MUST_REJECT and any(fullmatch(text) for text in negatives):
        return "0"
    return "1"


def main() -> None:
    stdout = sys.stdout
    stdout.write("ready\n")
    stdout.flush()
    for line in sys.stdin:
        request = json.loads(line)
        positives, negatives = request["pos"], request["neg"]
        for query in request["queries"]:
            stdout.write(run(query, positives, negatives) + "\n")
            stdout.flush()


if __name__ == "__main__":
    main()
