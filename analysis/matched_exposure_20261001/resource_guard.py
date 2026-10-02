#!/usr/bin/env python3
"""Run one TITAN fork and request graceful stop if device memory becomes unsafe."""

import argparse
import json
from pathlib import Path
import signal
import subprocess
import time


def meminfo():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemAvailable", "SwapFree"):
            values[key] = int(rest.strip().split()[0]) / 1024
    return values


def rss_mib(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
    return 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--min-available-mib", type=float, default=500.0)
    parser.add_argument("--min-swap-mib", type=float, default=100.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command or args.log.exists() or args.receipt.exists():
        parser.error("provide a command and unused log/receipt paths")
    initial = meminfo()
    if (initial["MemAvailable"] < 2000 or
            initial["MemAvailable"] + initial["SwapFree"] < 2500):
        parser.error("device has insufficient initial available memory for TITAN")
    started = time.monotonic()
    peak_rss = 0.0
    min_available = initial["MemAvailable"]
    min_swap = initial["SwapFree"]
    requested_stop = False
    with args.log.open("xb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            current = meminfo()
            peak_rss = max(peak_rss, rss_mib(process.pid))
            min_available = min(min_available, current["MemAvailable"])
            min_swap = min(min_swap, current["SwapFree"])
            if (not requested_stop and current["MemAvailable"] < args.min_available_mib and
                    current["SwapFree"] < args.min_swap_mib):
                requested_stop = True
                process.send_signal(signal.SIGINT)
            time.sleep(0.25)
        code = process.wait()
    receipt = {"schema": 1, "command": command, "exit_code": code,
               "safety_stop_requested": requested_stop,
               "elapsed_seconds": time.monotonic() - started,
               "initial_mem_available_mib": initial["MemAvailable"],
               "initial_swap_free_mib": initial["SwapFree"],
               "minimum_mem_available_mib": min_available,
               "minimum_swap_free_mib": min_swap,
               "peak_process_rss_mib": peak_rss}
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: receipt[key] for key in (
        "exit_code", "safety_stop_requested", "peak_process_rss_mib",
        "minimum_mem_available_mib", "minimum_swap_free_mib")}))
    if code or requested_stop:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
