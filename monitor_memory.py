#!/usr/bin/env python3
"""
Monitor the RSS memory of a process tree and kill it if it exceeds a threshold.

Usage:
    python monitor_memory.py <pid> [--limit-gb 200] [--interval 5]

    # Or launch a command directly and monitor it:
    python monitor_memory.py --cmd "python -m vmf_hac.experiments.explore_embedding_models" [--limit-gb 200]
"""

import argparse
import signal
import subprocess
import sys
import time

try:
    import psutil
except ImportError:
    print("psutil is required: pip install psutil", file=sys.stderr)
    sys.exit(1)

BYTES_PER_GB = 1024**3


def rss_gb(pid: int) -> float:
    """Return total RSS (GiB) of the process and all its children."""
    try:
        root = psutil.Process(pid)
        procs = [root, *root.children(recursive=True)]
        total = sum(p.memory_info().rss for p in procs if p.is_running())
        return total / BYTES_PER_GB
    except psutil.NoSuchProcess:
        return 0.0


def kill_tree(pid: int) -> None:
    """Send SIGTERM to the process group, then SIGKILL after 5 s if still alive."""
    try:
        root = psutil.Process(pid)
        children = root.children(recursive=True)
        for p in [root, *children]:
            try:
                p.send_signal(signal.SIGTERM)
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs([root, *children], timeout=5)
        for p in alive:
            try:
                p.send_signal(signal.SIGKILL)
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass


def monitor(pid: int, limit_gb: float, interval: float) -> None:
    print(f"Monitoring PID {pid} | limit={limit_gb} GiB | interval={interval}s")
    print(f"{'Time':>10}  {'RSS (GiB)':>12}  {'Limit (GiB)':>12}")
    print("-" * 40)

    try:
        while True:
            if not psutil.pid_exists(pid):
                print(f"\nPID {pid} no longer exists — exiting monitor.")
                break

            usage = rss_gb(pid)
            timestamp = time.strftime("%H:%M:%S")
            print(f"{timestamp:>10}  {usage:>12.2f}  {limit_gb:>12.1f}", flush=True)

            if usage >= limit_gb:
                print(f"\n⚠️  Memory limit exceeded ({usage:.2f} GiB >= {limit_gb} GiB). Killing PID {pid}...")
                kill_tree(pid)
                print("Process killed.")
                sys.exit(1)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitor interrupted by user.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory monitor / watchdog.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("pid", nargs="?", type=int, help="PID of an already-running process.")
    group.add_argument("--cmd", type=str, help="Shell command to launch and monitor.")
    parser.add_argument("--limit-gb", type=float, default=200.0, help="RSS limit in GiB (default: 200)")
    parser.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds (default: 5)")
    args = parser.parse_args()

    if args.cmd:
        proc = subprocess.Popen(args.cmd, shell=True)
        pid = proc.pid
        print(f"Launched '{args.cmd}' as PID {pid}")
    else:
        pid = args.pid

    monitor(pid, args.limit_gb, args.interval)


if __name__ == "__main__":
    main()
