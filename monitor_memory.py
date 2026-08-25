#!/usr/bin/env python3
"""
Monitor system-wide used RAM and kill a target process when it exceeds a threshold.

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


def system_used_gb() -> tuple[float, float]:
    """Return (used_GiB, total_GiB) of physical RAM (excludes reclaimable cache/buffers)."""
    vm = psutil.virtual_memory()
    return vm.used / BYTES_PER_GB, vm.total / BYTES_PER_GB


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
    _, total_gb = system_used_gb()
    print(
        f"Monitoring PID {pid} | system RAM={total_gb:.1f} GiB | kill threshold={limit_gb} GiB | interval={interval}s"
    )
    print(f"{'Time':>10}  {'Used (GiB)':>12}  {'Total (GiB)':>12}  {'Used %':>8}")
    print("-" * 50)

    try:
        while True:
            if not psutil.pid_exists(pid):
                print(f"\nPID {pid} no longer exists — exiting monitor.")
                break

            used_gb, total_gb = system_used_gb()
            pct = 100 * used_gb / total_gb if total_gb > 0 else 0
            timestamp = time.strftime("%H:%M:%S")
            print(f"{timestamp:>10}  {used_gb:>12.2f}  {total_gb:>12.1f}  {pct:>7.1f}%", flush=True)

            if used_gb >= limit_gb:
                print(f"\n⚠️  System memory limit exceeded ({used_gb:.2f} GiB >= {limit_gb} GiB). Killing PID {pid}...")
                kill_tree(pid)
                print("Process killed.")
                sys.exit(1)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitor interrupted by user.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="System memory watchdog — kills a process when system RAM exceeds a threshold."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("pid", nargs="?", type=int, help="PID of an already-running process.")
    group.add_argument("--cmd", type=str, help="Shell command to launch and monitor.")
    parser.add_argument(
        "--limit-gb", type=float, default=200.0, help="System used-RAM kill threshold in GiB (default: 200)"
    )
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
