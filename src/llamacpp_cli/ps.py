"""Show running llama.cpp processes."""


def _find_llamacpp_processes() -> list[dict]:
    """Find running llama-server and llama-cli processes."""
    import subprocess

    procs = []
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,ppid,etime,comm,args"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        llama_comms = ("llama-server", "llama-cli")
        for line in result.stdout.strip().splitlines()[1:]:
            parts = line.split(None, 4)
            if len(parts) < 5:
                continue
            pid, ppid, etime, comm, args = parts
            if comm in llama_comms or any(c in args for c in llama_comms):
                procs.append(
                    {
                        "pid": int(pid),
                        "ppid": int(ppid),
                        "etime": etime,
                        "comm": comm,
                        "args": args,
                    }
                )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return procs


def show_running() -> None:
    """Display running llama.cpp processes."""
    procs = _find_llamacpp_processes()
    if not procs:
        print("No running llama.cpp processes found.")
        return

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Running Processes")
    table.add_column("PID", style="cyan")
    table.add_column("TYPE", style="green")
    table.add_column("UPTIME", style="yellow")
    table.add_column("COMMAND", style="dim", max_width=60)

    for p in procs:
        table.add_row(str(p["pid"]), p["comm"], p["etime"], p["args"])

    console.print(table)


def stop_servers() -> None:
    """Stop all running llama-server processes and their parent llamacpp serve processes."""
    import os
    import signal

    procs = _find_llamacpp_processes()
    server_procs = [p for p in procs if "llama-server" in p["comm"] or "llama-server" in p["args"]]

    if not server_procs:
        print("No llama-server processes found.")
        return

    # Collect parent PIDs (likely the Python llamacpp serve processes)
    parent_pids = set()
    for p in server_procs:
        ppid = p["ppid"]
        # Check if parent is a llamacpp serve process (Python process)
        try:
            import subprocess

            result = subprocess.run(
                ["ps", "-p", str(ppid), "-o", "args="],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0 and "llamacpp serve" in result.stdout:
                parent_pids.add(ppid)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    if parent_pids:
        print(f"Found {len(parent_pids)} llamacpp serve parent process(es) to stop:")
        for ppid in parent_pids:
            print(f"  PID {ppid}")
        print("\nStopping parent processes (this will clean up child llama-server processes)...")
        for ppid in parent_pids:
            try:
                os.kill(ppid, signal.SIGTERM)
                print(f"  Sent SIGTERM to PID {ppid}")
            except ProcessLookupError:
                print(f"  PID {ppid} already terminated")
            except PermissionError:
                print(f"  Permission denied for PID {ppid} (not your process?)")
    else:
        # No parent found, kill the server processes directly
        print(f"Found {len(server_procs)} llama-server process(es):")
        for p in server_procs:
            print(f"  PID {p['pid']}: {p['args']}")

        print("\nStopping servers...")
        for p in server_procs:
            try:
                os.kill(p["pid"], signal.SIGTERM)
                print(f"  Sent SIGTERM to PID {p['pid']}")
            except ProcessLookupError:
                print(f"  PID {p['pid']} already terminated")
            except PermissionError:
                print(f"  Permission denied for PID {p['pid']} (not your process?)")

    print("\nDone.")
