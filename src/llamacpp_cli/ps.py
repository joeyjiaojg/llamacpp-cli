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
