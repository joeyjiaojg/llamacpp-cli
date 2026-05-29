"""CPU topology detection for NUMA optimization."""

import os
from typing import Any


def count_cpu_sockets() -> int:
    """Count physical CPU sockets by reading /proc/cpuinfo.

    Returns:
        Number of physical CPU sockets (1 if cannot determine)
    """
    try:
        with open("/proc/cpuinfo") as f:
            physical_ids = set()
            for line in f:
                if line.startswith("physical id"):
                    # Extract ID from "physical id : 0"
                    parts = line.split(":")
                    if len(parts) == 2:
                        physical_ids.add(parts[1].strip())

            # Return count, or 1 if no physical_id entries (VMs, etc)
            return len(physical_ids) if physical_ids else 1
    except FileNotFoundError:
        # Non-Linux system or no /proc/cpuinfo
        return 1
    except Exception:
        # Any other error, assume single socket
        return 1


def get_numa_nodes() -> list[int]:
    """Get list of NUMA node IDs from /sys/devices/system/node/.

    Returns:
        List of NUMA node IDs (e.g. [0, 1] for dual-socket)
    """
    try:
        node_path = "/sys/devices/system/node"
        if not os.path.exists(node_path):
            return [0]  # Default to single node

        nodes = []
        for entry in os.listdir(node_path):
            if entry.startswith("node") and entry[4:].isdigit():
                nodes.append(int(entry[4:]))

        return sorted(nodes) if nodes else [0]
    except Exception:
        return [0]


def get_cpus_for_node(node_id: int) -> list[int]:
    """Get list of CPU IDs for a specific NUMA node.

    Args:
        node_id: NUMA node ID

    Returns:
        List of CPU IDs on this node
    """
    try:
        cpulist_path = f"/sys/devices/system/node/node{node_id}/cpulist"
        with open(cpulist_path) as f:
            cpulist = f.read().strip()

        # Parse cpulist format: "0,2,4-10,12"
        cpus = []
        for part in cpulist.split(","):
            if "-" in part:
                start, end = part.split("-")
                cpus.extend(range(int(start), int(end) + 1))
            else:
                cpus.append(int(part))

        return sorted(cpus)
    except Exception:
        return []


def detect_numa_topology() -> dict[str, Any]:
    """Detect complete NUMA topology.

    Returns:
        Dictionary with:
        - num_sockets: Number of physical CPU sockets
        - numa_nodes: List of NUMA node IDs
        - cpus_per_node: Dict mapping node ID -> CPU list
        - has_numa: Whether system has multiple NUMA nodes
    """
    num_sockets = count_cpu_sockets()
    numa_nodes = get_numa_nodes()

    cpus_per_node = {}
    for node in numa_nodes:
        cpus_per_node[node] = get_cpus_for_node(node)

    return {
        "num_sockets": num_sockets,
        "numa_nodes": numa_nodes,
        "cpus_per_node": cpus_per_node,
        "has_numa": len(numa_nodes) > 1,
    }
