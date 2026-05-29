"""Tests for CPU topology detection."""

from unittest.mock import mock_open, patch

from llamacpp_cli.cpu_topology import (
    count_cpu_sockets,
    detect_numa_topology,
    get_cpus_for_node,
    get_numa_nodes,
)


class TestCountCpuSockets:
    """Tests for count_cpu_sockets function."""

    def test_single_socket(self):
        """Test detection of single physical socket."""
        cpuinfo_content = """processor	: 0
vendor_id	: GenuineIntel
cpu family	: 6
model		: 85
model name	: Intel(R) Xeon(R) CPU @ 2.00GHz
physical id	: 0
siblings	: 2
core id		: 0
cpu cores	: 2

processor	: 1
vendor_id	: GenuineIntel
cpu family	: 6
model		: 85
model name	: Intel(R) Xeon(R) CPU @ 2.00GHz
physical id	: 0
siblings	: 2
core id		: 1
cpu cores	: 2
"""
        with patch("builtins.open", mock_open(read_data=cpuinfo_content)):
            assert count_cpu_sockets() == 1

    def test_dual_socket(self):
        """Test detection of dual physical sockets."""
        cpuinfo_content = """processor	: 0
physical id	: 0
core id		: 0

processor	: 1
physical id	: 0
core id		: 1

processor	: 2
physical id	: 1
core id		: 0

processor	: 3
physical id	: 1
core id		: 1
"""
        with patch("builtins.open", mock_open(read_data=cpuinfo_content)):
            assert count_cpu_sockets() == 2

    def test_quad_socket(self):
        """Test detection of quad physical sockets."""
        cpuinfo_content = """processor	: 0
physical id	: 0

processor	: 16
physical id	: 1

processor	: 32
physical id	: 2

processor	: 48
physical id	: 3
"""
        with patch("builtins.open", mock_open(read_data=cpuinfo_content)):
            assert count_cpu_sockets() == 4

    def test_vm_no_physical_id(self):
        """Test VM without physical_id entries (should default to 1)."""
        cpuinfo_content = """processor	: 0
vendor_id	: GenuineIntel
cpu family	: 6
model		: 85
model name	: Intel(R) Xeon(R) CPU @ 2.00GHz
core id		: 0
cpu cores	: 2

processor	: 1
vendor_id	: GenuineIntel
cpu family	: 6
model		: 85
model name	: Intel(R) Xeon(R) CPU @ 2.00GHz
core id		: 1
cpu cores	: 2
"""
        with patch("builtins.open", mock_open(read_data=cpuinfo_content)):
            assert count_cpu_sockets() == 1

    def test_missing_cpuinfo(self):
        """Test behavior when /proc/cpuinfo is missing (non-Linux)."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert count_cpu_sockets() == 1

    def test_malformed_cpuinfo(self):
        """Test behavior with malformed /proc/cpuinfo."""
        cpuinfo_content = """processor	: 0
physical id INVALID LINE
core id		: 0
"""
        with patch("builtins.open", mock_open(read_data=cpuinfo_content)):
            # Should ignore malformed lines
            assert count_cpu_sockets() == 1

    def test_read_error(self):
        """Test handling of read errors."""
        with patch("builtins.open", side_effect=PermissionError):
            assert count_cpu_sockets() == 1


class TestGetNumaNodes:
    """Tests for get_numa_nodes function."""

    def test_single_node(self, tmp_path):
        """Test single NUMA node detection."""
        node_path = tmp_path / "node"
        node_path.mkdir()
        (node_path / "node0").mkdir()
        (node_path / "other_file").touch()

        with (
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=["node0", "other_file"]),
        ):
            assert get_numa_nodes() == [0]

    def test_dual_node(self, tmp_path):
        """Test dual NUMA node detection."""
        node_path = tmp_path / "node"
        node_path.mkdir()
        (node_path / "node0").mkdir()
        (node_path / "node1").mkdir()

        with (
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=["node0", "node1", "cpu"]),
        ):
            assert get_numa_nodes() == [0, 1]

    def test_quad_node(self, tmp_path):
        """Test quad NUMA node detection."""
        node_path = tmp_path / "node"
        node_path.mkdir()

        with (
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=["node0", "node1", "node2", "node3"]),
        ):
            assert get_numa_nodes() == [0, 1, 2, 3]

    def test_missing_sys_path(self):
        """Test behavior when /sys/devices/system/node is missing."""
        with patch("os.path.exists", return_value=False):
            assert get_numa_nodes() == [0]

    def test_empty_directory(self, tmp_path):
        """Test behavior with empty node directory."""
        node_path = tmp_path / "node"
        node_path.mkdir()

        with (
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=[]),
        ):
            assert get_numa_nodes() == [0]

    def test_non_node_entries(self, tmp_path):
        """Test filtering of non-node entries."""
        node_path = tmp_path / "node"
        node_path.mkdir()

        with (
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=["cpu", "memory", "node0", "node1"]),
        ):
            assert get_numa_nodes() == [0, 1]

    def test_read_error(self):
        """Test handling of read errors."""
        with (
            patch("os.path.exists", return_value=True),
            patch("os.listdir", side_effect=PermissionError),
        ):
            assert get_numa_nodes() == [0]


class TestGetCpusForNode:
    """Tests for get_cpus_for_node function."""

    def test_simple_list(self):
        """Test simple CPU list parsing."""
        cpulist_content = "0,1,2,3"
        with patch("builtins.open", mock_open(read_data=cpulist_content)):
            assert get_cpus_for_node(0) == [0, 1, 2, 3]

    def test_range_format(self):
        """Test CPU range parsing."""
        cpulist_content = "0-7"
        with patch("builtins.open", mock_open(read_data=cpulist_content)):
            assert get_cpus_for_node(0) == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_mixed_format(self):
        """Test mixed list and range parsing."""
        cpulist_content = "0,2,4-10,12"
        with patch("builtins.open", mock_open(read_data=cpulist_content)):
            assert get_cpus_for_node(0) == [0, 2, 4, 5, 6, 7, 8, 9, 10, 12]

    def test_complex_format(self):
        """Test complex CPU list with multiple ranges."""
        cpulist_content = "0-3,8-11,16-19,24-27"
        with patch("builtins.open", mock_open(read_data=cpulist_content)):
            expected = (
                list(range(0, 4))
                + list(range(8, 12))
                + list(range(16, 20))
                + list(range(24, 28))
            )
            assert get_cpus_for_node(0) == expected

    def test_single_cpu(self):
        """Test single CPU parsing."""
        cpulist_content = "5"
        with patch("builtins.open", mock_open(read_data=cpulist_content)):
            assert get_cpus_for_node(0) == [5]

    def test_missing_cpulist(self):
        """Test behavior when cpulist file is missing."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert get_cpus_for_node(0) == []

    def test_malformed_cpulist(self):
        """Test handling of malformed cpulist."""
        cpulist_content = "invalid-format"
        with patch("builtins.open", mock_open(read_data=cpulist_content)):
            # Should return empty list on error
            assert get_cpus_for_node(0) == []

    def test_whitespace_handling(self):
        """Test whitespace trimming."""
        cpulist_content = "  0,1,2,3  \n"
        with patch("builtins.open", mock_open(read_data=cpulist_content)):
            assert get_cpus_for_node(0) == [0, 1, 2, 3]


class TestDetectNumaTopology:
    """Tests for detect_numa_topology function."""

    def test_single_socket_single_node(self):
        """Test topology detection for single socket, single NUMA node."""
        cpuinfo_content = "physical id	: 0\nphysical id	: 0\n"

        with (
            patch("builtins.open", mock_open(read_data=cpuinfo_content)),
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=["node0"]),
            patch(
                "llamacpp_cli.cpu_topology.get_cpus_for_node",
                return_value=[0, 1, 2, 3],
            ),
        ):
            topology = detect_numa_topology()

            assert topology["num_sockets"] == 1
            assert topology["numa_nodes"] == [0]
            assert topology["cpus_per_node"] == {0: [0, 1, 2, 3]}
            assert topology["has_numa"] is False

    def test_dual_socket_dual_node(self):
        """Test topology detection for dual socket, dual NUMA node."""
        cpuinfo_content = "physical id	: 0\nphysical id	: 1\n"

        def mock_get_cpus(node_id):
            if node_id == 0:
                return [0, 1, 2, 3]
            return [4, 5, 6, 7]

        with (
            patch("builtins.open", mock_open(read_data=cpuinfo_content)),
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=["node0", "node1"]),
            patch(
                "llamacpp_cli.cpu_topology.get_cpus_for_node",
                side_effect=mock_get_cpus,
            ),
        ):
            topology = detect_numa_topology()

            assert topology["num_sockets"] == 2
            assert topology["numa_nodes"] == [0, 1]
            assert topology["cpus_per_node"] == {
                0: [0, 1, 2, 3],
                1: [4, 5, 6, 7],
            }
            assert topology["has_numa"] is True

    def test_quad_socket_quad_node(self):
        """Test topology detection for quad socket, quad NUMA node."""
        cpuinfo_content = (
            "physical id	: 0\nphysical id	: 1\nphysical id	: 2\nphysical id	: 3\n"
        )

        def mock_get_cpus(node_id):
            return list(range(node_id * 8, (node_id + 1) * 8))

        with (
            patch("builtins.open", mock_open(read_data=cpuinfo_content)),
            patch("os.path.exists", return_value=True),
            patch(
                "os.listdir",
                return_value=["node0", "node1", "node2", "node3"],
            ),
            patch(
                "llamacpp_cli.cpu_topology.get_cpus_for_node",
                side_effect=mock_get_cpus,
            ),
        ):
            topology = detect_numa_topology()

            assert topology["num_sockets"] == 4
            assert topology["numa_nodes"] == [0, 1, 2, 3]
            assert topology["has_numa"] is True
            assert len(topology["cpus_per_node"]) == 4

    def test_vm_fallback(self):
        """Test VM without physical_id or NUMA nodes."""
        cpuinfo_content = "processor	: 0\nprocessor	: 1\n"

        with (
            patch("builtins.open", mock_open(read_data=cpuinfo_content)),
            patch("os.path.exists", return_value=False),
        ):
            topology = detect_numa_topology()

            assert topology["num_sockets"] == 1
            assert topology["numa_nodes"] == [0]
            assert topology["has_numa"] is False

    def test_empty_numa_nodes(self):
        """Test behavior with no NUMA node information."""
        cpuinfo_content = "physical id	: 0\n"

        with (
            patch("builtins.open", mock_open(read_data=cpuinfo_content)),
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=[]),
        ):
            topology = detect_numa_topology()

            assert topology["num_sockets"] == 1
            assert topology["numa_nodes"] == [0]
            assert topology["cpus_per_node"] == {0: []}
            assert topology["has_numa"] is False
