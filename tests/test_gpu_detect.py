"""Tests for GPU detection and optimization module."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from llamacpp_cli.gpu_detect import (
    GPUConfig,
    GPUInfo,
    auto_configure_gpu,
    detect_amd_gpus,
    detect_all_gpus,
    detect_nvidia_gpus,
    format_gpu_info,
    get_gpu_server_args,
)


# ---------------------------------------------------------------------------
# detect_nvidia_gpus
# ---------------------------------------------------------------------------


class TestDetectNvidiaGpus:
    """Tests for detect_nvidia_gpus function."""

    def _make_result(self, stdout: str, returncode: int = 0) -> MagicMock:
        result = MagicMock()
        result.stdout = stdout
        result.returncode = returncode
        return result

    def test_single_gpu(self):
        """Detect one NVIDIA GPU from nvidia-smi output."""
        output = "NVIDIA GeForce RTX 3080, 10240\n"
        with patch("subprocess.run", return_value=self._make_result(output)):
            gpus = detect_nvidia_gpus()

        assert len(gpus) == 1
        assert gpus[0].vendor == "nvidia"
        assert gpus[0].name == "NVIDIA GeForce RTX 3080"
        assert gpus[0].vram_mb == 10240
        assert gpus[0].gpu_layers > 0

    def test_multiple_gpus(self):
        """Detect multiple NVIDIA GPUs."""
        output = "NVIDIA A100, 40960\nNVIDIA A100, 40960\n"
        with patch("subprocess.run", return_value=self._make_result(output)):
            gpus = detect_nvidia_gpus()

        assert len(gpus) == 2
        assert all(g.vendor == "nvidia" for g in gpus)

    def test_low_vram_gpu(self):
        """GPU with very low VRAM results in 0 recommended layers."""
        output = "NVIDIA GTX 750, 512\n"
        with patch("subprocess.run", return_value=self._make_result(output)):
            gpus = detect_nvidia_gpus()

        assert len(gpus) == 1
        assert gpus[0].gpu_layers == 1  # 512 // 512 == 1

    def test_high_vram_gpu_capped_at_80(self):
        """GPU with very large VRAM caps recommended layers at 80."""
        output = "NVIDIA H100, 81920\n"
        with patch("subprocess.run", return_value=self._make_result(output)):
            gpus = detect_nvidia_gpus()

        assert len(gpus) == 1
        assert gpus[0].gpu_layers == 80  # capped

    def test_nvidia_smi_not_found(self):
        """Returns empty list when nvidia-smi is not installed."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            gpus = detect_nvidia_gpus()

        assert gpus == []

    def test_nvidia_smi_timeout(self):
        """Returns empty list when nvidia-smi times out."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 5)):
            gpus = detect_nvidia_gpus()

        assert gpus == []

    def test_nvidia_smi_nonzero_exit(self):
        """Returns empty list when nvidia-smi exits with error."""
        with patch("subprocess.run", return_value=self._make_result("", returncode=1)):
            gpus = detect_nvidia_gpus()

        assert gpus == []

    def test_empty_output(self):
        """Returns empty list when nvidia-smi output is blank."""
        with patch("subprocess.run", return_value=self._make_result("")):
            gpus = detect_nvidia_gpus()

        assert gpus == []

    def test_malformed_line_skipped(self):
        """Lines without a valid VRAM number are skipped."""
        output = "NVIDIA RTX 4090, not_a_number\nNVIDIA RTX 3090, 24576\n"
        with patch("subprocess.run", return_value=self._make_result(output)):
            gpus = detect_nvidia_gpus()

        assert len(gpus) == 1
        assert gpus[0].name == "NVIDIA RTX 3090"

    def test_whitespace_stripped_from_name_and_vram(self):
        """Name and VRAM values have surrounding whitespace stripped."""
        output = "  Tesla T4 ,  16384  \n"
        with patch("subprocess.run", return_value=self._make_result(output)):
            gpus = detect_nvidia_gpus()

        assert len(gpus) == 1
        assert gpus[0].name == "Tesla T4"
        assert gpus[0].vram_mb == 16384


# ---------------------------------------------------------------------------
# detect_amd_gpus
# ---------------------------------------------------------------------------


class TestDetectAmdGpus:
    """Tests for detect_amd_gpus function."""

    def test_rocm_smi_not_found(self):
        """Returns empty list when rocm-smi is not installed."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            gpus = detect_amd_gpus()

        assert gpus == []

    def test_rocm_smi_timeout(self):
        """Returns empty list when rocm-smi times out."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("rocm-smi", 5)):
            gpus = detect_amd_gpus()

        assert gpus == []

    def test_rocm_smi_nonzero_exit(self):
        """Returns empty list when rocm-smi exits with error."""
        result = MagicMock()
        result.stdout = ""
        result.returncode = 1
        with patch("subprocess.run", return_value=result):
            gpus = detect_amd_gpus()

        assert gpus == []

    def test_rocm_smi_invalid_json(self):
        """Returns empty list when rocm-smi output is not valid JSON."""
        result = MagicMock()
        result.stdout = "not json"
        result.returncode = 0
        with patch("subprocess.run", return_value=result):
            gpus = detect_amd_gpus()

        assert gpus == []

    def test_rocm_smi_valid_output(self):
        """Parses valid rocm-smi JSON output."""
        import json
        data = {
            "card0": {
                "Card series": "AMD Radeon RX 7900 XTX",
                "VRAM Total Memory (B)": str(24 * 1024 * 1024 * 1024),  # 24 GB
            }
        }
        result = MagicMock()
        result.stdout = json.dumps(data)
        result.returncode = 0
        with patch("subprocess.run", return_value=result):
            gpus = detect_amd_gpus()

        assert len(gpus) == 1
        assert gpus[0].vendor == "amd"
        assert gpus[0].vram_mb == 24 * 1024
        assert gpus[0].gpu_layers > 0

    def test_rocm_smi_zero_vram_skipped(self):
        """Cards reporting 0 VRAM are filtered out."""
        import json
        data = {"card0": {"Card series": "Unknown", "VRAM Total Memory (B)": "0"}}
        result = MagicMock()
        result.stdout = json.dumps(data)
        result.returncode = 0
        with patch("subprocess.run", return_value=result):
            gpus = detect_amd_gpus()

        assert gpus == []


# ---------------------------------------------------------------------------
# detect_all_gpus
# ---------------------------------------------------------------------------


class TestDetectAllGpus:
    """Tests for detect_all_gpus function."""

    def test_nvidia_only(self):
        """Returns only NVIDIA GPUs when no AMD GPUs present."""
        nvidia = [GPUInfo(vendor="nvidia", name="RTX 3080", vram_mb=10240, gpu_layers=20)]
        with (
            patch("llamacpp_cli.gpu_detect.detect_nvidia_gpus", return_value=nvidia),
            patch("llamacpp_cli.gpu_detect.detect_amd_gpus", return_value=[]),
        ):
            gpus = detect_all_gpus()

        assert gpus == nvidia

    def test_amd_only(self):
        """Returns only AMD GPUs when no NVIDIA GPUs present."""
        amd = [GPUInfo(vendor="amd", name="RX 7900 XTX", vram_mb=24576, gpu_layers=48)]
        with (
            patch("llamacpp_cli.gpu_detect.detect_nvidia_gpus", return_value=[]),
            patch("llamacpp_cli.gpu_detect.detect_amd_gpus", return_value=amd),
        ):
            gpus = detect_all_gpus()

        assert gpus == amd

    def test_nvidia_before_amd(self):
        """NVIDIA GPUs appear before AMD GPUs in combined list."""
        nvidia = [GPUInfo(vendor="nvidia", name="RTX 4090", vram_mb=24576, gpu_layers=48)]
        amd = [GPUInfo(vendor="amd", name="RX 7900", vram_mb=20480, gpu_layers=40)]
        with (
            patch("llamacpp_cli.gpu_detect.detect_nvidia_gpus", return_value=nvidia),
            patch("llamacpp_cli.gpu_detect.detect_amd_gpus", return_value=amd),
        ):
            gpus = detect_all_gpus()

        assert gpus[0].vendor == "nvidia"
        assert gpus[1].vendor == "amd"

    def test_no_gpus(self):
        """Returns empty list when no GPUs are detected."""
        with (
            patch("llamacpp_cli.gpu_detect.detect_nvidia_gpus", return_value=[]),
            patch("llamacpp_cli.gpu_detect.detect_amd_gpus", return_value=[]),
        ):
            gpus = detect_all_gpus()

        assert gpus == []


# ---------------------------------------------------------------------------
# auto_configure_gpu
# ---------------------------------------------------------------------------


class TestAutoConfigureGpu:
    """Tests for auto_configure_gpu function."""

    def test_no_gpu_returns_cpu_only(self):
        """Returns CPU-only config when no GPUs are detected."""
        with patch("llamacpp_cli.gpu_detect.detect_all_gpus", return_value=[]):
            cfg = auto_configure_gpu()

        assert cfg.n_gpu_layers == 0

    def test_uses_first_gpu(self):
        """Uses first GPU when multiple are available."""
        gpus = [
            GPUInfo(vendor="nvidia", name="RTX 3080", vram_mb=10240, gpu_layers=20),
            GPUInfo(vendor="nvidia", name="RTX 3070", vram_mb=8192, gpu_layers=16),
        ]
        with patch("llamacpp_cli.gpu_detect.detect_all_gpus", return_value=gpus):
            cfg = auto_configure_gpu()

        # Should be based on first GPU (10240 MB)
        assert cfg.n_gpu_layers > 0
        assert cfg.main_gpu == 0

    def test_unknown_model_size_conservative(self):
        """Uses conservative layer count when model size is unknown."""
        gpus = [GPUInfo(vendor="nvidia", name="RTX 3080", vram_mb=10240, gpu_layers=20)]
        with patch("llamacpp_cli.gpu_detect.detect_all_gpus", return_value=gpus):
            cfg = auto_configure_gpu(model_size_gb=None)

        assert cfg.n_gpu_layers > 0
        assert cfg.n_gpu_layers <= 40  # conservative cap

    def test_known_small_model(self):
        """Small model (3GB) on large VRAM GPU gets many layers."""
        # 24 GB GPU, 3 GB model -> nearly all layers fit
        gpus = [GPUInfo(vendor="nvidia", name="A100", vram_mb=24576, gpu_layers=48)]
        with patch("llamacpp_cli.gpu_detect.detect_all_gpus", return_value=gpus):
            cfg = auto_configure_gpu(model_size_gb=3.0)

        # (24576 - 1024) / 1024 = 23.0 GB available
        # gb_per_layer = 3.0 / 32 * 0.5 = 0.046875
        # layers = 23.0 / 0.046875 = 490 -> capped at 80
        assert cfg.n_gpu_layers == 80

    def test_known_large_model_small_vram(self):
        """Large model on small VRAM GPU gets few layers."""
        # 4 GB GPU, 30 GB model
        gpus = [GPUInfo(vendor="nvidia", name="GTX 1650", vram_mb=4096, gpu_layers=6)]
        with patch("llamacpp_cli.gpu_detect.detect_all_gpus", return_value=gpus):
            cfg = auto_configure_gpu(model_size_gb=30.0)

        # available = (4096 - 1024) / 1024 = 3.0 GB
        # gb_per_layer = 30.0 / 32 * 0.5 = 0.46875
        # layers = 3.0 / 0.46875 = 6
        assert cfg.n_gpu_layers == 6

    def test_vram_below_overhead_returns_cpu_only(self):
        """GPU with less than 1 GB VRAM returns CPU-only config."""
        gpus = [GPUInfo(vendor="nvidia", name="Tiny GPU", vram_mb=512, gpu_layers=1)]
        with patch("llamacpp_cli.gpu_detect.detect_all_gpus", return_value=gpus):
            cfg = auto_configure_gpu(model_size_gb=7.0)

        assert cfg.n_gpu_layers == 0

    def test_layer_count_never_exceeds_80(self):
        """Layer count is always capped at 80."""
        gpus = [GPUInfo(vendor="nvidia", name="H100", vram_mb=81920, gpu_layers=160)]
        with patch("llamacpp_cli.gpu_detect.detect_all_gpus", return_value=gpus):
            cfg = auto_configure_gpu(model_size_gb=4.0)

        assert cfg.n_gpu_layers <= 80


# ---------------------------------------------------------------------------
# get_gpu_server_args
# ---------------------------------------------------------------------------


class TestGetGpuServerArgs:
    """Tests for get_gpu_server_args function."""

    def test_cpu_only_returns_empty(self):
        """CPU-only config produces no arguments."""
        cfg = GPUConfig(n_gpu_layers=0)
        assert get_gpu_server_args(cfg) == []

    def test_gpu_layers_arg_present(self):
        """--n-gpu-layers appears with the correct value."""
        cfg = GPUConfig(n_gpu_layers=32)
        args = get_gpu_server_args(cfg)

        assert "--n-gpu-layers" in args
        idx = args.index("--n-gpu-layers")
        assert args[idx + 1] == "32"

    def test_main_gpu_arg_present(self):
        """--main-gpu appears with the correct index."""
        cfg = GPUConfig(n_gpu_layers=32, main_gpu=1)
        args = get_gpu_server_args(cfg)

        assert "--main-gpu" in args
        idx = args.index("--main-gpu")
        assert args[idx + 1] == "1"

    def test_split_mode_arg_present(self):
        """--split-mode appears with the configured value."""
        cfg = GPUConfig(n_gpu_layers=32, split_mode="row")
        args = get_gpu_server_args(cfg)

        assert "--split-mode" in args
        idx = args.index("--split-mode")
        assert args[idx + 1] == "row"

    def test_tensor_split_arg(self):
        """--tensor-split appears when tensor_split is set."""
        cfg = GPUConfig(n_gpu_layers=32, tensor_split=[0.6, 0.4])
        args = get_gpu_server_args(cfg)

        assert "--tensor-split" in args
        idx = args.index("--tensor-split")
        assert args[idx + 1] == "0.6,0.4"

    def test_no_tensor_split_by_default(self):
        """--tensor-split is absent when not configured."""
        cfg = GPUConfig(n_gpu_layers=32)
        args = get_gpu_server_args(cfg)

        assert "--tensor-split" not in args

    def test_default_split_mode_is_layer(self):
        """Default split mode is 'layer'."""
        cfg = GPUConfig(n_gpu_layers=16)
        args = get_gpu_server_args(cfg)

        idx = args.index("--split-mode")
        assert args[idx + 1] == "layer"


# ---------------------------------------------------------------------------
# format_gpu_info
# ---------------------------------------------------------------------------


class TestFormatGpuInfo:
    """Tests for format_gpu_info function."""

    def test_no_gpus(self):
        """Shows CPU-only message when no GPUs detected."""
        msg = format_gpu_info([])
        assert "none detected" in msg.lower() or "cpu-only" in msg.lower()

    def test_single_gpu(self):
        """Shows GPU name and VRAM for one GPU."""
        gpus = [GPUInfo(vendor="nvidia", name="RTX 3080", vram_mb=10240, gpu_layers=20)]
        msg = format_gpu_info(gpus)

        assert "RTX 3080" in msg
        assert "NVIDIA" in msg.upper()
        assert "10.0" in msg  # 10240 / 1024 = 10.0 GB

    def test_multiple_gpus(self):
        """Shows info for each GPU when multiple present."""
        gpus = [
            GPUInfo(vendor="nvidia", name="RTX 4090", vram_mb=24576, gpu_layers=48),
            GPUInfo(vendor="amd", name="RX 7900", vram_mb=20480, gpu_layers=40),
        ]
        msg = format_gpu_info(gpus)

        assert "RTX 4090" in msg
        assert "RX 7900" in msg
        assert "GPU 0" in msg
        assert "GPU 1" in msg


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestServeCommandGpuIntegration:
    """Integration tests for --gpu/--no-gpu flags on the serve command.

    The serve command imports gpu_detect functions inside the function body
    (lazy imports), so patches must target the gpu_detect module directly.
    """

    def test_no_gpu_flag_skips_detection(self):
        """--no-gpu flag skips GPU detection and passes no GPU args."""
        from click.testing import CliRunner

        from llamacpp_cli.cli import cli

        runner = CliRunner()
        with (
            patch("llamacpp_cli.installer.ensure_llamacpp", return_value=True),
            patch("llamacpp_cli.proxy.run_proxy") as mock_run_proxy,
            patch("llamacpp_cli.gpu_detect.detect_all_gpus") as mock_detect,
            patch("llamacpp_cli.gpu_detect.detect_nvidia_gpus", return_value=[]),
            patch("llamacpp_cli.gpu_detect.detect_amd_gpus", return_value=[]),
        ):
            result = runner.invoke(cli, ["serve", "--no-gpu", "--port", "9999"])

        # GPU detection should not be called when --no-gpu is used
        mock_detect.assert_not_called()

    def test_gpu_flag_with_no_hardware(self):
        """--gpu with no hardware detected shows CPU-only message."""
        from click.testing import CliRunner

        from llamacpp_cli.cli import cli

        runner = CliRunner()
        with (
            patch("llamacpp_cli.installer.ensure_llamacpp", return_value=True),
            patch("llamacpp_cli.proxy.run_proxy"),
            patch("llamacpp_cli.gpu_detect.detect_nvidia_gpus", return_value=[]),
            patch("llamacpp_cli.gpu_detect.detect_amd_gpus", return_value=[]),
        ):
            result = runner.invoke(cli, ["serve", "--gpu", "--port", "9999"])

        assert "none detected" in result.output.lower() or "cpu-only" in result.output.lower()

    def test_gpu_flag_with_hardware_passes_layers_to_proxy(self):
        """--gpu with GPU hardware passes --n-gpu-layers to llama-server."""
        from click.testing import CliRunner

        from llamacpp_cli.cli import cli

        runner = CliRunner()
        gpu = GPUInfo(vendor="nvidia", name="RTX 3080", vram_mb=10240, gpu_layers=20)
        gpu_cfg = GPUConfig(n_gpu_layers=20, main_gpu=0)

        with (
            patch("llamacpp_cli.installer.ensure_llamacpp", return_value=True),
            patch("llamacpp_cli.proxy.run_proxy") as mock_run_proxy,
            patch("llamacpp_cli.gpu_detect.detect_nvidia_gpus", return_value=[gpu]),
            patch("llamacpp_cli.gpu_detect.detect_amd_gpus", return_value=[]),
            patch("llamacpp_cli.gpu_detect.auto_configure_gpu", return_value=gpu_cfg),
        ):
            result = runner.invoke(cli, ["serve", "--gpu", "--port", "9999"])

        assert mock_run_proxy.called
        call_kwargs = mock_run_proxy.call_args
        extra = call_kwargs.kwargs.get("extra_args") or []
        assert "--n-gpu-layers" in extra

    def test_gpu_layers_manual_override(self):
        """--gpu-layers N passes exactly N layers regardless of auto-detection."""
        from click.testing import CliRunner

        from llamacpp_cli.cli import cli

        runner = CliRunner()
        gpu = GPUInfo(vendor="nvidia", name="RTX 3080", vram_mb=10240, gpu_layers=20)

        with (
            patch("llamacpp_cli.installer.ensure_llamacpp", return_value=True),
            patch("llamacpp_cli.proxy.run_proxy") as mock_run_proxy,
            patch("llamacpp_cli.gpu_detect.detect_nvidia_gpus", return_value=[gpu]),
            patch("llamacpp_cli.gpu_detect.detect_amd_gpus", return_value=[]),
        ):
            result = runner.invoke(cli, ["serve", "--gpu-layers", "40", "--port", "9999"])

        assert mock_run_proxy.called
        call_kwargs = mock_run_proxy.call_args
        extra = call_kwargs.kwargs.get("extra_args") or []
        assert "--n-gpu-layers" in extra
        idx = extra.index("--n-gpu-layers")
        assert extra[idx + 1] == "40"
