"""
ComfyUI WAN Animate 2.2 Complete Installation Script for Windows
Uses SOURCE CODE version x.x.x of ComfyUI - Python installation!
"""

import sys
import os
import platform
import subprocess
import urllib.request
import urllib.error
import zipfile
import signal
import logging
import argparse
import threading
import time
import shutil
import socket
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# ============================================================================
# USER-EDITABLE CONFIGURATION - Edit these values to customize installation
# ============================================================================

# Python Version for Virtual Environment
# Specify version to use (e.g., "3.12", "3.11.5")
PYTHON_VERSION = "3.9"

# Find versions at: https://github.com/comfyanonymous/ComfyUI/releases
COMFYUI_VERSION = "v0.3.65"

# None, "auto", "default", or "" = auto-generate from COMFYUI_VERSION
# Example: r"D:\ComfyUI-0.3.65"
INSTALL_PATH = None

# Virtual Environment Name
VENV_NAME = "venv"

# CUDA version must match your GPU drivers (CUDA required for WAN Animate 2.2)
# Examples: "2.8.0+cu128" (CUDA 12.8), "2.8.0+cu124" (CUDA 12.4), "2.9.0+cu128" (CUDA 12.8)
PYTORCH_VERSION = "2.8.0+cu128"

# Model Repository - Where to download WAN Animate 2.2 models
HUGGINGFACE_BASE = "https://huggingface.co/Aitrepreneur/FLX/resolve/main"

# Additional Components - Performance optimizations
# Triton version constraint (see: https://github.com/woct0rdho/triton-windows/releases)
# - Pytorch 2.9.x requires Triton 3.5
# - Pytorch 2.8.x requires Triton 3.4
# - Pytorch 2.7.x requires Triton 3.3
# Example: TRITON_VERSION = "3.5"
TRITON_VERSION = "3.5"

# IMPORTANT: Must match your "PYTORCH_VERSION" above!
# SageAttention wheel URL (see: https://github.com/woct0rdho/SageAttention/releases)
# Format: sageattention-{VERSION}+{CUDA}torch{TORCH}.post{POST}-cp39-abi3-win_amd64.whl
# Examples:
# - For PyTorch 2.8.0+cu128: sageattention-2.2.0+cu128torch2.8.0.post3-cp39-abi3-win_amd64.whl
# - For PyTorch 2.9.0+cu128: sageattention-2.2.0+cu128torch2.9.0.post3-cp39-abi3-win_amd64.whl
# - For PyTorch 2.7.0+cu124: sageattention-2.2.0+cu124torch2.7.0.post3-cp39-abi3-win_amd64.whl
SAGEATTENTION_WHEEL_URL = "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post4/sageattention-2.2.0+cu128torch2.8.0andhigher.post4-cp39-abi3-win_amd64.whl"

# Custom nodes to clone - Required for WAN Animate 2.2
CUSTOM_NODES = (
    "https://github.com/ltdrdata/ComfyUI-Manager.git",
    "https://github.com/kijai/ComfyUI-WanVideoWrapper",
    "https://github.com/rgthree/rgthree-comfy",
    "https://github.com/kijai/ComfyUI-KJNodes",
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
    "https://github.com/kijai/ComfyUI-segment-anything-2",
    "https://github.com/9nate-drake/Comfyui-SecNodes",
    "https://github.com/kijai/ComfyUI-WanAnimatePreprocess",
)

# Models to download from HuggingFace
# Uncomment the line "MODELS = {}" at the end of the 'MODELS' list to disable
MODELS = {
    "clip_vision": ["clip_vision_h.safetensors"],
    "detection": [
        "vitpose_h_wholebody_data.bin",
        "vitpose_h_wholebody_model.onnx",
        "vitpose-l-wholebody.onnx",
        "yolov10m.onnx"
    ],
    "diffusion_models": ["Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors"],
    "loras": [
        "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        "WanAnimate_relight_lora_fp16.safetensors"
    ],
    "sams": ["SeC-4B-fp16.safetensors"],
    "text_encoders": ["umt5-xxl-enc-bf16.safetensors"],
    "vae": ["Wan2_1_VAE_bf16.safetensors"]
}

# MODELS = {} # Uncomment to disable model downloads

# Minimum Requirements
MIN_PYTHON_VERSION = (3, 9)
MIN_DISK_SPACE_GB = 60

# ComfyUI Server Settings
# IP address to bind to (127.0.0.1 for localhost only, 0.0.0.0 for all interfaces)
COMFYUI_LOCAL_IP = "localhost"
COMFYUI_LOCAL_PORT = 8188

# Installation Configuration Constants
MAX_DOWNLOAD_ATTEMPTS = 3
INTERNET_CHECK_RETRIES = 3
INTERNET_CHECK_DELAY = 2
PROGRESS_LOG_INTERVAL = 5
MAX_LOGGED_OUTPUT_LINES = 50

# ============================================================================
# CONFIGURATION DATACLASS (Internal - uses above constants as defaults)
# ============================================================================

@dataclass
class Config:
    """Internal configuration object - built from constants above and CLI args"""
    
    # User-customizable settings (from constants above)
    python_version: Optional[str] = PYTHON_VERSION
    comfyui_version: str = COMFYUI_VERSION
    install_path: Optional[Path] = INSTALL_PATH
    venv_name: str = VENV_NAME
    pytorch_version: str = PYTORCH_VERSION
    huggingface_base: str = HUGGINGFACE_BASE
    sageattention_wheel_url: str = SAGEATTENTION_WHEEL_URL
    custom_nodes: Tuple[str, ...] = CUSTOM_NODES
    models: Dict[str, List[str]] = field(default_factory=lambda: MODELS.copy())
    min_python_version: Tuple[int, int] = MIN_PYTHON_VERSION
    min_disk_space_gb: int = MIN_DISK_SPACE_GB
    COMFYUI_LOCAL_IP: str = COMFYUI_LOCAL_IP
    COMFYUI_LOCAL_PORT: int = COMFYUI_LOCAL_PORT
    max_download_attempts: int = MAX_DOWNLOAD_ATTEMPTS
    internet_check_retries: int = INTERNET_CHECK_RETRIES
    internet_check_delay: int = INTERNET_CHECK_DELAY
    progress_log_interval: int = PROGRESS_LOG_INTERVAL
    max_logged_output_lines: int = MAX_LOGGED_OUTPUT_LINES

    # Triton formats it's downloads as "triton-windows<3.6" for version 3.5, so we add 0.1 to the version to get the correct version number, assuming the user set TRITON_VERSION = "3.5"
    triton_version: str = "triton-windows<" + str(float(TRITON_VERSION) + 0.1)
    
    # CLI Flags (set from command line arguments)
    no_cache: bool = False
    clear_venv: bool = False
    reinstall_venv: bool = False
    upgrade_venv: bool = False
    upgrade_deps: bool = False
    skip_models: bool = False
    reinstall_comfyui: bool = False
    
    # Derived values (computed in __post_init__)
    python_executable: Optional[str] = field(init=False, default=None)
    install_dir: Path = field(init=False)
    archive_url: str = field(init=False)
    pytorch_index_url: str = field(init=False)  # Always set (CUDA required)
    log_file: Path = field(init=False)
    
    def __post_init__(self):
        """Compute derived values after initialization"""
        # Validate CUDA requirement early (fail fast)
        if "+cu" not in self.pytorch_version:
            raise ValueError(
                f"WAN Animate 2.2 requires CUDA-enabled PyTorch.\n"
                f"  Current version: {self.pytorch_version}\n"
                f"  Please use a CUDA version (e.g., 2.8.0+cu128)"
            )
        
        self.python_executable = find_python_executable(self.python_version)
        
        if self.install_path is None or str(self.install_path).lower() in ["auto", "default", ""]:
            self.install_dir = Path.cwd() / f"ComfyUI-{self.comfyui_version.lstrip('v')}"
        else:
            self.install_dir = Path(self.install_path).resolve()
        
        self.archive_url = f"https://github.com/comfyanonymous/ComfyUI/archive/refs/tags/{self.comfyui_version}.zip"
        
        # Extract CUDA version and build PyTorch index URL (always present after validation above)
        cuda_version = self.pytorch_version.split("+cu")[1]
        self.pytorch_index_url = f"https://download.pytorch.org/whl/cu{cuda_version}"
        
        self.log_file = (self.install_dir.parent / "comfyui_installation.log").resolve()
        
        # Validate SageAttention wheel URL matches PyTorch version
        self._validate_sageattention_compatibility()
    
    def _validate_sageattention_compatibility(self):
        """Warn if SageAttention wheel URL doesn't match PyTorch version"""
        if not self.sageattention_wheel_url:
            return
        
        # Extract CUDA and PyTorch version from PYTORCH_VERSION (e.g., "2.8.0+cu128")
        # CUDA is already validated in __post_init__, so we can safely split
        torch_ver = self.pytorch_version.split("+")[0]  # "2.8.0"
        cuda_suffix = self.pytorch_version.split("+")[1]  # "cu128"
        
        # Extract major.minor from torch version (e.g., "2.8.0" -> "2.8")
        torch_major_minor = ".".join(torch_ver.split(".")[:2])
        
        # Check if URL contains expected patterns
        expected_cuda = f"+{cuda_suffix}torch"
        expected_torch = f"torch{torch_major_minor}."
        
        url_lower = self.sageattention_wheel_url.lower()
        if expected_cuda.lower() not in url_lower or expected_torch.lower() not in url_lower:
            import warnings
            warnings.warn(
                f"\n{'='*70}\n"
                f"WARNING: SageAttention wheel may not match PyTorch version!\n"
                f"  PyTorch version: {self.pytorch_version}\n"
                f"  Expected in URL: {expected_cuda}{torch_major_minor}\n"
                f"  Current URL: {self.sageattention_wheel_url}\n"
                f"  Check: https://github.com/woct0rdho/SageAttention/releases\n"
                f"{'='*70}",
                UserWarning,
                stacklevel=2
            )

# ============================================================================
# MODULE-LEVEL UTILITY FUNCTIONS
# ============================================================================

def find_python_executable(version_string: Optional[str]) -> Optional[str]:
    """
    Robustly find a Python executable for a requested version, Windows-first.
    On Windows, probe `py -X.Y[.Z]` to get the exact Python path.
    """
    if version_string is None:
        return sys.executable
    
    version_parts = version_string.split(".")
    is_windows = (platform.system() == "Windows")
    
    if is_windows:
        def probe_py(tag: str) -> Optional[str]:
            try:
                out = subprocess.run(
                    ["py", f"-{tag}", "-c", "import sys;print(sys.executable)"],
                    capture_output=True, text=True, timeout=5
                )
                exe = out.stdout.strip()
                return exe if out.returncode == 0 and exe else None
            except Exception:
                return None
        
        # Most specific first
        if len(version_parts) >= 3:
            exe = probe_py(".".join(version_parts[:3]))
            if exe:
                return exe
        if len(version_parts) >= 2:
            exe = probe_py(".".join(version_parts[:2]))
            if exe:
                return exe
        if len(version_parts) >= 1:
            exe = probe_py(version_parts[0])
            if exe:
                return exe
    
    # Fallback PATH-based names
    candidates = []
    if len(version_parts) >= 3:
        candidates.append(f"python{version_parts[0]}.{version_parts[1]}.{version_parts[2]}")
    if len(version_parts) >= 2:
        candidates.append(f"python{version_parts[0]}.{version_parts[1]}")
    if len(version_parts) >= 1:
        candidates.append(f"python{version_parts[0]}")
    
    for cmd in candidates:
        exe_path = shutil.which(cmd)
        if exe_path:
            return exe_path
    
    return None

# ============================================================================
# LOGGING FORMATTERS
# ============================================================================

class ConsoleFormatter(logging.Formatter):
    """Formatter for console output (no timestamp)"""
    def format(self, record):
        return record.getMessage()

class FileFormatter(logging.Formatter):
    """Formatter for file output (with timestamp and level)"""
    def format(self, record):
        return f"{self.formatTime(record, '%Y-%m-%d %H:%M:%S')} [{record.levelname}] {record.getMessage()}"

# ============================================================================
# ERROR HANDLER CLASS
# ============================================================================

class ErrorHandler:
    """Centralized error handling with retry logic, cleanup, and recovery suggestions"""
    
    def __init__(self, logger):
        self.logger = logger
        self.cleanup_tasks = []
    
    @contextmanager  
    def network_operation(self, operation_name, max_retries=3, delay=2, backoff=2):
        """Network operations with automatic retry and exponential backoff"""
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                yield attempt
                return  # Success, exit context manager
            except (urllib.error.URLError, urllib.error.HTTPError, 
                    ConnectionError, TimeoutError) as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = delay * (backoff ** (attempt - 1))
                    self.logger.warning(f"{operation_name} - Attempt {attempt}/{max_retries} failed: {e}")
                    self.logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    self.handle_network_error(e, operation_name)
                    raise last_error
    
    @contextmanager
    def file_operation(self, operation_name, cleanup_on_fail=True):
        """Context manager for file operations"""
        try:
            yield
        except (PermissionError, FileNotFoundError, OSError, zipfile.BadZipFile) as e:
            self.handle_filesystem_error(e, operation_name)
            if cleanup_on_fail:
                self.cleanup_all()
            raise
    
    def handle_network_error(self, error, context=None):
        """Categorize and log network errors with helpful messages"""
        if isinstance(error, urllib.error.HTTPError):
            self.logger.error(f"HTTP {error.code}: {error.reason}")
            if error.code == 404:
                self.logger.error("File not found. Check URL or version number.")
                self.suggest_fix('http_404', context)
            elif error.code == 403:
                self.logger.error("Access denied. May require authentication.")
                self.suggest_fix('http_403', context)
            elif error.code >= 500:
                self.logger.error("Server error. Try again later.")
                self.suggest_fix('http_5xx', context)
        elif isinstance(error, urllib.error.URLError):
            self.logger.error(f"Network error: {error.reason}")
            self.suggest_fix('network', context)
        elif isinstance(error, TimeoutError):
            self.logger.error("Connection timeout. Check internet connection.")
            self.suggest_fix('timeout', context)
        else:
            self.logger.error(f"Network operation failed: {error}")
    
    def handle_filesystem_error(self, error, context=None):
        """Categorize filesystem errors"""
        if isinstance(error, PermissionError):
            self.logger.error(f"Permission denied: {context}")
            self.suggest_fix('permissions', context)
        elif isinstance(error, FileNotFoundError):
            self.logger.error(f"File not found: {context}")
            self.suggest_fix('file_not_found', context)
        elif isinstance(error, OSError):
            if hasattr(error, 'errno') and error.errno == 28:
                self.logger.error("No space left on device")
                self.suggest_fix('disk_space', context)
            else:
                self.logger.error(f"Filesystem error: {error}")
        elif isinstance(error, zipfile.BadZipFile):
            self.logger.error(f"Corrupted archive file: {context}")
            self.suggest_fix('bad_zip', context)
        else:
            self.logger.error(f"File operation failed: {error}")
    
    def register_cleanup(self, cleanup_func, description):
        """Register a cleanup task to run on failure"""
        self.cleanup_tasks.append((cleanup_func, description))
    
    def cleanup_all(self):
        """Execute all registered cleanup tasks"""
        if not self.cleanup_tasks:
            return
        
        self.logger.info("Cleaning up partial operations...")
        for cleanup_func, description in reversed(self.cleanup_tasks):
            try:
                self.logger.debug(f"Cleanup: {description}")
                cleanup_func()
            except Exception as e:
                self.logger.warning(f"Cleanup failed for {description}: {e}")
        self.cleanup_tasks.clear()
    
    def cleanup_path(self, filepath, description="path"):
        """Remove a file or directory"""
        try:
            path = Path(filepath)
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink()
                self.logger.debug(f"Removed {description}: {filepath}")
        except Exception as e:
            self.logger.warning(f"Failed to remove {filepath}: {e}")
    
    # Convenience aliases for specific use cases
    def cleanup_partial_download(self, filepath):
        """Remove a partially downloaded file"""
        self.cleanup_path(filepath, "partial download")
    
    def cleanup_partial_extraction(self, directory):
        """Remove a partially extracted directory"""
        self.cleanup_path(directory, "partial extraction")
    
    def cleanup_failed_clone(self, directory):
        """Remove a partially cloned git repository"""
        self.cleanup_path(directory, "partial clone")
    
    def suggest_fix(self, error_type, context=None):
        """Provide recovery suggestions based on error type"""
        suggestions = {
            'network': [
                "Check your internet connection",
                "Verify firewall/antivirus is not blocking connections",
                "Try using a VPN if behind a restrictive firewall"
            ],
            'http_404': [
                "Verify the version number in configuration",
                "Check if the resource has been moved or renamed",
                f"Visit the URL manually to confirm: {context}" if context else None
            ],
            'http_403': [
                "Check if authentication is required",
                "Verify API keys or tokens if applicable"
            ],
            'http_5xx': [
                "The server is experiencing issues",
                "Wait a few minutes and try again",
                "Check service status page if available"
            ],
            'timeout': [
                "Your connection may be slow or unstable",
                "Try increasing timeout values",
                "Check if VPN is slowing connection"
            ],
            'permissions': [
                "Run the script from a directory where you have write access",
                "On Windows, try running as Administrator",
                f"Check permissions on: {context}" if context else None
            ],
            'file_not_found': [
                "Verify the file path is correct",
                "Ensure previous steps completed successfully"
            ],
            'disk_space': [
                "Free up disk space (60GB+ recommended)",
                "Delete unnecessary files or move to larger drive",
                "Check disk usage with system tools"
            ],
            'bad_zip': [
                "The downloaded file may be corrupted",
                "Check your internet connection stability"
            ]
        }
        
        fixes = suggestions.get(error_type, [])
        if fixes:
            self.logger.info("Suggested fixes:")
            for i, fix in enumerate(fixes, 1):
                if fix:  # Skip None entries
                    self.logger.info(f"  {i}. {fix}")


# ============================================================================
# PREREQUISITE CHECKER CLASS
# ============================================================================

class PrerequisiteChecker:
    """Handles all prerequisite validation checks"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.cfg = config
        self.logger = logger
    
    def check_windows_only(self) -> bool:
        """Ensure we are on Windows 10/11, 64-bit."""
        system = platform.system()
        if system != "Windows":
            self.logger.critical("[FAIL] This installer supports Windows only (Windows 10/11).")
            return False

        arch = platform.machine().lower()
        if arch not in ("amd64", "x86_64"):
            self.logger.critical(f"[FAIL] Unsupported architecture: {arch}. Only AMD64/x86_64 is supported.")
            return False

        rel = platform.release()
        if rel not in ("10", "11"):
            self.logger.warning(f"[WARN] Windows release reported as '{rel}'. Continuing, but this is tested on Windows 10/11.")

        self.logger.info("[OK] Windows platform validated")
        return True
    
    def check_python_version(self) -> bool:
        """Check if Python version meets minimum requirements"""
        if self.cfg.python_executable is None:
            if self.cfg.python_version:
                self.logger.error(f"[FAIL] Python {self.cfg.python_version} not found in PATH")
                self.logger.error(f"       Please install Python {self.cfg.python_version} or set python_version = None")
            else:
                self.logger.error("[FAIL] Python executable not found")
            return False
        
        try:
            result = subprocess.run(
                [self.cfg.python_executable, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            python_version_str = result.stdout.strip()
            self.logger.info(f"Python executable: {self.cfg.python_executable}")
            self.logger.info(f"Python version: {python_version_str}")
            
            # Parse version for minimum check
            import re
            match = re.search(r'(\d+)\.(\d+)\.(\d+)', python_version_str)
            if match:
                version_tuple = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
                if version_tuple[:2] < self.cfg.min_python_version:
                    self.logger.error(f"[FAIL] Python {self.cfg.min_python_version[0]}.{self.cfg.min_python_version[1]} or higher required")
                    self.logger.error(f"       Found version: {version_tuple[0]}.{version_tuple[1]}.{version_tuple[2]}")
                    self.logger.error("       Download from: https://www.python.org/downloads/")
                    return False
                else:
                    self.logger.info("[OK] Python version compatible")
            else:
                self.logger.warning("[WARNING] Could not parse Python version")
        except Exception as e:
            self.logger.error(f"[FAIL] Could not verify Python version: {e}")
            return False
        
        return True
    
    def check_git_installation(self) -> bool:
        """Check if Git is installed and accessible"""
        try:
            result = subprocess.run(['git', '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                git_version = result.stdout.strip()
                self.logger.info(f"[OK] Git found: {git_version}")
                return True
            else:
                self.logger.error("[FAIL] Git command failed")
                return False
        except FileNotFoundError:
            self.logger.error("[FAIL] Git not found")
            self.logger.error("       Install from: https://git-scm.com/download/win")
            return False
        except subprocess.TimeoutExpired:
            self.logger.error("[FAIL] Git command timed out")
            return False
    
    def check_required_modules(self) -> bool:
        """Check if required standard library modules are available"""
        required_modules = [
            ('venv', 'venv'),
            ('urllib.request', 'urllib'),
            ('zipfile', 'zipfile'),
            ('subprocess', 'subprocess'),
            ('pathlib', 'pathlib'),
            ('signal', 'signal'),
            ('logging', 'logging'),
        ]
        
        all_present = True
        for module_path, display_name in required_modules:
            try:
                __import__(module_path)
                self.logger.debug(f"[OK] {display_name} module available")
            except ImportError:
                self.logger.error(f"[FAIL] {display_name} module not available")
                all_present = False
        
        return all_present
    
    def check_internet_connectivity(self) -> bool:
        """Check internet connectivity with retries"""
        self.logger.info("Testing internet connectivity...")
        max_retries = self.cfg.internet_check_retries
        retry_delay = self.cfg.internet_check_delay
        
        def animate_dots(attempt, max_retries, stop_event):
            """Display animated dots while checking connection"""
            sys.stdout.write(f"\r[{attempt}/{max_retries}] ")
            sys.stdout.flush()
            while not stop_event.is_set():
                sys.stdout.write(".")
                sys.stdout.flush()
                time.sleep(1)
            sys.stdout.write("\n")
            sys.stdout.flush()
        
        for attempt in range(1, max_retries + 1):
            stop_animation = threading.Event()
            animation_thread = threading.Thread(target=animate_dots, args=(attempt, max_retries, stop_animation))
            animation_thread.daemon = True
            animation_thread.start()
            
            try:
                with urllib.request.urlopen('https://clients3.google.com/generate_204', timeout=5) as _:
                    pass  # HTTP 204 No Content expected
                stop_animation.set()
                animation_thread.join(timeout=0.5)
                self.logger.info("[OK] Internet connection active")
                return True
            except Exception as e:
                stop_animation.set()
                animation_thread.join(timeout=0.5)
                if attempt < max_retries:
                    self.logger.warning(f"Attempt {attempt} failed: {e}")
                    self.logger.info(f"Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"[FAIL] Internet connectivity issue after {max_retries} attempts: {e}")
                    self.logger.error("       This script requires internet to download packages")
                    return False
        
        return False
    
    def check_disk_space(self) -> bool:
        """Check available disk space"""
        try:
            stats = shutil.disk_usage(Path.cwd())
            free_gb = stats.free / (1024**3)
            self.logger.info(f"[OK] Available disk space: {free_gb:.1f} GB")
            
            if free_gb < self.cfg.min_disk_space_gb:
                self.logger.error(f"[FAIL] Insufficient disk space: {free_gb:.1f} GB available, {self.cfg.min_disk_space_gb} GB required")
                return False
            
            return True
        except Exception as e:
            self.logger.warning(f"[WARNING] Could not check disk space: {e}")
            return True  # Don't fail on disk space check error
    
    def check_write_permissions(self) -> bool:
        """Check write permissions in current directory"""
        try:
            test_file = Path.cwd() / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            self.logger.info("[OK] Write permissions verified")
            return True
        except Exception as e:
            self.logger.error(f"[FAIL] No write permission in current directory: {e}")
            return False
    
    def check_all(self) -> bool:
        """Check all prerequisites before starting installation"""
        self.logger.info("="*70)
        self.logger.info("Checking Prerequisites")
        self.logger.info("="*70)
        
        # Run all checks
        checks = [
            ("Windows platform", self.check_windows_only),
            ("Python version", self.check_python_version),
            ("Git installation", self.check_git_installation),
            ("required modules", self.check_required_modules),
            ("internet connectivity", self.check_internet_connectivity),
            ("disk space", self.check_disk_space),
            ("write permissions", self.check_write_permissions),
        ]
        
        all_good = True
        for check_name, check_func in checks:
            if not check_func():
                all_good = False
        
        self.logger.info("="*70)
        
        if all_good:
            self.logger.info("All prerequisites met! Proceeding with installation...")
            return True
        else:
            self.logger.error("Prerequisites check failed! Please fix the issues above.")
            return False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_step(message: str, logger: logging.Logger) -> None:
    """Print a formatted step message"""
    logger.info("="*70)
    logger.info(f"STEP: {message}")
    logger.info("="*70)

def download_file(url: str, destination: str, description: str, logger: logging.Logger, 
                  error_handler: ErrorHandler, max_attempts: int = 3, 
                  progress_interval: int = 5) -> bool:
    """Download a file with progress indication, retry logic, and cleanup"""
    logger.info(f"Downloading {description}...")
    logger.debug(f"URL: {url}")
    logger.debug(f"Destination: {destination}")
    
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(1, max_attempts + 1):
        last_percent = 0
        
        def report_progress(block_num, block_size, total_size):
            nonlocal last_percent
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, (downloaded / total_size) * 100)
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                
                # Log progress every X%
                if int(percent / progress_interval) > int(last_percent / progress_interval):
                    logger.debug(f"Download progress: {percent:.0f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
                    last_percent = percent
                
                # Print real-time progress (not logged)
                print(f"\rProgress: {percent:.1f}% ({downloaded_mb:.1f} MB / {total_mb:.1f} MB)", end='', flush=True)
        
        # Register cleanup for partial downloads
        error_handler.register_cleanup(
            lambda dest=destination: error_handler.cleanup_partial_download(dest),
            f"Remove partial download: {destination.name}"
        )
        
        try:
            with error_handler.network_operation(f"Downloading {description}", max_retries=1) as _:
                urllib.request.urlretrieve(url, destination, reporthook=report_progress)
                print()  # Newline after progress
            logger.info(f"Download complete: {description}")
            # Clear cleanup tasks since download was successful
            error_handler.cleanup_tasks.clear()
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError):
            print()  # Newline after progress
            error_handler.cleanup_partial_download(destination)
            
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                logger.warning(f"Download attempt {attempt}/{max_attempts} failed")
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Download failed after {max_attempts} attempts")
                return False
        except Exception as e:
            print()  # Newline after progress
            logger.error(f"Unexpected error downloading {description}: {e}")
            error_handler.cleanup_partial_download(destination)
            return False
    
    return False

def run_command(cmd, cwd=None, description="command", show_output=True, logger=None, interrupted_check=None, max_log_lines=50):
    """Run a shell command and show live output with interrupt handling and logging"""
    if logger is None:
        logger = logging.getLogger(__name__)
    if interrupted_check is None:
        def interrupted_check():
            return False
    
    logger.info(f"Executing: {description}")
    logger.debug(f"Command: {cmd}")
    
    process = None
    output_lines = []
    last_output_time = [time.time()]  # Use list for mutable reference in nested function
    heartbeat_active = [True]
    dots_printed = [False]  # Track if dots were printed
    
    def heartbeat_thread():
        """Print activity indicator during silent periods"""
        while heartbeat_active[0]:
            time.sleep(1)  # Check every second
            if heartbeat_active[0]:
                elapsed = time.time() - last_output_time[0]
                if elapsed >= 1 and show_output:  # No output for 1+ seconds
                    print(".", end='', flush=True)
                    dots_printed[0] = True
    
    try:
        # Start heartbeat thread for activity indication
        heartbeat = threading.Thread(target=heartbeat_thread, daemon=True)
        heartbeat.start()
        
        # Use Popen for live output streaming with unbuffered output
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=0  # Unbuffered for immediate output
        )
        
        # Stream output line by line with immediate flush
        for line in iter(process.stdout.readline, ''):
            if interrupted_check():
                heartbeat_active[0] = False
                process.terminate()
                process.wait(timeout=5)
                logger.warning(f"Command interrupted: {description}")
                return False
            if line:
                last_output_time[0] = time.time()  # Update last output time
                stripped_line = line.rstrip()
                output_lines.append(stripped_line)
                if show_output:
                    # If dots were printed, add newline before actual output
                    if dots_printed[0]:
                        print()  # Newline to separate dots from output
                        dots_printed[0] = False
                    # Print to console (live output, not logged to avoid duplication)
                    print(line, end='', flush=True)
                    
        process.wait()
        heartbeat_active[0] = False  # Stop heartbeat
        
        # Log summary to file only (not console to avoid spam)
        if output_lines:
            logger.debug(f"Command output ({len(output_lines)} lines):")
            for log_line in output_lines[:max_log_lines]:  # Only log first N lines to avoid huge logs
                logger.debug(f"  {log_line}")
            if len(output_lines) > max_log_lines:
                logger.debug(f"  ... ({len(output_lines) - max_log_lines} more lines omitted)")
        
        if process.returncode == 0:
            logger.info(f"Command succeeded: {description}")
        else:
            logger.error(f"Command failed with code {process.returncode}: {description}")
        
        return process.returncode == 0
        
    except KeyboardInterrupt:
        heartbeat_active[0] = False  # Stop heartbeat
        logger.warning(f"Keyboard interrupt during: {description}")
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
        
    except Exception as e:
        heartbeat_active[0] = False  # Stop heartbeat
        logger.error(f"Exception running {description}: {e}")
        if process:
            process.terminate()
        return False

# ============================================================================
# COMFYUI INSTALLER CLASS
# ============================================================================

class ComfyUIInstaller:
    """Main installer class with all installation steps as methods"""
    
    def __init__(self, config: Config):
        self.cfg = config
        self.logger = self._setup_logger()
        self.error_handler = ErrorHandler(self.logger)
        self.interrupted = False
        self._setup_signal_handler()
        
        # Background download infrastructure
        self.background_download_thread: Optional[threading.Thread] = None
        self.download_lock = threading.Lock()
        self.download_complete = threading.Event()
        self.download_results = {
            'downloaded': 0,
            'skipped': 0,
            'failed': 0,
            'total_bytes': 0,
            'failed_files': {},
            'start_time': 0,
            'end_time': 0
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with file and console handlers"""
        logger = logging.getLogger("comfyui_installer")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.cfg.log_file, mode='w', encoding='utf-8')
        file_handler.setFormatter(FileFormatter())
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ConsoleFormatter())
        console_handler.setLevel(logging.INFO)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger
    
    def _setup_signal_handler(self):
        """Setup Ctrl-C signal handler"""
        def handle_interrupt(signum, frame):
            self.interrupted = True
            self.logger.warning("="*70)
            self.logger.warning("INSTALLATION INTERRUPTED BY USER")
            self.logger.warning("="*70)
            self.logger.info(f"Log saved to: {self.cfg.log_file.absolute()}")
            raise KeyboardInterrupt()
        
        signal.signal(signal.SIGINT, handle_interrupt)
    
    def _print_step(self, message: str):
        """Print formatted step message"""
        print_step(message, self.logger)
    
    def _pip(self, python_exe: Path, args: str, desc: str) -> bool:
        """Run pip with consistent parameters"""
        cmd = f'"{python_exe}" -m pip {args}'
        return run_command(cmd, description=desc, logger=self.logger,
                         interrupted_check=lambda: self.interrupted,
                         max_log_lines=self.cfg.max_logged_output_lines)
    
    def _venv_python(self) -> Path:
        """Get path to venv python executable with existence check"""
        py = self.cfg.install_dir / self.cfg.venv_name / "Scripts" / "python.exe"
        if not py.exists():
            self.logger.error(f"Python executable not found in venv: {py}")
        return py
    
    def _has_nvidia_gpu(self) -> bool:
        """Check if NVIDIA GPU is available"""
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
            return out.returncode == 0 and "GPU" in (out.stdout or "")
        except Exception:
            return False
    
    def _create_model_directories(self) -> bool:
        """Create model directories (can be called early for parallel downloads)"""
        models_dir = self.cfg.install_dir / "models"
        total_dirs = len(self.cfg.models.keys())
        
        self.logger.info(f"Creating {total_dirs} model directories")
        self.logger.debug(f"Models directory: {models_dir}")
        
        try:
            for model_type in self.cfg.models.keys():
                dir_path = models_dir / model_type
                dir_path.mkdir(parents=True, exist_ok=True)
                file_count = len(self.cfg.models[model_type])
                self.logger.debug(f"  {model_type}/ ({file_count} files needed)")
            
            self.logger.info("Model directories created successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create model directories: {e}")
            return False
    
    def _background_download_models(self):
        """Background thread method to download models silently (file logging only)"""
        try:
            with self.download_lock:
                self.download_results['start_time'] = time.time()
            
            self.logger.info("[Background] Starting model downloads")
            self.logger.debug(f"[Background] Source: {self.cfg.huggingface_base}")
            
            models_dir = self.cfg.install_dir / "models"
            total_files = sum(len(files) for files in self.cfg.models.values())
            downloaded = skipped = failed = 0
            failed_files = {}
            total_bytes_downloaded = 0
            max_file_attempts = 3
            
            for model_type, file_list in self.cfg.models.items():
                if self.interrupted:
                    self.logger.warning("[Background] Download interrupted by user")
                    break
                    
                target_dir = models_dir / model_type
                
                for filename in file_list:
                    if self.interrupted:
                        break
                        
                    file_key = f"{model_type}/{filename}"
                    target_file = target_dir / filename
                    
                    self.logger.debug(f"[Background] [{downloaded + skipped + failed + 1}/{total_files}] {file_key}")
                    
                    # Check if file already exists and is valid
                    try:
                        if target_file.exists() and target_file.stat().st_size > 1024:
                            file_size_mb = target_file.stat().st_size / (1024 * 1024)
                            self.logger.debug(f"[Background]   Already exists ({file_size_mb:.1f} MB) - skipping")
                            skipped += 1
                            with self.download_lock:
                                self.download_results['skipped'] = skipped
                            continue
                    except Exception:
                        pass
                    
                    # Construct download URL
                    url = f"{self.cfg.huggingface_base}/{filename}"
                    
                    # Download without progress hook (silent)
                    for attempt in range(1, max_file_attempts + 1):
                        try:
                            old_timeout = socket.getdefaulttimeout()
                            socket.setdefaulttimeout(300)
                            
                            try:
                                urllib.request.urlretrieve(url, target_file)
                            finally:
                                socket.setdefaulttimeout(old_timeout)
                            
                            # Validate downloaded file
                            if target_file.exists() and target_file.stat().st_size > 1024:
                                file_size_mb = target_file.stat().st_size / (1024 * 1024)
                                bytes_down = target_file.stat().st_size
                                self.logger.debug(f"[Background]   Downloaded successfully ({file_size_mb:.1f} MB)")
                                downloaded += 1
                                total_bytes_downloaded += bytes_down
                                with self.download_lock:
                                    self.download_results['downloaded'] = downloaded
                                    self.download_results['total_bytes'] = total_bytes_downloaded
                                break
                            else:
                                raise ValueError("Downloaded file is invalid or too small")
                                
                        except Exception as e:
                            self.logger.debug(f"[Background]   Download attempt {attempt} failed: {type(e).__name__}")
                            if target_file.exists():
                                try:
                                    target_file.unlink()
                                except Exception:
                                    pass
                            
                            if attempt >= max_file_attempts:
                                self.logger.warning(f"[Background]   Failed after {max_file_attempts} attempts: {file_key}")
                                failed += 1
                                failed_files[file_key] = max_file_attempts
                                with self.download_lock:
                                    self.download_results['failed'] = failed
                                    self.download_results['failed_files'] = failed_files.copy()
                            else:
                                time.sleep(5 * (2 ** (attempt - 1)))
            
            with self.download_lock:
                self.download_results['end_time'] = time.time()
            
            self.logger.info(f"[Background] Model downloads complete: {downloaded} downloaded, {skipped} skipped, {failed} failed")
            
        except Exception as e:
            self.logger.error(f"[Background] Unexpected error in background download: {e}")
            import traceback
            self.logger.debug(f"[Background] Traceback: {traceback.format_exc()}")
        finally:
            self.download_complete.set()
    
    def step_1_download_comfyui(self) -> bool:
        """Download ComfyUI source code"""
        self._print_step("Step 1: Download ComfyUI Source Code (Standard ZIP)")
        
        archive_path = Path(f"ComfyUI-{self.cfg.comfyui_version}.zip")
        
        # Force re-download if --reinstall-comfyui flag is set
        if self.cfg.reinstall_comfyui and archive_path.exists():
            self.logger.info("--reinstall-comfyui flag detected: re-downloading ComfyUI ZIP")
            try:
                archive_path.unlink()
                self.logger.debug(f"Removed existing archive: {archive_path}")
            except Exception as e:
                self.logger.warning(f"Could not remove existing archive: {e}")
        
        if archive_path.exists():
            self.logger.info(f"Archive already exists at {archive_path}")
            return True
        
        self.logger.info("Using SOURCE CODE version - standard .zip compression")
        return download_file(self.cfg.archive_url, str(archive_path), "ComfyUI source code", 
                           self.logger, self.error_handler, self.cfg.max_download_attempts, 
                           self.cfg.progress_log_interval)

    def step_2_extract_comfyui(self) -> bool:
        """Extract ComfyUI using Python's built-in zipfile"""
        self._print_step("Step 2: Extract ComfyUI (Pure Python)")
        
        archive_path = Path(f"ComfyUI-{self.cfg.comfyui_version}.zip")
        
        if not archive_path.exists():
            self.logger.error("Archive not found. Please run step 1 first.")
            return False
        
        # Check if ComfyUI is already extracted (verify key files exist)
        main_py = self.cfg.install_dir / "main.py"
        requirements_txt = self.cfg.install_dir / "requirements.txt"
        
        # Skip extraction if files exist and not forcing reinstall
        if not self.cfg.reinstall_comfyui and self.cfg.install_dir.exists() and main_py.exists() and requirements_txt.exists():
            self.logger.info(f"ComfyUI already extracted at {self.cfg.install_dir} - skipping")
            
            # Start background downloads even if extraction was skipped
            if not self.cfg.skip_models and self.background_download_thread is None:
                self.logger.info("Creating model directories for parallel downloads...")
                if self._create_model_directories():
                    self.logger.info("Starting model downloads in background (parallel with installation)...")
                    self.background_download_thread = threading.Thread(
                        target=self._background_download_models,
                        daemon=True,
                        name="ModelDownloadThread"
                    )
                    self.background_download_thread.start()
            
            return True
        
        # Log reinstall message if forcing re-extraction
        if self.cfg.reinstall_comfyui:
            self.logger.info("--reinstall-comfyui flag detected: extracting and overwriting existing files")
            self.logger.info("Note: venv, models, and custom_nodes will not be affected")
        
        self.logger.info(f"Extracting {archive_path}")
        self.logger.debug(f"Destination: {self.cfg.install_dir.parent}")
        
        # Register cleanup for partial extraction
        self.error_handler.register_cleanup(
            lambda: self.error_handler.cleanup_partial_extraction(self.cfg.install_dir),
            f"Remove partial extraction: {self.cfg.install_dir.name}"
        )
        
        try:
            with self.error_handler.file_operation("Extracting ComfyUI", cleanup_on_fail=True):
                # Verify archive integrity first
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    # Test the archive
                    bad_file = zip_ref.testzip()
                    if bad_file:
                        raise zipfile.BadZipFile(f"Corrupted file in archive: {bad_file}")
                    
                    # Extract all files at once - much faster!
                    file_count = len(zip_ref.filelist)
                    self.logger.info(f"Extracting {file_count} files... (this may take a minute)")
                    zip_ref.extractall(self.cfg.install_dir.parent)
                
            self.logger.info("Extraction complete!")
            # Clear cleanup tasks since extraction was successful
            self.error_handler.cleanup_tasks.clear()
            
            # Create model directories early for background downloads
            if not self.cfg.skip_models:
                self.logger.info("Creating model directories for parallel downloads...")
                if self._create_model_directories():
                    # Start background model downloads
                    self.logger.info("Starting model downloads in background (parallel with installation)...")
                    self.background_download_thread = threading.Thread(
                        target=self._background_download_models,
                        daemon=True,
                        name="ModelDownloadThread"
                    )
                    self.background_download_thread.start()
                else:
                    self.logger.warning("Failed to create model directories - will try again at step 9")
            
            return True
        except (zipfile.BadZipFile, PermissionError, OSError):
            # Error already handled by error_handler.file_operation
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during extraction: {e}")
            self.error_handler.cleanup_partial_extraction(self.cfg.install_dir)
            return False

    def step_3_setup_python_env(self) -> bool:
        """Set up Python virtual environment"""
        self._print_step("Step 3: Setup Python Virtual Environment")
        
        venv_dir = self.cfg.install_dir / self.cfg.venv_name
        
        # If --clear-venv flag, just clear and exit (don't continue installation)
        if self.cfg.clear_venv:
            if not venv_dir.exists():
                self.logger.warning("Virtual environment does not exist, nothing to clear")
                return True
            self.logger.info(f"Clearing virtual environment at {venv_dir}")
            self.logger.info("Note: --clear-venv only clears the venv, does not reinstall packages")
            venv_cmd = [self.cfg.python_executable, "-m", "venv", "--clear", str(venv_dir)]
            try:
                subprocess.run(venv_cmd, check=True)
                self.logger.info("Virtual environment cleared successfully")
                self.logger.info("Stopping here (--clear-venv does not continue installation)")
                return False  # signals "stop pipeline gracefully"
            except Exception as e:
                self.logger.error(f"Failed to clear venv: {e}")
                return False
        
        if venv_dir.exists() and not self.cfg.reinstall_venv and not self.cfg.upgrade_venv and not self.cfg.upgrade_deps:
            self.logger.info("Virtual environment already exists - skipping")
            return True
        
        self.logger.info(f"Creating virtual environment at {venv_dir}")
        self.logger.debug(f"Using Python: {self.cfg.python_executable}")
        
        # Build venv command with flags
        venv_cmd = [self.cfg.python_executable, "-m", "venv"]
        
        if self.cfg.reinstall_venv:
            venv_cmd.append("--clear")
            self.logger.debug("Flag: --clear (delete existing environment and reinstall)")
        
        if self.cfg.upgrade_venv:
            venv_cmd.append("--upgrade")
            self.logger.debug("Flag: --upgrade (upgrade Python version)")
        
        if self.cfg.upgrade_deps:
            venv_cmd.append("--upgrade-deps")
            self.logger.debug("Flag: --upgrade-deps (upgrade pip/setuptools)")
        
        venv_cmd.append(str(venv_dir))
        
        try:
            subprocess.run(venv_cmd, check=True)
            self.logger.info("Virtual environment created successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create venv: {e}")
            return False

    def step_4_install_pytorch(self) -> bool:
        """Install PyTorch with CUDA support for WAN Animate 2.2"""
        self._print_step("Step 4: Install PyTorch with CUDA")
        
        python_exe = self._venv_python()
        if not python_exe.exists():
            return False
        
        # CUDA requirement already validated in Config.__post_init__
        if self.cfg.no_cache:
            self.logger.info(f"Installing PyTorch {self.cfg.pytorch_version} (NO CACHE - fresh download)...")
        else:
            self.logger.info(f"Installing PyTorch {self.cfg.pytorch_version} (large download ~3.5GB, please wait)...")
        
        # Check for NVIDIA GPU
        if not self._has_nvidia_gpu():
            self.logger.warning("WARNING: No NVIDIA GPU detected (nvidia-smi not found or no GPUs)")
            self.logger.warning("WAN Animate 2.2 requires an NVIDIA GPU with CUDA support")
            self.logger.warning("Installation will continue, but ComfyUI may not work without CUDA GPU")
        
        # Common pip flags
        common_flags = f'--progress-bar on --only-binary :all: {"--no-cache-dir" if self.cfg.no_cache else ""}'
        
        # Install CUDA version (torchvision/torchaudio versions auto-selected by pip from index URL)
        self.logger.info("Note: NVIDIA GPU required for WAN Animate 2.2 (minimum 8GB VRAM recommended)")
        
        ok = self._pip(
            python_exe,
            f'install --index-url {self.cfg.pytorch_index_url} '
            f'torch=={self.cfg.pytorch_version} '
            f'torchvision torchaudio {common_flags}',
            "Install PyTorch (CUDA)"
        )
        
        if not ok:
            self.logger.error("PyTorch installation failed")
            self.logger.error("Please check your internet connection and CUDA version compatibility")
            
        return ok
    
    def step_5_install_comfyui_deps(self) -> bool:
        """Install ComfyUI dependencies"""
        self._print_step("Step 5: Install ComfyUI Dependencies")
        
        python_exe = self._venv_python()
        if not python_exe.exists():
            return False
        
        requirements = self.cfg.install_dir / "requirements.txt"
        
        # First, upgrade pip toolchain
        self._pip(python_exe, 
                 f'install --upgrade pip setuptools wheel --progress-bar on {"--no-cache-dir" if self.cfg.no_cache else ""}',
                 "Upgrade pip/setuptools/wheel")
        
        if not requirements.exists():
            self.logger.warning("requirements.txt not found")
            return True
        
        self.logger.info(f"Installing ComfyUI dependencies from {requirements}")
        cmd = f'"{python_exe}" -m pip install -r "{requirements}" --progress-bar on {"--no-cache-dir" if self.cfg.no_cache else ""}'
        
        return run_command(cmd, description="Install ComfyUI dependencies", logger=self.logger,
                         interrupted_check=lambda: self.interrupted, 
                         max_log_lines=self.cfg.max_logged_output_lines)

    def step_6_clone_custom_nodes(self) -> bool:
        """Clone all required custom nodes with retry and cleanup"""
        self._print_step("Step 6: Clone Custom Nodes")
        
        # Check for Git LFS and initialize if available
        try:
            lfs = subprocess.run(["git", "lfs", "version"], capture_output=True, text=True, timeout=5)
            if lfs.returncode == 0:
                run_command("git lfs install --system", cwd=None, description="Enable Git LFS", 
                          logger=self.logger, interrupted_check=lambda: self.interrupted, 
                          max_log_lines=self.cfg.max_logged_output_lines)
            else:
                self.logger.info("Git LFS not found; continuing without it.")
        except Exception:
            self.logger.info("Git LFS not available; continuing without it.")
        
        custom_nodes_dir = self.cfg.install_dir / "custom_nodes"
        custom_nodes_dir.mkdir(parents=True, exist_ok=True)
        
        total_nodes = len(self.cfg.custom_nodes)
        self.logger.info(f"Cloning {total_nodes} custom nodes")
        self.logger.debug(f"Target directory: {custom_nodes_dir}")
        
        def clone_with_retry(repo_url, target_dir, repo_name):
            """Clone a repository with automatic retry"""
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    # Register cleanup for failed clone
                    self.error_handler.register_cleanup(
                        lambda tdir=target_dir: self.error_handler.cleanup_failed_clone(tdir),
                        f"Remove partial clone: {repo_name}"
                    )
                    
                    success = run_command(
                        f'git clone --depth 1 --quiet "{repo_url}"',
                        cwd=custom_nodes_dir,
                        description=f"Clone {repo_name}",
                        show_output=False,
                        logger=self.logger,
                        interrupted_check=lambda: self.interrupted,
                        max_log_lines=self.cfg.max_logged_output_lines
                    )
                    
                    if success:
                        # Clear cleanup tasks since clone was successful
                        self.error_handler.cleanup_tasks.clear()
                        return True
                    elif attempt < max_attempts:
                        self.logger.warning(f"  Clone attempt {attempt} failed, retrying...")
                        self.error_handler.cleanup_failed_clone(target_dir)
                        time.sleep(2 ** attempt)
                    else:
                        self.error_handler.cleanup_failed_clone(target_dir)
                        return False
                except Exception as e:
                    self.logger.error(f"  Error during clone: {e}")
                    if attempt < max_attempts:
                        self.logger.info(f"  Retrying in {2 ** attempt}s...")
                        self.error_handler.cleanup_failed_clone(target_dir)
                        time.sleep(2 ** attempt)
                    else:
                        self.error_handler.cleanup_failed_clone(target_dir)
                        return False
            return False
        
        for idx, repo_url in enumerate(self.cfg.custom_nodes, 1):
            repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
            target_dir = custom_nodes_dir / repo_name
            
            self.logger.info(f"[{idx}/{total_nodes}] {repo_name}")
            
            if target_dir.exists():
                self.logger.info("  Already exists - skipping")
                continue
            
            self.logger.debug(f"  Cloning from {repo_url}")
            
            if clone_with_retry(repo_url, target_dir, repo_name):
                self.logger.info("  Cloned successfully")
            else:
                self.logger.error("  Failed to clone after retries")
        
        self.logger.info("Completed cloning all custom nodes")
        return True

    def step_7_install_node_dependencies(self) -> bool:
        """Install Python dependencies for custom nodes"""
        self._print_step("Step 7: Install Custom Node Dependencies")
        
        python_exe = self._venv_python()
        if not python_exe.exists():
            return False
        
        custom_nodes_dir = self.cfg.install_dir / "custom_nodes"
        
        nodes_with_requirements = [
            "ComfyUI-Manager",
            "ComfyUI-WanVideoWrapper",
            "rgthree-comfy",
            "ComfyUI-KJNodes",
            "ComfyUI-VideoHelperSuite",
            "ComfyUI-segment-anything-2",
            "Comfyui-SecNodes",
            "ComfyUI-WanAnimatePreprocess"
        ]
        
        # Count nodes that have requirements
        nodes_to_install = []
        for node_name in nodes_with_requirements:
            req_file = custom_nodes_dir / node_name / "requirements.txt"
            if req_file.exists():
                nodes_to_install.append((node_name, req_file))
        
        total = len(nodes_to_install)
        self.logger.info(f"Found {total} nodes with requirements to install")
        
        for idx, (node_name, req_file) in enumerate(nodes_to_install, 1):
            self.logger.info(f"[{idx}/{total}] Installing dependencies for {node_name}")
            success = run_command(
                f'"{python_exe}" -m pip install -r "{req_file}" --progress-bar on {"--no-cache-dir" if self.cfg.no_cache else ""}',
                description=f"Install {node_name} dependencies",
                show_output=True,
                logger=self.logger,
                interrupted_check=lambda: self.interrupted,
                max_log_lines=self.cfg.max_logged_output_lines
            )
            
            if not success:
                self.logger.warning(f"  Some dependencies may have failed for {node_name}")
        
        self.logger.info("Completed installing dependencies for all custom nodes")
        return True
    
    def step_8_install_extras(self) -> bool:
        """Install extra dependencies"""
        self._print_step("Step 8: Install Extra Dependencies")
        
        # Windows-only check for Triton
        if platform.system() != "Windows":
            self.logger.error("Triton Windows is only available on Windows. Aborting extras step.")
            return False
        
        python_exe = self._venv_python()
        if not python_exe.exists():
            return False
        
        extras = [
            (self.cfg.triton_version, "Triton Windows"),
            ("hf_transfer", "HuggingFace Transfer"),
            (self.cfg.sageattention_wheel_url, "SageAttention"),
        ]
        
        total = len(extras)
        self.logger.info(f"Installing {total} extra packages")
        
        for idx, (package, name) in enumerate(extras, 1):
            self.logger.info(f"[{idx}/{total}] Installing {name}")
            self.logger.debug(f"  Package: {package}")
            success = run_command(
                f'"{python_exe}" -m pip install "{package}" --progress-bar on {"--no-cache-dir" if self.cfg.no_cache else ""}',
                description=f"Install {name}",
                show_output=True,
                logger=self.logger,
                interrupted_check=lambda: self.interrupted,
                max_log_lines=self.cfg.max_logged_output_lines
            )
            
            if success:
                self.logger.info(f"  {name} installed successfully")
            else:
                self.logger.warning(f"  {name} installation failed (may be optional)")
        
        self.logger.info("Completed installing extra packages")
        return True
    
    def step_9_create_models_dirs(self) -> bool:
        """Create model directories (or verify if already created in step 2)"""
        models_dir = self.cfg.install_dir / "models"
        
        # Check if directories already created (by step 2 for parallel downloads)
        if models_dir.exists() and self.background_download_thread is not None:
            self._print_step("Step 9: Verify Model Directories")
            self.logger.info("Model directories already created in step 2 (for parallel downloads)")
            return True
        
        # Otherwise, create them now
        self._print_step("Step 9: Create Model Directories")
        return self._create_model_directories()
    
    def _validate_existing_model_file(self, target_file: Path) -> str:
        """
        Check if a model file already exists and is valid.
        Returns: 'skip' or 'download'
        """
        if not target_file.exists():
            return 'download'
        
        try:
            file_size_mb = target_file.stat().st_size / (1024 * 1024)
            if target_file.stat().st_size > 1024:
                self.logger.info(f"  Already exists ({file_size_mb:.1f} MB) - skipping")
                return 'skip'
            else:
                self.logger.warning(f"  File exists but is too small ({file_size_mb:.3f} MB) - re-downloading")
                self.error_handler.cleanup_partial_download(target_file)
                return 'download'
        except Exception as e:
            self.logger.warning(f"  Error checking existing file: {e} - re-downloading")
            self.error_handler.cleanup_partial_download(target_file)
            return 'download'
    
    def _download_single_model_file(self, url: str, target_file: Path, file_key: str, max_attempts: int = 3) -> Tuple[bool, int]:
        """
        Download a single model file with retries.
        Returns: (success: bool, bytes_downloaded: int)
        """
        for attempt in range(1, max_attempts + 1):
            self.logger.debug(f"  Attempt {attempt}/{max_attempts}")
            
            self.error_handler.register_cleanup(
                lambda: self.error_handler.cleanup_partial_download(target_file),
                f"Cleanup partial download: {target_file.name}"
            )
            
            try:
                with self.error_handler.network_operation(f"Download {file_key}", max_retries=2, delay=3, backoff=2):
                    old_timeout = socket.getdefaulttimeout()
                    socket.setdefaulttimeout(300)
                    
                    def progress_hook(block_num, block_size, total_size):
                        if total_size <= 0:
                            return
                        downloaded_bytes = block_num * block_size
                        percent = min(100, (downloaded_bytes / total_size) * 100)
                        downloaded_mb = downloaded_bytes / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        print(f"\r  Progress: {percent:.1f}% ({downloaded_mb:.1f} MB/{total_mb:.1f} MB)", end='', flush=True)
                    
                    try:
                        urllib.request.urlretrieve(url, target_file, reporthook=progress_hook)
                        print()
                    finally:
                        socket.setdefaulttimeout(old_timeout)
                
                # Validate downloaded file
                if target_file.exists() and target_file.stat().st_size > 1024:
                    file_size_mb = target_file.stat().st_size / (1024 * 1024)
                    bytes_downloaded = target_file.stat().st_size
                    self.logger.info(f"  Downloaded successfully ({file_size_mb:.1f} MB)")
                    return (True, bytes_downloaded)
                else:
                    raise ValueError("Downloaded file is invalid or too small")
                    
            except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError, socket.timeout) as e:
                print()
                self.logger.warning(f"  Network error on attempt {attempt}: {type(e).__name__}")
                self.error_handler.cleanup_partial_download(target_file)
                self.error_handler.handle_network_error(e, file_key)
                if attempt < max_attempts:
                    wait_time = 5 * (2 ** (attempt - 1))
                    self.logger.info(f"  Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                print()
                self.logger.error(f"  Unexpected error on attempt {attempt}: {type(e).__name__}: {e}")
                self.error_handler.cleanup_partial_download(target_file)
                if attempt < max_attempts:
                    wait_time = 5 * (2 ** (attempt - 1))
                    self.logger.info(f"  Retrying in {wait_time}s...")
                    time.sleep(wait_time)
            
            finally:
                self.error_handler.cleanup_tasks.clear()
        
        return (False, 0)
    
    def _print_model_download_summary(self, downloaded: int, skipped: int, failed: int, 
                                       total_bytes: int, elapsed_time: float, 
                                       failed_files: Dict[str, int], max_attempts: int):
        """Print download statistics and failed file warnings"""
        elapsed_minutes = int(elapsed_time // 60)
        elapsed_seconds = int(elapsed_time % 60)
        total_gb = total_bytes / (1024**3)
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Model Download Summary:")
        self.logger.info(f"  Downloaded: {downloaded} files ({total_gb:.2f} GB)")
        self.logger.info(f"  Skipped (already exist): {skipped} files")
        self.logger.info(f"  Failed: {failed} files")
        self.logger.info(f"  Time elapsed: {elapsed_minutes}m {elapsed_seconds}s")
        if downloaded > 0 and elapsed_time > 0:
            avg_speed_mbps = (total_bytes / elapsed_time) / (1024 * 1024)
            self.logger.info(f"  Average speed: {avg_speed_mbps:.2f} MB/s")
        self.logger.info("="*70)
        
        if failed > 0:
            self.logger.warning("")
            self.logger.warning(f"⚠ {failed} model(s) failed to download after {max_attempts} attempts each")
            self.logger.warning("Failed files:")
            for file_key in failed_files:
                self.logger.warning(f"  - {file_key}")
            self.logger.warning("")
            self.logger.warning("You can:")
            self.logger.warning("  1. Re-run the script to retry failed downloads")
            self.logger.warning("  2. Manually download from: https://huggingface.co/Aitrepreneur/FLX")
            self.logger.warning(f"  3. Check log file for details: {self.cfg.log_file.absolute()}")
            self.logger.warning("")
    
    def _synchronous_download_models(self) -> bool:
        """Synchronous model download (fallback when background thread not started)"""
        models_dir = self.cfg.install_dir / "models"
        total_files = sum(len(files) for files in self.cfg.models.values())
        downloaded = skipped = failed = 0
        failed_files = {}
        total_bytes_downloaded = 0
        max_file_attempts = 3
        start_time = time.time()
        
        self.logger.info(f"Downloading {total_files} model files from HuggingFace")
        self.logger.info("Note: Large models may take considerable time (GB+ each)")
        self.logger.info(f"Source: {self.cfg.huggingface_base}")
        self.logger.info("Features: auto-retry, resume capability, partial download cleanup")
        self.logger.info("")
        
        for model_type, file_list in self.cfg.models.items():
            target_dir = models_dir / model_type
            
            for filename in file_list:
                file_num = downloaded + skipped + failed + 1
                file_key = f"{model_type}/{filename}"
                self.logger.info(f"[{file_num}/{total_files}] {file_key}")
                
                target_file = target_dir / filename
                
                # Check if file already exists and is valid
                try:
                    status = self._validate_existing_model_file(target_file)
                    if status == 'skip':
                        skipped += 1
                        continue
                except Exception:
                    pass  # Continue to download
                
                # Construct download URL
                url = f"{self.cfg.huggingface_base}/{filename}"
                
                # Attempt download with helper method
                success, bytes_down = self._download_single_model_file(url, target_file, file_key, max_file_attempts)
                
                if success:
                    downloaded += 1
                    total_bytes_downloaded += bytes_down
                else:
                    self.logger.error(f"  Failed after {max_file_attempts} attempts")
                    failed += 1
                    failed_files[file_key] = failed_files.get(file_key, 0) + 1
        
        # Print summary
        self._print_model_download_summary(
            downloaded, skipped, failed,
            total_bytes_downloaded, time.time() - start_time,
            failed_files, max_file_attempts
        )
        
        if failed > 0:
            return True  # Don't block installation, just warn
        
        self.logger.info("All model files downloaded successfully!")
        return True
    
    def step_10_download_models(self) -> bool:
        """Wait for background model downloads to complete (or download synchronously if needed)"""
        self._print_step("Step 10: Download Models from HuggingFace")
        
        # Check for --skip-models flag
        if self.cfg.skip_models:
            self.logger.info("--skip-models flag detected: skipping model downloads")
            self.logger.info("Models can be downloaded manually from: https://huggingface.co/Aitrepreneur/FLX")
            return True
        
        # If no background thread was started, do synchronous download
        if self.background_download_thread is None:
            self.logger.info("Starting model downloads (no background thread available)...")
            return self._synchronous_download_models()
        
        # Check if background download already completed
        if self.download_complete.is_set():
            self.logger.info("Model downloads completed in background during installation!")
        else:
            self.logger.info("Waiting for background model downloads to complete...")
            self.logger.info("(Downloads started in parallel after step 2)")
            
            # Show live progress while waiting
            total_files = sum(len(files) for files in self.cfg.models.values())
            while not self.download_complete.wait(timeout=2):
                with self.download_lock:
                    downloaded = self.download_results['downloaded']
                    skipped = self.download_results['skipped']
                    total_gb = self.download_results['total_bytes'] / (1024**3)
                    completed = downloaded + skipped
                self.logger.info(f"  Progress: {completed}/{total_files} files ({total_gb:.2f} GB downloaded)")
        
        # Wait for thread to fully finish
        if self.background_download_thread.is_alive():
            self.background_download_thread.join(timeout=10)
        
        # Display final summary from background results
        with self.download_lock:
            downloaded = self.download_results['downloaded']
            skipped = self.download_results['skipped']
            failed = self.download_results['failed']
            total_bytes = self.download_results['total_bytes']
            failed_files = self.download_results['failed_files']
            elapsed_time = self.download_results['end_time'] - self.download_results['start_time']
        
        self._print_model_download_summary(
            downloaded, skipped, failed,
            total_bytes, elapsed_time,
            failed_files, 3
        )
        
        if failed > 0:
            return True  # Don't block installation, just warn
        
        self.logger.info("All model files downloaded successfully!")
        return True
    
    def step_11_create_launch_scripts(self) -> bool:
        """Create launch scripts (batch and PowerShell)"""
        self._print_step("Step 11: Create Launch Scripts")
        
        # Create batch script
        batch_script = self.cfg.install_dir / "run_comfyui.bat"
        batch_content = f"""@echo off
echo Starting ComfyUI...
cd /d "%~dp0"
call {self.cfg.venv_name}\\Scripts\\activate.bat
python main.py --listen {self.cfg.COMFYUI_LOCAL_IP} --port {self.cfg.COMFYUI_LOCAL_PORT}
echo.
echo.
echo ComfyUI is running on port {self.cfg.COMFYUI_LOCAL_PORT}
echo.
echo.
echo -----------------------------------------------------------
echo - Do not close this window until you want to stop ComfyUI
echo -----------------------------------------------------------
echo - Open your browser to: http://{self.cfg.COMFYUI_LOCAL_IP}:{self.cfg.COMFYUI_LOCAL_PORT}
echo -----------------------------------------------------------
echo.
echo.
pause
"""
        
        batch_script.write_text(batch_content)
        self.logger.info(f"Created batch script: {batch_script}")
        
        # Create PowerShell script
        ps_script = self.cfg.install_dir / "run_comfyui.ps1"
        ps_content = f"""# ComfyUI PowerShell Launch Script
Write-Host "Starting ComfyUI..." -ForegroundColor Green
Set-Location $PSScriptRoot
& ".\\{self.cfg.venv_name}\\Scripts\\Activate.ps1"
python main.py --listen {self.cfg.COMFYUI_LOCAL_IP} --port {self.cfg.COMFYUI_LOCAL_PORT}
Write-Host "Press any key to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
"""
        
        ps_script.write_text(ps_content, encoding='utf-8')
        self.logger.info(f"Created PowerShell script: {ps_script}")
        self.logger.info("To run ComfyUI:")
        self.logger.info("  - Windows: double-click run_comfyui.bat")
        self.logger.info("  - PowerShell: .\\run_comfyui.ps1")
        self.logger.info("")
        self.logger.info("If PowerShell blocks the script, run once as Admin:")
        self.logger.info("  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned")
        
        return True
    
    def run_installation(self) -> int:
        """Run all installation steps with progress tracking"""
        steps = [
            ("Download ComfyUI Source", self.step_1_download_comfyui),
            ("Extract ZIP (Pure Python)", self.step_2_extract_comfyui),
            ("Setup Virtual Environment", self.step_3_setup_python_env),
            ("Install PyTorch", self.step_4_install_pytorch),
            ("Install ComfyUI Dependencies", self.step_5_install_comfyui_deps),
            ("Clone Custom Nodes", self.step_6_clone_custom_nodes),
            ("Install Node Dependencies", self.step_7_install_node_dependencies),
            ("Install Extra Dependencies", self.step_8_install_extras),
            ("Create Model Directories", self.step_9_create_models_dirs),
            ("Download Models from HuggingFace", self.step_10_download_models),
            ("Create Launch Scripts", self.step_11_create_launch_scripts),
        ]
        
        total_steps = len(steps)
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info(f"INSTALLATION PLAN: {total_steps} steps")
        self.logger.info(f"Installation directory: {self.cfg.install_dir}")
        self.logger.info("="*70)
        
        start_time = time.time()
        had_errors = False
        
        for idx, (step_name, step_func) in enumerate(steps, 1):
            if self.interrupted:
                self.logger.warning("[STOPPED] Installation interrupted by user")
                break
                
            step_start = time.time()
            self.logger.info("")
            self.logger.info(f"[STEP {idx}/{total_steps}] {step_name}")
            # Progress bar only to console (not logged to avoid clutter)
            print(f"Progress: [{'#' * idx}{'-' * (total_steps - idx)}] {(idx/total_steps)*100:.0f}%")
            
            try:
                success = step_func()
                elapsed = time.time() - step_start
                
                # Check for graceful exit from --clear-venv
                if success is False and self.cfg.clear_venv:
                    self.logger.info("Exiting after venv clear by request.")
                    break
                
                if not success:
                    had_errors = True
                    self.logger.warning(f"[WARNING] Completed with issues ({elapsed:.1f}s)")
                else:
                    self.logger.info(f"[SUCCESS] Completed in {elapsed:.1f}s")
                    
            except KeyboardInterrupt:
                self.logger.warning(f"[INTERRUPTED] User stopped during step {idx}")
                break
                
            except Exception as e:
                elapsed = time.time() - step_start
                self.logger.error(f"[ERROR] Error after {elapsed:.1f}s: {e}")
                import traceback
                error_trace = traceback.format_exc()
                self.logger.debug("Full traceback:")
                self.logger.debug(error_trace)
                # Clean up any registered cleanup tasks for this step
                self.error_handler.cleanup_all()
                self.logger.info("Continuing with next step...")
        
        total_time = time.time() - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        
        # Check for download failures
        download_failures = 0
        if self.background_download_thread is not None or not self.cfg.skip_models:
            with self.download_lock:
                download_failures = self.download_results.get('failed', 0)
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("INSTALLATION COMPLETE!" if not had_errors else "INSTALLATION FINISHED WITH WARNINGS")
        self.logger.info("="*70)
        self.logger.info(f"Total time: {minutes}m {seconds}s")
        self.logger.info(f"ComfyUI installed to: {self.cfg.install_dir}")
        self.logger.info("Launch scripts:")
        self.logger.info(f"  - Batch:      {self.cfg.install_dir / 'run_comfyui.bat'}")
        self.logger.info(f"  - PowerShell: {self.cfg.install_dir / 'run_comfyui.ps1'}")
        
        # Show prominent warning if model downloads failed
        if download_failures > 0:
            self.logger.info("")
            self.logger.info("="*70)
            self.logger.info("WARNING: SOME MODEL DOWNLOADS FAILED!")
            self.logger.info("="*70)
            self.logger.info(f"{download_failures} model file(s) failed to download.")
            self.logger.info("To retry failed downloads:")
            self.logger.info("  1. Re-run this script - it will skip already installed packages")
            self.logger.info("  2. Only failed downloads will be retried automatically")
            self.logger.info("  3. Check the log file for specific download errors")
            self.logger.info("="*70)
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("NEXT STEPS:")
        self.logger.info("="*70)
        self.logger.info(f"1. Run: {self.cfg.install_dir / 'run_comfyui.bat'} \n    or: {self.cfg.install_dir / 'run_comfyui.ps1'}")
        self.logger.info(f"2. Open browser: http://{self.cfg.COMFYUI_LOCAL_IP}:{self.cfg.COMFYUI_LOCAL_PORT}")
        self.logger.info("3. Start using ComfyUI with WAN Animate 2.2!")
        self.logger.info("="*70)
        
        return 0

# ============================================================================
# MAIN ENTRY POINT HELPERS
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser"""
    parser = argparse.ArgumentParser(
        description='ComfyUI WAN Animate 2.2 Automated Installer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    See README.md for examples, configuration options, and troubleshooting.

 !! To redownload models, simply run the script again. It will skip already installed packages and files !!
    """
    )
    parser.add_argument(
        '--python',
        type=str,
        metavar='VERSION',
        help='Python version to use for venv (e.g., "3.12", "3.11.5")'
    )
    parser.add_argument(
        '--comfyui',
        type=str,
        metavar='VERSION',
        help='ComfyUI version to install (e.g., "v0.3.65", "v0.3.66")'
    )
    parser.add_argument(
        '--torch',
        type=str,
        metavar='VERSION',
        help='PyTorch CUDA version to install (e.g., "2.8.0+cu128", "2.8.0+cu124")'
    )
    parser.add_argument(
        '--path',
        type=str,
        metavar='PATH',
        help='Custom installation directory path'
    )
    parser.add_argument(
        '--venv',
        type=str,
        metavar='NAME',
        help='Custom virtual environment folder name (default: "venv")'
    )
    parser.add_argument(
        '--clear-venv',
        action='store_true',
        dest='clear_venv',
        help='Clear the virtual environment only (does not reinstall packages, exits after clearing)'
    )
    parser.add_argument(
        '--reinstall-venv',
        action='store_true',
        dest='reinstall_venv',
        help='Clear the virtual environment and reinstall all packages (Not ComfyUI or models)'
    )
    parser.add_argument(
        '--upgrade-venv',
        action='store_true',
        dest='upgrade_venv',
        help='Upgrade the virtual python environment'
    )
    parser.add_argument(
        '--upgrade-deps',
        action='store_true',
        dest='upgrade_deps',
        help='Upgrade core dependencies (pip, setuptools) to the latest version in PyPI'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Do not use pip cache (applies to pip installs only, not ComfyUI or models)'
    )
    parser.add_argument(
        '--skip-models',
        action='store_true',
        dest='skip_models',
        help='Install without model downloads'
    )
    parser.add_argument(
        '--reinstall-comfyui',
        action='store_true',
        dest='reinstall_comfyui',
        help='Force re-download and re-extract ComfyUI (overwrites core files, preserves venv/models/custom_nodes)'
    )
    return parser

def validate_arguments(args) -> Optional[int]:
    """Validate command-line arguments. Returns error code if invalid, None if valid."""
    # Validate venv arguments - these flags cannot be combined with each other
    venv_flags_used = sum([args.clear_venv, args.reinstall_venv, args.upgrade_venv, args.upgrade_deps])
    if venv_flags_used > 1:
        print("ERROR: Virtual environment management flags (--clear-venv, --reinstall-venv, --upgrade-venv, --upgrade-deps) "
              "cannot be combined with each other. Use only one at a time.", file=sys.stderr)
        return 1
    
    # Check if venv management flags are used with installation arguments
    venv_management_mode = args.clear_venv or args.reinstall_venv or args.upgrade_venv or args.upgrade_deps
    other_install_args = (
        args.python is not None or 
        args.comfyui is not None or 
        args.torch is not None or 
        args.path is not None or 
        args.venv is not None
    )
    
    # Venv management flags can only be used with --no-cache, not with other installation args
    if venv_management_mode and other_install_args:
        print("ERROR: Virtual environment management flags (--clear-venv, --reinstall-venv, --upgrade-venv, --upgrade-deps) "
              "can only be combined with --no-cache, not with installation arguments.", file=sys.stderr)
        return 1
    
    # Early Windows-only check
    if platform.system() != "Windows":
        print("ERROR: This installer supports Windows 10/11 only due to Triton Windows requirements.", file=sys.stderr)
        return 1
    
    return None  # All validations passed

def build_config_from_args(args) -> Optional[Config]:
    """Build Config object from parsed arguments. Returns None on error."""
    # Build Config object with CLI overrides
    config_params = {
        'no_cache': args.no_cache,
        'clear_venv': args.clear_venv,
        'reinstall_venv': args.reinstall_venv,
        'upgrade_venv': args.upgrade_venv,
        'upgrade_deps': args.upgrade_deps,
        'skip_models': args.skip_models,
        'reinstall_comfyui': args.reinstall_comfyui,
    }
    
    # Apply CLI overrides for installation parameters
    if args.python:
        config_params['python_version'] = args.python
    if args.comfyui:
        config_params['comfyui_version'] = args.comfyui
    if args.torch:
        config_params['pytorch_version'] = args.torch
    if args.path:
        config_params['install_path'] = Path(args.path)
    if args.venv:
        config_params['venv_name'] = args.venv
    
    # Create Config object
    try:
        config = Config(**config_params)
    except Exception as e:
        print(f"ERROR: Failed to create configuration: {e}", file=sys.stderr)
        return None
    
    # Validate install path early
    try:
        if config.install_path is not None:
            parent = Path(config.install_path).resolve().parent
            if not parent.exists():
                print(f"ERROR: Parent directory does not exist: {parent}", file=sys.stderr)
                return None
            if not os.access(parent, os.W_OK):
                print(f"ERROR: Parent directory not writable: {parent}", file=sys.stderr)
                return None
    except Exception as e:
        print(f"ERROR: Could not validate install path: {e}", file=sys.stderr)
        return None
    
    return config

def display_configuration(config: Config, logger: logging.Logger):
    """Display configuration summary to user"""
    logger.info("="*70)
    logger.info("ComfyUI WAN Animate 2.2 - Installation Script")
    logger.info("="*70)
    logger.info(f"Python version: {config.python_version or 'current'}")
    logger.info(f"ComfyUI version: {config.comfyui_version}")
    logger.info(f"PyTorch version: {config.pytorch_version}")
    logger.info(f"Install directory: {config.install_dir}")
    logger.info(f"Virtual environment: {config.venv_name}")
    if config.no_cache:
        logger.info("Mode: NO CACHE (pip installs without cache)")
    if config.clear_venv:
        logger.info("Mode: CLEAR VENV ONLY (no reinstall)")
    if config.reinstall_venv:
        logger.info("Mode: REINSTALL VENV (full reinstall)")
    if config.upgrade_venv:
        logger.info("Mode: UPGRADE VENV (Python version upgrade)")
    if config.upgrade_deps:
        logger.info("Mode: UPGRADE DEPS (pip/setuptools upgrade)")
    if config.skip_models:
        logger.info("Mode: SKIP MODELS (no model downloads)")
    if config.reinstall_comfyui:
        logger.info("Mode: REINSTALL COMFYUI (force re-download and re-extraction)")
    logger.info("="*70)

def main(argv=None) -> int:
    """Main entry point for the installer"""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args(argv)
    
    # Validate arguments
    error_code = validate_arguments(args)
    if error_code is not None:
        return error_code
    
    # Build configuration
    config = build_config_from_args(args)
    if config is None:
        return 1
    
    # Create installer (this will setup logging and signal handlers)
    installer = ComfyUIInstaller(config)
    
    # Display configuration summary
    display_configuration(config, installer.logger)
    
    # Run prerequisite checks
    checker = PrerequisiteChecker(config, installer.logger)
    if not checker.check_all():
        installer.logger.error("Prerequisites check failed. Please fix the issues above.")
        return 1
    
    # Run installation
    return installer.run_installation()

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInstallation interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
