# ComfyUI with WAN Animate 2.2 - Automated Installer for Windows

**Fully automated Python installation script** for ComfyUI with WAN Animate 2.2 support, featuring parallel model downloads, CUDA optimization, and comprehensive error handling.

## Table of Contents

- [Quick Start](#quick-start)
- [Models](#models)
- [System Requirements](#system-requirements)
- [Command Line Arguments](#command-line-arguments)
  - [Installation Options](#installation-options)
  - [Virtual Environment Management](#virtual-environment-management)
- [Running ComfyUI](#running-comfyui)
- [Common Tasks](#common-tasks)
  - [Skipping Model Downloads](#skipping-model-downloads)
  - [Re-running After Failures](#re-running-after-failures)
  - [Fixing Broken ComfyUI Installation](#fixing-broken-comfyui-installation)
  - [Fresh Virtual Environment Installation](#fresh-virtual-environment-installation)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)
- [License](#license)

---

## Quick Start

```bash
# Basic installation (recommended)
python install_comfyui-wan22.py

# Show all options
python install_comfyui-wan22.py --help
```

**What happens (default settings):**
1. Validate prerequisites (Windows, Python, Git, internet, disk space)
2. Download & extract ComfyUI v0.3.65 source
3. Downloads models in background
4. Create Python virtual environment
5. Install PyTorch 2.9.0 + CUDA 13.0
6. Install ComfyUI dependencies
7. Clone 8 WAN Animate 2.2 required custom nodes
8. Install Triton Windows, SageAttention, hf_transfer
9. Wait for model downloads to complete (~40GB)
10. Create launch scripts (.bat & .ps1)

---

## Models

**11 files, ~40GB total**

Downloaded automatically from [HuggingFace/Aitrepreneur/FLX](https://huggingface.co/Aitrepreneur/FLX):

| Category | Files | Size | Description |
|----------|-------|------|-------------|
| **clip_vision** | 1 | 1.2GB | CLIP vision encoder |
| **detection** | 4 | 3.6GB | Pose estimation & object detection models |
| **diffusion_models** | 1 | 16.5GB | Main WAN Animate 2.2 model (14B parameters, fp8) |
| **loras** | 2 | 2.1GB | Fine-tuning adapters (I2V, relight) |
| **sams** | 1 | 7.5GB | Segment Anything Model (SeC-4B) |
| **text_encoders** | 1 | 10.8GB | UMT5-XXL encoder |
| **vae** | 1 | 242MB | Video autoencoder |

---

## System Requirements

- Windows 10/11 x64
- Python 3.9 or newer
- Git installed and in PATH
- Internet connectivity
- 60GB+ free disk space
- NVIDIA GPU with CUDA support
- NVIDIA Driver 560 or newer
- RTX 20 series and newer

---

## Command Line Arguments

**Note:** Command-line arguments override script settings.

```bash
--python VERSION         # Python version for venv (e.g., "3.12", "3.11.5")
--comfyui VERSION        # ComfyUI version (e.g., "v0.3.65", "v0.3.66")
--torch VERSION          # PyTorch CUDA version (e.g., "2.9.0+cu130", "2.8.0+cu128")
--path PATH              # Installation directory
--venv NAME              # Custom venv folder name (default: "venv")
--no-cache               # Skip pip cache
--skip-models            # Skip model downloads
--reinstall-comfyui      # Force re-download ComfyUI and extract
```

**Examples:**
```bash
# Use Python 3.12 with custom path
python install_comfyui-wan22.py --python 3.12 --path "D:\AI\ComfyUI"

# Different CUDA version
python install_comfyui-wan22.py --torch 2.8.0+cu128

# Custom venv name without cache
python install_comfyui-wan22.py --venv .venv --no-cache

# Skip models for quick testing
python install_comfyui-wan22.py --skip-models

# Fix corrupted ComfyUI installation (keeps venv/models)
python install_comfyui-wan22.py --reinstall-comfyui
```

### Virtual Environment Management

```bash
--clear-venv         # Clear venv only (no reinstall, exits after clearing)
--reinstall-venv     # Clear venv and reinstall all packages
--upgrade-venv       # Upgrade Python version
--upgrade-deps       # Upgrade pip/setuptools only
--no-cache           # Force redownload pip packages
--skip-models        # Skip model downloads
--reinstall-comfyui  # Force re-download ComfyUI and extract
```

**Examples:**
```bash
# Clear venv only
python install_comfyui-wan22.py --clear-venv

# Clear and reinstall virtual environment
python install_comfyui-wan22.py --reinstall-venv --no-cache

# Upgrade to Python 3.12
python install_comfyui-wan22.py --upgrade-venv --python 3.12
```

---

## Running ComfyUI

After installation completes, launch ComfyUI:

### Option 1: Batch Script
```bash
ComfyUI-0.3.65\run_comfyui.bat
```

### Option 2: PowerShell Script
```powershell
.\ComfyUI-0.3.65\run_comfyui.ps1
```

### Option 3: Manually
```bash
cd ComfyUI-0.3.65
venv\Scripts\activate.bat
python main.py --listen localhost --port 8188
```

**Access ComfyUI:**
- Default: http://localhost:8188

---

### Skipping Model Downloads

For quick testing or if you want to download models manually:

```bash
python install_comfyui-wan22.py --skip-models
```

Models can be manually downloaded from: https://huggingface.co/Aitrepreneur/FLX

### Re-running After Failures

If installation or model downloads fail:

```bash
# Try re-running the script
python install_comfyui-wan22.py
```

### Fixing Broken ComfyUI Installation

```bash
# Re-download ComfyUI and re-extract
python install_comfyui-wan22.py --reinstall-comfyui
```
### Fresh Virtual Environment Installation

```bash
# Clear venv and reinstall all packages
python install_comfyui-wan22.py --reinstall-venv

# Or manually delete the venv directory
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **"CUDA required" error** | Script requires CUDA-enabled PyTorch. Edit `PYTORCH_VERSION` to include `+cu###` (e.g., `2.9.0+cu130`) |
| **"No NVIDIA GPU detected"** | Warning only - installation continues. Verify GPU with `nvidia-smi` command. Update NVIDIA drivers to 560+ for CUDA 13.0. |
| **"SageAttention version mismatch"** | Edit `SAGEATTENTION_WHEEL_URL` to match your `PYTORCH_VERSION`. Check [SageAttention releases](https://github.com/woct0rdho/SageAttention/releases). |
| **Model downloads fail** | Re-run script (retries failed downloads only). Check log file for specific errors. Slow/unstable internet may require multiple runs. |
| **"Prerequisites check failed"** | Check `comfyui_installation.log`. Common: Python < 3.9, Git not installed, < 60GB free space, no internet. |
| **Corrupted ComfyUI files** | Run `python install_comfyui-wan22.py --reinstall-comfyui` to re-extract (preserves venv/models). |
| **"requirements.txt not found"** | ComfyUI extraction failed or incomplete. Run `python install_comfyui-wan22.py --reinstall-comfyui`. |
| **PowerShell script blocked** | Run as Admin: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| **Port 8188 already in use** | Edit `COMFYUI_LOCAL_PORT = 8000` (or other port) in script before install. |

---

## Resources

### Dependencies
- **Triton Windows:** https://github.com/woct0rdho/triton-windows/releases
- **SageAttention:** https://github.com/woct0rdho/SageAttention/releases
- **PyTorch:** https://pytorch.org/get-started/locally/

## License

This installer script is provided as-is. ComfyUI and all dependencies have their own licenses. Please review:
- ComfyUI: GPL-3.0
- PyTorch: BSD-3-Clause
- Individual custom nodes: See respective repositories
