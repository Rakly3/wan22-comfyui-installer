# Never blindly trust any installer scripts!
I have no control over the software this installer downloads. You shouldn't even trust this script wthout reviewing it!

# ComfyUI WAN Animate 2.2 - Automated Installer

> Fully automated Python installation script for ComfyUI with WAN Animate 2.2 support, featuring parallel model downloads, CUDA optimization, and comprehensive error handling.

<br>

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Models](#-models)
- [System Requirements](#-system-requirements)
- [Command Line Arguments](#-command-line-arguments)
- [Examples](#examples)
- [Virtual Environment Management](#-virtual-environment-management)
- [Running ComfyUI with WAN 2.2](#running-comfyui-with-wan-22)
- [Common Tasks](#-common-tasks)
  - [Skipping Model Downloads](#-skipping-model-downloads)
  - [Re-running After Failures](#re-running-after-failures)
  - [Fixing Broken ComfyUI Installation](#fixing-broken-comfyui-installation)
  - [Fresh Virtual Environment Installation](#fresh-virtual-environment-installation)
- [Troubleshooting](#troubleshooting)
- [Resources](#-resources)
- [License](#-license)

<br>

## 🚀 Quick Start

```bash
# Basic installation (recommended)
python install_comfyui-wan22.py

# Show all options
python install_comfyui-wan22.py --help
```

### default settings:
 1. Validates prerequisites (Windows, Python, Git, internet, disk space)
 2. Download & extract ComfyUI v0.3.65 source
 3. Downloads models in background
 4. Create Python virtual environment
 5. Install PyTorch 2.9.0 + CUDA 12.8
 6. Install ComfyUI dependencies
 7. Clone 8 WAN Animate 2.2 required custom nodes
 8. Install Triton Windows, SageAttention, hf_transfer
 9. Wait for model downloads to complete (~40GB)
 10. Create launch scripts (.bat & .ps1)

**Note:** *Everything is configurable in the script.*

<br>

## 📦 Models

| Category | Files | Size | Description |
|----------:|:-------:|------:|-------------|
| **clip_vision** | 1 | 1.2GB | CLIP vision encoder |
| **detection** | 4 | 3.6GB | Pose estimation & object detection models |
| **diffusion_models** | 1 | 16.5GB | Main WAN Animate 2.2 model (14B parameters, fp8) |
| **loras** | 2 | 2.1GB | Fine-tuning adapters (I2V, relight) |
| **sams** | 1 | 7.5GB | Segment Anything Model (SeC-4B) |
| **text_encoders** | 1 | 10.8GB | UMT5-XXL encoder |
| **vae** | 1 | 242MB | Video autoencoder |
| | **11 files** | ~40GB |

Downloaded automatically from [HuggingFace/Aitrepreneur/FLX](https://huggingface.co/Aitrepreneur/FLX)

<br>

## 💻 System Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Windows 10/11 x64 |
| **Python** | 3.9 or newer |
| **Git** | Installed and in PATH |
| **Internet** | Stable connectivity |
| **Storage** | 60GB+ free disk space |
| **GPU** | NVIDIA GPU with CUDA support |
| **Driver** | NVIDIA Driver 560 or newer |
| **Architecture** | RTX 20 series and newer |

<br>

## 🐍 Command Line Arguments

```bash
--python VERSION         # Python version for venv (e.g., "3.12", "3.11.5")
--comfyui VERSION        # ComfyUI version (e.g., "v0.3.65", "v0.3.66")
--torch VERSION          # PyTorch CUDA version (e.g., "2.9.0+cu130", "2.8.0+cu128")
--path PATH              # Installation directory
--venv NAME              # Custom venv folder name (default: "venv")
--no-cache               # Skip pip cache
--skip-models            # Skip model downloads

--reinstall-comfyui      # Force re-download ComfyUI and extract
--clear-venv             # Clear venv only (no reinstall, exits after clearing)
--reinstall-venv         # Clear venv and reinstall all packages
--upgrade-venv           # Upgrade Python version
--upgrade-deps           # Upgrade pip/setuptools only
```

**Note:** *Command-line arguments override script settings.*

<br>

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

<br>

## Running ComfyUI with WAN 2.2

After installation completes, launch ComfyUI:

### Batch Script
```bash
ComfyUI-0.3.65\run_comfyui.bat
```

### PowerShell Script
```powershell
.\ComfyUI-0.3.65\run_comfyui.ps1
```

### 🛠️ Option 3: Manually
```cmd
cd ComfyUI-0.3.65
venv\Scripts\activate.bat
python main.py --listen localhost --port 8188
```
<br>

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **CUDA required error** | Script requires CUDA-enabled PyTorch. Edit `PYTORCH_VERSION` to include `+cu###` (e.g., `2.9.0+cu130`) |
| **No NVIDIA GPU detected error** | Warning only - installation continues. Verify GPU with `nvidia-smi` command. Update NVIDIA drivers to 560+ for CUDA 13.0. |
| **SageAttention version mismatch error** | Edit `SAGEATTENTION_WHEEL_URL` to match your `PYTORCH_VERSION`. Check [SageAttention releases](https://github.com/woct0rdho/SageAttention/releases). |
| **Model downloads fail** | Re-run script (retries failed downloads only). Check log file for specific errors. Slow/unstable internet may require multiple runs. |
| **Prerequisites check failed error** | Check `comfyui_installation.log`. Common: Python < 3.9, Git not installed, < 60GB free space, no internet. |

<br>

## 📚 Resources

### Dependencies
- **Triton Windows:** https://github.com/woct0rdho/triton-windows/releases
- **SageAttention:** https://github.com/woct0rdho/SageAttention/releases
- **PyTorch:** https://pytorch.org/get-started/locally/
- **Wan 2.2:** https://github.com/Wan-Video/Wan2.2
- **Models:** https://huggingface.co/Aitrepreneur/FLX
- **AItrepeeneur:** https://www.youtube.com/watch?v=aTGAiZe6SXU


<br>

## 📄 License

This installer script is provided as-is. ComfyUI and all dependencies have their own licenses. Please review:

| Component | License |
|-----------|---------|
| **ComfyUI** | [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html) |
| **PyTorch** | [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause) |
| **Custom Nodes** | See respective repositories |

<br>

---
---

<br>

<div align="center">

**⭐ If this installer helped you, please consider giving it a star! ⭐**

</div>



