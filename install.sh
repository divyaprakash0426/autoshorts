#!/usr/bin/env bash
# AutoShorts Installation Script
# Works on Linux, macOS, and Windows (via WSL/Git Bash)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  AutoShorts Installation"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect OS
OS_TYPE="$(uname -s)"
case "$OS_TYPE" in
    Linux*)     OS="linux";;
    Darwin*)    OS="macos";;
    MINGW*|MSYS*|CYGWIN*)    OS="windows";;
    *)          OS="unknown";;
esac

echo -e "${GREEN}Detected OS: $OS${NC}"
echo ""

# Check for CUDA (optional but recommended)
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    echo -e "${GREEN}✓ CUDA detected: $CUDA_VERSION${NC}"
else
    echo -e "${YELLOW}⚠ CUDA not found. GPU acceleration will not be available.${NC}"
    echo "  Install CUDA 12.x from: https://developer.nvidia.com/cuda-downloads"
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.10 or later.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | sed -n 's/Python \([0-9]*\.[0-9]*\).*/\1/p')
echo -e "${GREEN}✓ Python detected: $PYTHON_VERSION${NC}"

# Check Python version (need 3.10+)
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; }; then
    echo -e "${RED}✗ Python 3.10+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi

# Check for system dependencies
echo ""
echo "Checking system dependencies..."

# FFmpeg
if command -v ffmpeg &> /dev/null; then
    echo -e "${GREEN}✓ FFmpeg installed${NC}"
else
    echo -e "${YELLOW}⚠ FFmpeg not found${NC}"
    echo "  Install with:"
    case "$OS" in
        linux)
            echo "    sudo apt install ffmpeg  # Debian/Ubuntu"
            echo "    sudo pacman -S ffmpeg    # Arch"
            ;;
        macos)
            echo "    brew install ffmpeg"
            ;;
        windows)
            echo "    choco install ffmpeg  # or download from ffmpeg.org"
            ;;
    esac
fi

# Sox
if command -v sox &> /dev/null; then
    echo -e "${GREEN}✓ Sox installed${NC}"
else
    echo -e "${YELLOW}⚠ Sox not found (required for TTS)${NC}"
    echo "  Install with:"
    case "$OS" in
        linux)
            echo "    sudo apt install sox  # Debian/Ubuntu"
            echo "    sudo pacman -S sox    # Arch"
            ;;
        macos)
            echo "    brew install sox"
            ;;
        windows)
            echo "    choco install sox"
            ;;
    esac
fi

echo ""
read -p "Continue with installation? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    exit 0
fi

echo ""
echo "=========================================="
echo "  Step 1: Creating Virtual Environment"
echo "=========================================="

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
fi

# Activate venv
if [ "$OS" = "windows" ]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

echo ""
echo "=========================================="
echo "  Step 2: Installing PyTorch with CUDA"
echo "=========================================="

pip install --upgrade pip wheel setuptools

echo "Installing PyTorch 2.9.1 with CUDA 12.8..."
pip install torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "Installing torchcodec..."
pip install --no-deps torchcodec

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"

echo ""
echo "=========================================="
echo "  Step 3: Installing Python Dependencies"
echo "=========================================="

pip install -r requirements.txt

# Install Playwright browsers (required for PyCaps)
echo "Installing Playwright browsers..."
playwright install chromium

echo ""
echo "=========================================="
echo "  Step 4: Installing FlashAttention 2"
echo "=========================================="

PYVER=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
echo "Detected Python version: $PYVER"

case "$PYVER" in
    cp310)
        FLASH_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.7.4+cu128torch2.9-cp310-cp310-linux_x86_64.whl"
        ;;
    cp311)
        FLASH_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.7.4+cu128torch2.9-cp311-cp311-linux_x86_64.whl"
        ;;
    cp312)
        FLASH_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.7.4+cu128torch2.9-cp312-cp312-linux_x86_64.whl"
        ;;
    *)
        echo -e "${YELLOW}⚠ No prebuilt wheel for Python $PYVER, attempting source build...${NC}"
        MAX_JOBS=4 pip install flash-attn --no-build-isolation
        ;;
esac

if [ ! -z "$FLASH_WHEEL" ]; then
    pip install "$FLASH_WHEEL"
fi

# Verify
python -c "import flash_attn; print(f'FlashAttention {flash_attn.__version__} installed')"

echo ""
echo "=========================================="
echo "  Step 5: Installing Decord (GPU)"
echo "=========================================="

if command -v make &> /dev/null; then
    echo "Building Decord from source with CUDA support..."
    make install_decord
else
    echo -e "${YELLOW}⚠ Make not found. Skipping Decord build.${NC}"
    echo "  Install manually: make install_decord"
fi

echo ""
echo "=========================================="
echo "  Step 6: Verifying Installation"
echo "=========================================="

python << 'VERIFY'
import sys

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
except Exception as e:
    print(f"✗ PyTorch: {e}")
    sys.exit(1)

try:
    import torchcodec
    print(f"✓ TorchCodec available")
except Exception as e:
    print(f"✗ TorchCodec: {e}")
    sys.exit(1)

try:
    import flash_attn
    print(f"✓ FlashAttention {flash_attn.__version__}")
except Exception as e:
    print(f"✗ FlashAttention: {e}")
    sys.exit(1)

print("\n✓ All critical dependencies OK")
VERIFY

echo ""
echo "=========================================="
echo "  Step 7: Download AI Models"
echo "=========================================="
echo ""
echo "Downloading Qwen3-TTS VoiceDesign model (~2GB)..."
python -c "from src.tts_generator import download_model; download_model()"

echo -e "${GREEN}✓ TTS model downloaded successfully${NC}"

echo ""
echo "=========================================="
echo "  ✓ Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment:"
if [ "$OS" = "windows" ]; then
    echo "       .venv\\Scripts\\activate"
else
    echo "       source .venv/bin/activate"
fi
echo ""
echo "  2. Configure API keys in .env file"
echo ""
echo "  3. Run AutoShorts:"
echo "       python run.py"
echo ""
echo "  4. Or start the dashboard:"
echo "       ./bin/dashboard"
echo ""
