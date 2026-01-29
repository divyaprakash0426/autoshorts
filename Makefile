# Makefile for shorts-maker-gpu setup

# Configuration
VENV_DIR = .venv
BIN_DIR = bin
OS_NAME = $(shell uname -s | tr '[:upper:]' '[:lower:]')
ARCH_NAME = $(shell uname -m)

# FlashAttention wheel URLs (torch 2.10 + CUDA 12.8)
# See https://github.com/mjun0812/flash-attention-prebuild-wheels for other versions
FLASH_ATTN_WHEEL_CP310 = https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.6.3+cu128torch2.10-cp310-cp310-linux_x86_64.whl
FLASH_ATTN_WHEEL_CP311 = https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.6.3+cu128torch2.10-cp311-cp311-linux_x86_64.whl
FLASH_ATTN_WHEEL_CP312 = https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.6.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl

# Detect Mamba/Conda
ifneq ($(shell which conda),)
    MAMBA_EXE = conda
    MAMBA_CMD = env update --prefix ./$(VENV_DIR) --file environment.yml --prune
else ifneq ($(shell which micromamba),)
    MAMBA_EXE = micromamba
    MAMBA_CMD = create -y -p ./$(VENV_DIR) -f environment.yml
else
    MAMBA_EXE = ./$(BIN_DIR)/micromamba
    MAMBA_CMD = create -y -p ./$(VENV_DIR) -f environment.yml
endif

# Use absolute paths for Python/Pip
PYTHON = $(shell pwd)/$(VENV_DIR)/bin/python
PIP = $(shell pwd)/$(VENV_DIR)/bin/pip
DECORD_REPO = https://github.com/dmlc/decord
DECORD_BUILD_DIR = _build_decord
NVCC_PATH = $(shell which nvcc)

# Default target
all: install

# Ensure local bin directory exists
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Download Micromamba if not found globally
setup_mamba: $(BIN_DIR)
	@if [ "$(MAMBA_EXE)" = "./$(BIN_DIR)/micromamba" ] && [ ! -f "$(MAMBA_EXE)" ]; then \
		echo "Conda/Mamba not found. Downloading local Micromamba for $(OS_NAME)-$(ARCH_NAME)..."; \
		curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $(BIN_DIR) bin/micromamba; \
		mv $(BIN_DIR)/bin/micromamba $(BIN_DIR)/micromamba; \
		rmdir $(BIN_DIR)/bin; \
		chmod +x $(MAMBA_EXE); \
	fi

# Create/Update environment
$(VENV_DIR): setup_mamba environment.yml requirements.txt
	@echo "Creating/Updating environment in $(VENV_DIR) using $(MAMBA_EXE)..."
	$(MAMBA_EXE) $(MAMBA_CMD)

# Build dependencies
NV_CODEC_HEADERS_DIR = _build_nv_codec_headers

# Install NV Codec Headers (Required for Decord + CUDA)
install_nv_codec_headers:
	@echo "Cloning NV Codec Headers..."
	rm -rf $(NV_CODEC_HEADERS_DIR)
	git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git $(NV_CODEC_HEADERS_DIR)
	@echo "Installing NV Codec Headers to ./$(VENV_DIR)..."
	cd $(NV_CODEC_HEADERS_DIR) && make install PREFIX=$(shell pwd)/$(VENV_DIR)
	rm -rf $(NV_CODEC_HEADERS_DIR)

# Build and install decord with CUDA support
install_decord: $(VENV_DIR) install_nv_codec_headers
	@echo "Checking for NVCC..."
	@if [ -z "$(NVCC_PATH)" ]; then echo "Error: nvcc not found. Please install CUDA toolkit."; exit 1; fi
	@echo "Found NVCC at: $(NVCC_PATH)"
	
	@echo "Cloning Decord..."
	rm -rf $(DECORD_BUILD_DIR)
	git clone --recursive $(DECORD_REPO) $(DECORD_BUILD_DIR)
	
	@echo "Building Decord Shared Library..."
	mkdir -p $(DECORD_BUILD_DIR)/build
	cd $(DECORD_BUILD_DIR)/build && cmake .. \
		-DUSE_CUDA=ON \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CUDA_COMPILER=$(NVCC_PATH) \
		-DCMAKE_INSTALL_PREFIX=$(VENV_DIR) \
		-DCMAKE_PREFIX_PATH=$(shell pwd)/$(VENV_DIR)
	cd $(DECORD_BUILD_DIR)/build && make -j$(shell nproc)
	
	@echo "Installing Decord Python Config..."
	cd $(DECORD_BUILD_DIR)/python && $(PYTHON) setup.py install
	
	@echo "Manually copying libdecord.so to site-packages (fix for loading error)..."
	cp $(DECORD_BUILD_DIR)/build/libdecord.so $(shell pwd)/$(VENV_DIR)/lib/python3.10/site-packages/decord/
	
	@echo "Cleaning up..."
	rm -rf $(DECORD_BUILD_DIR)

# Install PyTorch with CUDA 12.6 support
install_torch: $(VENV_DIR)
	@echo "Installing PyTorch 2.10 with CUDA 12.6..."
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
	@echo "Installing torchcodec for torchaudio video support..."
	$(PIP) install torchcodec

# Install FlashAttention 2 from prebuilt wheel
install_flash_attn: $(VENV_DIR)
	@echo "Installing FlashAttention 2 (prebuilt wheel for torch 2.10)..."
	@PYVER=$$($(PYTHON) -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"); \
	echo "Detected Python version: $$PYVER"; \
	if [ "$$PYVER" = "cp310" ]; then \
		$(PIP) install $(FLASH_ATTN_WHEEL_CP310); \
	elif [ "$$PYVER" = "cp311" ]; then \
		$(PIP) install $(FLASH_ATTN_WHEEL_CP311); \
	elif [ "$$PYVER" = "cp312" ]; then \
		$(PIP) install $(FLASH_ATTN_WHEEL_CP312); \
	else \
		echo "Warning: No prebuilt flash-attn wheel for Python $$PYVER"; \
		echo "Attempting to build from source (this may take a while)..."; \
		MAX_JOBS=4 $(PIP) install flash-attn --no-build-isolation; \
	fi
	@echo "Verifying FlashAttention installation..."
	$(PYTHON) -c "import flash_attn; print(f'FlashAttention {flash_attn.__version__} installed successfully')"

# Install pip requirements (from requirements.txt)
install_pip_requirements: $(VENV_DIR)
	@echo "Installing pip requirements..."
	$(PIP) install -r requirements.txt

# Download TTS model for offline use
download_tts_model: $(VENV_DIR)
	@echo "Downloading Qwen3-TTS VoiceDesign model..."
	$(PYTHON) -c "from src.tts_generator import download_model; download_model()"

# Install system dependencies (requires sudo)
install_system_deps:
	@echo "Installing system dependencies..."
	@if command -v pacman >/dev/null 2>&1; then \
		sudo pacman -S --noconfirm sox ffmpeg; \
	elif command -v apt-get >/dev/null 2>&1; then \
		sudo apt-get update && sudo apt-get install -y sox ffmpeg; \
	elif command -v dnf >/dev/null 2>&1; then \
		sudo dnf install -y sox ffmpeg; \
	else \
		echo "Please install sox and ffmpeg manually for your distribution"; \
	fi

# Full install (all components)
install: $(VENV_DIR) install_torch install_pip_requirements install_flash_attn install_decord
	@echo "----------------------------------------------------------------"
	@echo "Installation complete!"
	@echo ""
	@echo "To activate the environment:"
	@echo "  overlay use .venv/bin/activate.nu    # Nushell"
	@echo "  source .venv/bin/activate            # Bash/Zsh"
	@echo ""
	@echo "Optional: Download TTS model for offline use:"
	@echo "  make download_tts_model"
	@echo ""
	@echo "To run AutoShorts:"
	@echo "  python run.py"
	@echo ""
	@echo "To run the dashboard:"
	@echo "  ./bin/dashboard"
	@echo "----------------------------------------------------------------"

# Quick install (skip decord build - use if you have decord already)
install_quick: $(VENV_DIR) install_torch install_pip_requirements install_flash_attn
	@echo "Quick installation complete (without Decord build)"

# TTS-only install (for adding TTS to existing setup)
install_tts: install_torch install_pip_requirements install_flash_attn download_tts_model
	@echo "TTS installation complete with FlashAttention 2"

clean:
	rm -rf $(VENV_DIR)
	rm -rf $(DECORD_BUILD_DIR)
	rm -rf $(BIN_DIR)

.PHONY: all install install_quick install_tts install_torch install_flash_attn install_pip_requirements install_decord install_nv_codec_headers install_system_deps download_tts_model setup_mamba clean dashboard

dashboard:
	bash ./bin/dashboard
