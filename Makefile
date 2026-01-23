# Makefile for shorts-maker-gpu setup

# Configuration
VENV_DIR = .venv
BIN_DIR = bin
OS_NAME = $(shell uname -s | tr '[:upper:]' '[:lower:]')
ARCH_NAME = $(shell uname -m)

# Detect Mamba/Conda
# If conda is in path, use it. Otherwise use local micromamba.
# We use ./$(VENV_DIR) to ensure it is treated as a local path.
ifneq ($(shell which conda),)
    MAMBA_EXE = conda
    MAMBA_CMD = env update --prefix ./$(VENV_DIR) --file environment.yml --prune
else ifneq ($(shell which micromamba),)
    MAMBA_EXE = micromamba
    MAMBA_CMD = create -y -p ./$(VENV_DIR) -f environment.yml
else
    MAMBA_EXE = ./$(BIN_DIR)/micromamba
    # We use 'create' which works for new envs (and typically handles updates/existing gracefully or we can check)
    MAMBA_CMD = create -y -p ./$(VENV_DIR) -f environment.yml
endif

# Use absolute paths for Python/Pip to avoid issues when cd-ing into subdirs
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
# Since conda doesn't provide them easily, we build them into the environment manually.
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
	# CMAKE_PREFIX_PATH ensures we find ffmpeg/nv-codec-headers in our .venv
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

# Install pip requirements (from requirements.txt)
install_pip_requirements: $(VENV_DIR)
	@echo "Installing pip requirements..."
	$(PIP) install -r requirements.txt

# Install steps 
install: $(VENV_DIR) install_pip_requirements install_decord
	@echo "----------------------------------------------------------------"
	@echo "Installation complete!"
	@echo "To activate the environment:"
	@echo "  overlay use .venv/bin/activate.nu    # Nushell"
	@echo "  source .venv/bin/activate            # Bash/Zsh"
	@echo ""
	@echo "To run AutoShorts:"
	@echo "  python run.py"
	@echo "----------------------------------------------------------------"

clean:
	rm -rf $(VENV_DIR)
	rm -rf $(DECORD_BUILD_DIR)
	rm -rf $(BIN_DIR)
