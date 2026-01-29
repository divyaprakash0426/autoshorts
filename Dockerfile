# Base image: PyTorch 2.6 with CUDA 12.6 (upgrade to 2.10 via pip)
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    build-essential \
    cmake \
    pkg-config \
    wget \
    sox \
    # Playwright browser dependencies
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Install FFmpeg 4.4.2 (required for Decord compatibility)
RUN conda update -n base -c defaults conda -y && \
    conda install -y -c conda-forge "ffmpeg=4.4.2"

# 3. Set up environment
ENV FFMPEG_BINARY=/opt/conda/bin/ffmpeg
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV DECORD_EOF_RETRY_MAX=65536
ENV DECORD_SKIP_TAIL_FRAMES=0
ENV NVIDIA_DRIVER_CAPABILITIES=all

# 4. Install NV Codec Headers for NVENC support
RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && \
    make install && \
    cd .. && rm -rf nv-codec-headers

# 5. Install NVIDIA driver libraries for linking
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvidia-decode-535 \
    libnvidia-encode-535 \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for CMake and Python
RUN ln -sf /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so && \
    ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-encode.so

# 6. Upgrade PyTorch to 2.10 with CUDA 12.6
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir torchcodec

# 7. Install FlashAttention 2 (prebuilt wheel for torch 2.10)
# Using manylinux wheel for broader compatibility
RUN pip install --no-cache-dir \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.6.3+cu128torch2.10-cp311-cp311-linux_x86_64.whl

# 8. Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 9. Install Playwright browsers for PyCaps
RUN playwright install chromium

# 10. Build Decord with CUDA support
RUN git clone --recursive https://github.com/dmlc/decord && \
    cd decord && \
    mkdir build && cd build && \
    cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_nvcuvid_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvcuvid.so && \
    make -j$(nproc) && \
    cd ../python && \
    python setup.py install && \
    cd /app && rm -rf decord

# 11. C++ library fix for compatibility
RUN rm -f /opt/conda/lib/libstdc++.so.6 && \
    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6

# 12. Cleanup build-time NVIDIA libraries (use host-mounted ones at runtime)
RUN apt-get purge -y libnvidia-decode-535 libnvidia-encode-535 && \
    rm -f /usr/lib/x86_64-linux-gnu/libnvcuvid.so /usr/lib/x86_64-linux-gnu/libnvidia-encode.so && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# 13. Copy application code
COPY . .

# Create directories for input/output
RUN mkdir -p gameplay generated assets

# 14. Pre-download TTS model (optional - comment out to download on first run)
# RUN python -c "from src.tts_generator import download_model; download_model()"

# Verify installations
RUN python -c "import torch; import flash_attn; print(f'PyTorch {torch.__version__}, FlashAttention {flash_attn.__version__}')"

CMD ["python", "run.py"]