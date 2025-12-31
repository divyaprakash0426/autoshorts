FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    build-essential \
    cmake \
    pkg-config \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. FIX: Install FFmpeg 4.4.2
# This is the latest version with which Decord reliably compiles without errors
RUN conda update -n base -c defaults conda -y && \
    conda install -y -c conda-forge "ffmpeg=4.4.2"

# 3. Set up environment
ENV FFMPEG_BINARY=/opt/conda/bin/ffmpeg
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# Increase Decord EOF retry limit to better handle long 4K videos with slow tail retrieval
ENV DECORD_EOF_RETRY_MAX=65536
ENV DECORD_SKIP_TAIL_FRAMES=0

# Ensure NVIDIA driver capabilities include video for codecs
ENV NVIDIA_DRIVER_CAPABILITIES=all

# 4. Install codec headers
# Important: for the older ffmpeg it's better to use a pinned header version, but git master should also work
RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && \
    make install && \
    cd .. && rm -rf nv-codec-headers

# 5. Install NVIDIA driver libraries for linking
# We install "headless" driver libraries so we have the .so files for linking.
# At runtime, the NVIDIA Container Toolkit will mount the host driver's files over these.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvidia-decode-535 \
    libnvidia-encode-535 \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks, so CMake and Python can find the libraries (packages typically provide .so.1/.so.535)
RUN ln -sf /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so && \
    ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-encode.so

# 6. Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 7. Build Decord
RUN git clone --recursive https://github.com/dmlc/decord && \
    cd decord && \
    mkdir build && cd build && \
    cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_nvcuvid_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvcuvid.so && \
    make -j$(nproc) && \
    cd ../python && \
    python setup.py install && \
    cd /app && rm -rf decord

# 8. C++ library fix
RUN rm /opt/conda/lib/libstdc++.so.6 && \
    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6

# 9. Cleanup build-time dependencies
# Remove the installed NVIDIA libraries so the container uses the host mounted ones at runtime
RUN apt-get purge -y libnvidia-decode-535 libnvidia-encode-535 && \
    rm -f /usr/lib/x86_64-linux-gnu/libnvcuvid.so /usr/lib/x86_64-linux-gnu/libnvidia-encode.so && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["python", "shorts.py"]